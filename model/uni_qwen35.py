import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GenerationConfig, LogitsProcessorList, Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.cache_utils import Cache
from transformers.generation.configuration_utils import CompileConfig
from transformers.generation.utils import logger
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5CausalLMOutputWithPast

from model.perceiver import PerceiverAR


class RegressionHeadPerceiver(nn.Module):
    def __init__(
        self,
        text_hidden_size: int,
        vision_hidden_size: int,
        spatial_scale: int,
        max_prefix_len: int = 8192,
        num_heads: int = 8,
        dropout: float = 0.1,
        cross_depth: int = 2,
        self_depth: int = 8,
    ) -> None:
        super().__init__()
        self.use_perceiver = os.environ.get("LATENT_SKETCHPAD_QWEN35_USE_PERCEIVER", "0") == "1"
        self.perceiver = PerceiverAR(
            dim=vision_hidden_size,
            depth=self_depth,
            dim_head=vision_hidden_size // num_heads,
            heads=num_heads,
            max_seq_len=32768,
            cross_attn_seq_len=max_prefix_len,
            cross_attn_dropout=dropout,
            perceive_depth=cross_depth,
        )
        self.mlp = nn.Sequential(
            nn.Linear(text_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, vision_hidden_size * spatial_scale),
        )
        self.dim = vision_hidden_size * spatial_scale
        self.vision_hidden_size = vision_hidden_size
        self.spatial_scale = spatial_scale

    def forward(self, x, prefix_mask=None, total_mask=None):
        out = torch.zeros(x.shape[0], x.shape[1], self.dim, device=x.device, dtype=x.dtype)
        out[total_mask] = self.mlp(x[total_mask])
        out = out.reshape(out.shape[0], -1, self.vision_hidden_size)
        if self.use_perceiver:
            out = self.perceiver(out, prefix_mask=prefix_mask.repeat_interleave(self.spatial_scale, dim=1))
        else:
            prefix_len = prefix_mask.shape[1] * self.spatial_scale
            out = out[:, prefix_len:]
        return out


@dataclass
class UniQwen35OutputWithImageLoss(Qwen3_5CausalLMOutputWithPast):
    image_loss: Optional[torch.FloatTensor] = None
    transformed_features: Optional[torch.FloatTensor] = None


class UniQwen35ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)
        self.spatial_scale = config.vision_config.spatial_merge_size ** 2
        self.flatten_size = config.vision_config.hidden_size * self.spatial_scale
        self.image_seq_len = 196
        self.max_img_cnt = 32
        self.max_prefix_len = self.image_seq_len * self.max_img_cnt
        self.image_token_index = config.image_token_id
        self.generation_type = os.environ.get("GENERATION_TYPE", "text_only")
        self.regression_head = RegressionHeadPerceiver(
            max_prefix_len=self.max_prefix_len * self.spatial_scale,
            text_hidden_size=config.text_config.hidden_size,
            vision_hidden_size=config.vision_config.hidden_size,
            spatial_scale=self.spatial_scale,
            num_heads=config.vision_config.num_heads,
            cross_depth=2,
            self_depth=8,
        )
        self.post_init()

    def _reset_perceiver_rotary_inv_freq(self) -> None:
        rotary = self.regression_head.perceiver.rotary_pos_emb
        dim = rotary.inv_freq.numel() * 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=rotary.inv_freq.device).float() / dim))
        rotary.inv_freq.data.copy_(inv_freq)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model._reset_perceiver_rotary_inv_freq()
        return model

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_context_mask(self, input_length, total_len=None):
        batch_size = input_length.shape[0]
        positions = torch.arange(total_len, device=input_length.device).unsqueeze(0).expand(batch_size, total_len)
        cutoffs = (total_len - input_length).unsqueeze(1)
        return positions >= cutoffs

    def get_vit_features(self, pixel_values, image_grid_thw, normalize: bool = True):
        vision_outputs = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        features = vision_outputs.last_hidden_state
        if pixel_values.ndim == 3:
            batch_size = image_grid_thw.shape[0]
            features = features.reshape(batch_size, -1, self.config.vision_config.hidden_size)
        return features

    def get_image_features(self, pixel_values, image_grid_thw):
        vision_outputs = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        features = vision_outputs.pooler_output
        if pixel_values.ndim == 3:
            batch_size = image_grid_thw.shape[0]
            features = features.reshape(batch_size, -1, self.config.vision_config.out_hidden_size)
        return features

    def project_vision_tokens(self, vit_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = vit_tokens.shape[0]
        orig_dtype = vit_tokens.dtype
        vit_tokens = vit_tokens.reshape(-1, self.config.vision_config.hidden_size)
        vit_tokens = vit_tokens.to(self.model.visual.merger.linear_fc1.weight.dtype)
        image_embeds = self.model.visual.merger(vit_tokens)
        image_embeds = image_embeds.reshape(batch_size, -1, self.config.vision_config.out_hidden_size)
        vit_tokens = vit_tokens.reshape(batch_size, -1, self.config.vision_config.hidden_size)
        return image_embeds.to(orig_dtype), vit_tokens.to(orig_dtype)

    def fill_image_embeds(self, input_ids, pixel_values, image_grid_thw):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.model.visual.patch_embed.proj.weight.dtype)
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            image_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

    def perceiver_forward(self, input_embeds, shifted_outputs, input_mask, label_length):
        batch_size = label_length.shape[0]
        input_length = input_mask.sum(dim=1)

        batch_context = torch.ones(
            batch_size,
            self.max_prefix_len,
            shifted_outputs.shape[-1],
            device=shifted_outputs.device,
            dtype=input_embeds.dtype,
        )
        context_mask = self.get_context_mask(input_length, self.max_prefix_len)
        batch_context[context_mask] = input_embeds[input_mask]
        max_label_len = int(label_length.max().item())
        batch_labels = torch.ones(
            batch_size,
            max_label_len,
            shifted_outputs.shape[-1],
            device=shifted_outputs.device,
            dtype=shifted_outputs.dtype,
        )
        out_mask = self.get_context_mask(label_length, total_len=max_label_len)
        batch_labels[out_mask] = shifted_outputs
        x = torch.cat([batch_context, batch_labels], dim=1)
        total_mask = torch.cat([context_mask, out_mask], dim=1)
        transformed_vit = self.regression_head(x, context_mask, total_mask)
        transformed_features, transformed_vit = self.project_vision_tokens(transformed_vit)
        return transformed_features.type_as(shifted_outputs), transformed_vit.type_as(shifted_outputs)

    def perceiver_inference(self, input_embeds, hidden_states, input_ids):
        prefix_image_mask = (input_ids == self.image_token_index).to(input_ids.device)
        eoi_mask = (input_ids == self.config.vision_end_token_id).to(input_ids.device)
        pos = torch.arange(input_ids.shape[-1], device=input_ids.device).unsqueeze(0)
        last_eoi_pos = (eoi_mask * pos).max(dim=1).values
        has_eoi = eoi_mask.any(dim=1)
        keep_mask = (pos < last_eoi_pos.unsqueeze(1)) | (~has_eoi.unsqueeze(1))
        prefix_image_mask &= keep_mask

        batch_size = input_embeds.shape[0]
        batch_context = torch.ones(
            batch_size,
            self.max_prefix_len,
            hidden_states.shape[-1],
            device=hidden_states.device,
            dtype=input_embeds.dtype,
        )
        input_length = prefix_image_mask.sum(dim=1)
        context_mask = self.get_context_mask(input_length, self.max_prefix_len)
        batch_context[context_mask] = input_embeds[prefix_image_mask]
        x = torch.cat([batch_context, hidden_states], dim=1)
        total_mask = torch.cat(
            [context_mask, torch.ones(hidden_states.shape[:2], dtype=context_mask.dtype, device=context_mask.device)], dim=1
        )
        transformed_vit = self.regression_head(x, context_mask, total_mask)
        transformed_features, transformed_vit = self.project_vision_tokens(transformed_vit)
        return transformed_features.type_as(input_embeds), transformed_vit.type_as(input_embeds)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, UniQwen35OutputWithImageLoss]:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            second_per_grid_ts=second_per_grid_ts,
            cache_position=cache_position,
            mm_token_type_ids=mm_token_type_ids,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        image_mask = (labels == self.image_token_index).to(labels.device) if labels is not None else None
        image_loss = None
        transformed_features = None
        if image_mask is not None and torch.any(image_mask):
            input_image_mask = (input_ids == self.image_token_index).to(input_ids.device).logical_and(~image_mask)
            if inputs_embeds is None:
                inputs_embeds = self.fill_image_embeds(input_ids, pixel_values, image_grid_thw)

            prompt_img_cnt = input_image_mask.sum(dim=1) // self.image_seq_len
            label_img_cnt = image_mask.sum(dim=1) // self.image_seq_len
            image_lengths = torch.stack([prompt_img_cnt, label_img_cnt], dim=1).reshape(-1)
            segment_ids = torch.arange(image_lengths.size(0), device=image_lengths.device)
            seg_masks = torch.repeat_interleave(segment_ids % 2 == 1, image_lengths)
            label_pixel_values = pixel_values.reshape(
                -1,
                self.image_seq_len * self.spatial_scale,
                pixel_values.size(-1),
            )[seg_masks]
            label_image_grid_thw = image_grid_thw[seg_masks]

            target_vit = self.get_vit_features(label_pixel_values, label_image_grid_thw)
            target_features = self.get_image_features(label_pixel_values, label_image_grid_thw)
            shifted_outputs = outputs.hidden_states[-1][:, :-1, :][image_mask[:, 1:]].clone()

            transformed_features, transformed_vit = self.perceiver_forward(
                inputs_embeds,
                shifted_outputs,
                input_image_mask,
                image_mask.sum(dim=1),
            )
            loss_type = kwargs.pop("loss_type", "mse")
            transformed_vit = transformed_vit.reshape(-1, target_vit.shape[-1])
            transformed_features = transformed_features.reshape(-1, target_features.shape[-1])
            target_vit = target_vit.reshape(-1, target_vit.shape[-1])
            target_features = target_features.reshape(-1, target_features.shape[-1])

            if loss_type == "mse":
                image_loss = nn.functional.mse_loss(transformed_vit, target_vit)
            elif loss_type == "l1":
                image_loss = nn.functional.l1_loss(transformed_vit, target_vit)
                image_loss += nn.functional.l1_loss(transformed_features, target_features)
            elif loss_type == "cosine":
                cos_sim_vit = nn.functional.cosine_similarity(transformed_vit, target_vit, dim=-1)
                cos_sim_features = nn.functional.cosine_similarity(transformed_features, target_features, dim=-1)
                image_loss = 0.5 * (1.0 - cos_sim_vit).mean() + 0.5 * (1.0 - cos_sim_features).mean()

        return UniQwen35OutputWithImageLoss(
            image_loss=image_loss,
            transformed_features=transformed_features,
            loss=image_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer=None,
        **model_kwargs,
    ):
        if self.generation_type == "text_only":
            return super()._sample(
                input_ids=input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        pad_token_id = generation_config._pad_token_tensor
        do_sample = generation_config.do_sample
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        output_hidden_states = generation_config.output_hidden_states
        return_dict_in_generate = generation_config.return_dict_in_generate

        boi_id = self.config.vision_start_token_id
        eoi_id = self.config.vision_end_token_id
        image_id = self.image_token_index

        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            if self.config._attn_implementation == "flash_attention_2" and getattr(
                model_kwargs.get("past_key_values"), "is_compileable", False
            ):
                if generation_config.compile_config is None:
                    generation_config.compile_config = CompileConfig(fullgraph=False)
                elif generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use CompileConfig(fullgraph=True). "
                        "Overriding to fullgraph=False."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = generation_config.prefill_chunk_size is None
        image_grid_thw = model_kwargs["image_grid_thw"]
        inputs_embeds = self.fill_image_embeds(input_ids, model_kwargs.pop("pixel_values", None), image_grid_thw)
        inputs_vit_feats = torch.randn(
            batch_size,
            inputs_embeds.shape[1],
            self.spatial_scale,
            self.config.vision_config.hidden_size,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        in_image = input_ids.new_zeros(batch_size, dtype=torch.bool)
        output_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        next_token_emb = None
        image_hidden_states = torch.randn(
            batch_size,
            self.image_seq_len,
            self.config.text_config.hidden_size,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs.update({"output_attentions": False, "output_hidden_states": output_hidden_states or True})
            model_inputs.pop("input_ids", None)
            model_inputs.update({"inputs_embeds": next_token_emb} if next_token_emb is not None else {"inputs_embeds": inputs_embeds})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            token_emb = self.get_input_embeddings()(next_tokens)
            token_vit_feats = torch.randn(
                batch_size,
                self.spatial_scale,
                self.config.vision_config.hidden_size,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )

            if in_image.any():
                img_seq_len = (input_ids == self.image_token_index).logical_and(output_mask).sum(dim=1)
                boi_cnt = (input_ids == boi_id).logical_and(output_mask).sum(dim=1)
                fin_image = (img_seq_len // self.image_seq_len) == boi_cnt
                if fin_image.any():
                    next_tokens[fin_image] = eoi_id
                    token_emb[fin_image] = self.get_input_embeddings()(next_tokens[fin_image])
                in_image &= ~fin_image
                if in_image.any():
                    next_tokens[in_image] = image_id
                    last_hidden = outputs.hidden_states[-1][:, -1, :]
                    batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
                    image_hidden_states[batch_idx, img_seq_len % self.image_seq_len, :] = last_hidden
                    proj, vit_feat = self.perceiver_inference(inputs_embeds, image_hidden_states, input_ids)
                    proj = proj[batch_idx, img_seq_len % self.image_seq_len, :]
                    vit_feat = vit_feat[
                        batch_idx,
                        (img_seq_len % self.image_seq_len) * self.spatial_scale :
                        (img_seq_len % self.image_seq_len + 1) * self.spatial_scale,
                        :,
                    ]
                    token_emb[in_image] = proj[in_image]
                    token_vit_feats[in_image] = vit_feat[in_image]

            is_boi = next_tokens == boi_id
            in_image |= is_boi

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            inputs_embeds = torch.cat([inputs_embeds, token_emb.unsqueeze(1)], dim=1)
            output_mask = torch.cat(
                [output_mask, torch.ones(batch_size, 1, dtype=torch.bool, device=output_mask.device)], dim=1
            )
            next_token_emb = token_emb.unsqueeze(1)
            inputs_vit_feats = torch.cat([inputs_vit_feats, token_vit_feats.unsqueeze(1)], dim=1)

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            del outputs

        if streamer is not None:
            streamer.end()

        valid_img_seq_len = ((input_ids == self.image_token_index).logical_and(output_mask).sum(dim=1) // self.image_seq_len)
        valid_img_seq_len = valid_img_seq_len * self.image_seq_len
        image_mask = (input_ids == self.image_token_index).logical_and(output_mask)
        image_embeds = inputs_embeds[image_mask][: valid_img_seq_len.sum()].view(-1, self.image_seq_len, inputs_embeds.shape[-1])
        image_vit_feats = inputs_vit_feats[image_mask][: valid_img_seq_len.sum()].view(-1, self.image_seq_len * self.spatial_scale, self.config.vision_config.hidden_size)
        return input_ids, image_embeds, image_vit_feats
