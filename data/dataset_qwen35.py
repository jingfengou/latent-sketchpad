import json
import os
import random
import re
from typing import Dict, List, Optional

import requests
import torch
from PIL import Image

from gen_utils import left_padding


IGNORE_INDEX = -100
BOI_TOKEN_QWEN = "<|vision_start|><|image_pad|><|vision_end|>"


def _keep_only_first_image_token(text: str) -> str:
    count = 0

    def _replace(match):
        nonlocal count
        count += 1
        return "<image>" if count == 1 else ""

    return re.sub(r"<image>", _replace, text)


def _remove_all_image_tokens(text: str) -> str:
    return re.sub(r"<image>", "", text)


def load_image(image_path: str, image_size: int = 448):
    try:
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
        return image
    except Exception:
        print(f"Error occurred when dealing with {image_path}")
        raise


class MultimodalQwen35Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        processor,
        json_file: List[str],
        image_dir: str,
        image_size: int = 448,
        stage1: bool = False,
        ignore_image: bool = False,
    ):
        self.image_dir = image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.processor = processor
        self.stage1 = stage1
        self.ignore_image = ignore_image
        self.data = []
        for file in json_file:
            with open(file, "r", encoding="utf-8") as handle:
                self.data.extend(json.load(handle))

    def __len__(self) -> int:
        return len(self.data)

    def _prepare_features(self, text: str, images):
        batch = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            images_kwargs={"do_resize": False},
        )
        batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)
        return batch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = dict(self.data[idx])

        if self.stage1 and random.random() < 0.5:
            item["input_text"] = _keep_only_first_image_token(item["input_text"])
            item["label_text"] = _remove_all_image_tokens(item["label_text"])
            item["label_img"] = []
            item["input_img"] = item["input_img"][:1]

        input_text = item["input_text"].replace("<image>", BOI_TOKEN_QWEN)
        label_text = item["label_text"].replace("<image>", BOI_TOKEN_QWEN)
        text = input_text + label_text
        prompt = input_text

        if isinstance(text, list):
            raw_text = self.tokenizer.eos_token.join(text)
        elif isinstance(text, str):
            raw_text = text + self.tokenizer.eos_token
        else:
            raise NotImplementedError(f"Unsupported text type: {type(text)!r}")

        input_images = [load_image(os.path.join(self.image_dir, path), image_size=self.image_size) for path in item["input_img"]]
        label_images = [load_image(os.path.join(self.image_dir, path), image_size=self.image_size) for path in item["label_img"]]

        text_dict = self._prepare_features(raw_text, input_images + label_images)
        prompt_dict = self._prepare_features(prompt, input_images)

        labels = text_dict["input_ids"].clone().squeeze(0)
        input_len = prompt_dict["input_ids"].shape[-1]
        labels[:input_len] = IGNORE_INDEX
        labels[labels == self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")] = IGNORE_INDEX
        if self.ignore_image:
            labels[labels == self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")] = IGNORE_INDEX

        output = {
            "input_ids": text_dict["input_ids"].squeeze(0),
            "attention_mask": text_dict["attention_mask"].squeeze(0),
            "labels": labels,
            "pixel_values": text_dict["pixel_values"],
            "image_grid_thw": text_dict["image_grid_thw"],
        }
        if "mm_token_type_ids" in text_dict:
            output["mm_token_type_ids"] = text_dict["mm_token_type_ids"].squeeze(0)
        return output


class MultimodalQwen35EvalDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, processor, json_file: str, image_dir: str, image_size: int = 448):
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_dir = image_dir
        self.image_size = image_size
        with open(json_file, "r", encoding="utf-8") as handle:
            self.data = json.load(handle)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        raw_text = item["input_text"].replace("<image>", BOI_TOKEN_QWEN)
        input_images = [load_image(os.path.join(self.image_dir, path), image_size=self.image_size) for path in item["input_img"]]
        batch = self.processor(
            text=raw_text,
            images=input_images,
            return_tensors="pt",
            images_kwargs={"do_resize": False},
        )
        batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)
        output = {
            "input_ids": batch["input_ids"].squeeze(0),
            "attention_mask": batch["attention_mask"].squeeze(0),
            "pixel_values": batch["pixel_values"],
            "image_grid_thw": batch["image_grid_thw"],
        }
        if "mm_token_type_ids" in batch:
            output["mm_token_type_ids"] = batch["mm_token_type_ids"].squeeze(0)
        return output


class DataCollatorForQwen35Inputs:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features if "labels" in feature]
        pixel_values = torch.cat([feature["pixel_values"] for feature in features], dim=0)
        image_grid_thw = torch.cat([feature["image_grid_thw"] for feature in features], dim=0)

        padded_input_ids = left_padding(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_labels = left_padding(labels, batch_first=True, padding_value=IGNORE_INDEX) if labels else None
        attention_mask = padded_input_ids.ne(self.pad_token_id)

        batch = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        if "mm_token_type_ids" in features[0]:
            mm_token_type_ids = [feature["mm_token_type_ids"] for feature in features]
            batch["mm_token_type_ids"] = left_padding(mm_token_type_ids, batch_first=True, padding_value=0)
        return batch
