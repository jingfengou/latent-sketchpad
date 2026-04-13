import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel


class Qwen3_5_VisionEncoder(Qwen3_5VisionModel):
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        outputs = super().forward(hidden_states, grid_thw)
        return outputs.last_hidden_state
