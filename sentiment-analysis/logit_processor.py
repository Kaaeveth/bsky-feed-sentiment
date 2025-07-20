from transformers import LogitsProcessor
import torch

class ConstrainedLogitProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: list[int]):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float('-inf'))
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = scores[:, token_id]
        return mask
