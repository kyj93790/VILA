import torch
from transformers import Trainer

from .base import BaseTrainer


class FT(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        retain_data = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_data[0],
            "attention_mask": retain_data[1],
            "labels": retain_data[2],
        }

        outputs = model(**retain_inputs)

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
    

