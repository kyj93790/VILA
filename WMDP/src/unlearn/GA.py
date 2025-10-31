import random
import torch
import numpy as np

from torch import tensor
from transformers import Trainer

from .base import BaseTrainer

from torch import nn
from typing import Dict, Union, Any

from torch.backends.cuda import SDPBackend, sdp_kernel
from torchmetrics.utilities.data import to_onehot
from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_format
from torchmetrics.functional.classification.hinge import (
    _multiclass_hinge_loss_arg_validation, 
    _multiclass_hinge_loss_tensor_validation,
    _hinge_loss_compute
)

def _custom_multiclass_hinge_loss_update(
    preds,
    target,
    alpha,
    squared,
    multiclass_mode = "crammer-singer"
):
    if not torch.all((preds >= 0) * (preds <= 1)):
        preds = preds.softmax(1)

    target = to_onehot(target, max(2, preds.shape[1])).bool()
    if multiclass_mode == "crammer-singer":
        margin = preds[target]
        margin -= torch.max(preds[~target].view(preds.shape[0], -1), dim=1)[0]
    else:
        target = target.bool()
        margin = torch.zeros_like(preds)
        margin[target] = preds[target]
        margin[~target] = -preds[~target]

    measures = alpha + margin
    measures = torch.clamp(measures, 0)

    if squared:
        measures = measures.pow(2)

    total = tensor(target.shape[0], device=target.device)
    return measures.sum(dim=0), total

def multiclass_hinge_loss(
    preds,
    target,
    num_classes,
    alpha = 1.0,
    squared = False,
    multiclass_mode = "crammer-singer",
    ignore_index = None,
    validate_args = True,
):
    if validate_args:
        _multiclass_hinge_loss_arg_validation(num_classes, squared, multiclass_mode, ignore_index)
        _multiclass_hinge_loss_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index, convert_to_labels=False)
    measures, total = _custom_multiclass_hinge_loss_update(
        preds, 
        target, 
        alpha,
        squared, 
        multiclass_mode,
    )
    return _hinge_loss_compute(measures, total)

class GA(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_data = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_data[0],
            "attention_mask": forget_data[1],
            "labels": forget_data[2],
        }

        outputs = model(**forget_inputs)

        loss = -outputs.loss

        return (loss, outputs) if return_outputs else loss


class GA_FT(GA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_data = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_data[0],
            "attention_mask": forget_data[1],
            "labels": forget_data[2],
        }

        retain_data = inputs["retain"]

        retain_inputs = {
            "input_ids": retain_data[0],
            "attention_mask": retain_data[1],
            "labels": retain_data[2],
        }

        forget_outputs = model(**forget_inputs)
        retain_outputs = model(**retain_inputs)

        loss = -forget_outputs.loss + self.gamma * retain_outputs.loss
        return (loss, forget_outputs) if return_outputs else loss




class IHL_FT(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_data = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_data[0],
            "attention_mask": forget_data[1],
            "labels": forget_data[2],
        }

        retain_data = inputs["retain"]

        retain_inputs = {
            "input_ids": retain_data[0],
            "attention_mask": retain_data[1],
            "labels": retain_data[2],
        }

        forget_outputs = model(**forget_inputs)
        retain_outputs = model(**retain_inputs)

        scores = forget_outputs.logits
        shift_logits = scores[..., :-1, :].contiguous().squeeze().view(-1, scores.size(-1)) # [BN, V]
        shift_labels = forget_inputs['labels'][..., 1:].contiguous().squeeze().view(-1) # [BN,]
        forget_loss = multiclass_hinge_loss(
            shift_logits[shift_labels != -100,:], # ignore pad tokens
            shift_labels[shift_labels != -100],
            shift_logits.size(-1),
        )

        loss = forget_loss + self.gamma * retain_outputs.loss
        return (loss, forget_outputs) if return_outputs else loss

class NPO_FT(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        forget_data = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_data[0],
            "attention_mask": forget_data[1],
            "labels": forget_data[2],
        }

        retain_data = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_data[0],
            "attention_mask": retain_data[1],
            "labels": retain_data[2],
        }

        outputs = model(**forget_inputs)
        current_forget_loss = outputs.loss

        with torch.no_grad():
            ref_outputs = self.infer_model(**forget_inputs)
            ref_forget_loss = ref_outputs.loss
        
        neg_log_ratios = current_forget_loss - ref_forget_loss

        retain_outputs = model(**retain_inputs)
        retain_loss = retain_outputs.loss
        
        forget_loss = - torch.nn.functional.logsigmoid(self.beta*neg_log_ratios).mean()*2/self.beta

        loss = forget_loss + self.gamma * retain_loss
        return (loss, outputs) if return_outputs else loss

