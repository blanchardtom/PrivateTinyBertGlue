from typing import Any, Dict, Union
import torch
import torch.nn as nn 

def compute_loss(model, inputs):
    """
    Pass the inputs into the model, computes the loss and the predictions if the model is in evaluation mode.

    Args:
        model (:obj:`nn.Module`):
            The model to evaluate.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
    
    Return:
        Union[
            (
                (loss (:obj:`torch.Tensor`), predictions (:obj:`torch.Tensor`), labels (:obj:`torch.Tensor`)),
                (loss (:obj:`torch.Tensor`)
            )       
        ]
    """
    outputs = model(**inputs)

    logits = outputs["logits"]
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, inputs["labels"])
    
    if model.training :
        return loss
    else :
        predictions = outputs.logits.argmax(dim=-1)
        return loss, predictions, inputs["labels"]


def training_step(model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
                  optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (:obj:`nn.Module`):
            The model to train.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
        optimizer (torch.optim.Optimizer): 
            Optimizer instance for the training loop.
        lr_scheduler (torch.optim.lr_scheduler): 
            LR scheduler instance for the training loop.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.

    Return:
        :obj:`torch.Tensor`: The tensor with training loss on this batch.
    """
    model.train()
    loss= compute_loss(model, inputs)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    return loss.detach()


def eval_step(model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (:obj:`nn.Module`):
            The model to train.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
        optimizer (torch.optim.Optimizer): 
            Optimizer instance for the training loop.
        lr_scheduler (torch.optim.lr_scheduler): 
            LR scheduler instance for the training loop.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.

    Return:
        loss (:obj:`torch.Tensor`),
        predictions (:obj:`torch.Tensor`),
        labels (:obj:`torch.Tensor`)
    """
    model.eval()
    model.zero_grad()
    loss, predictions, labels = compute_loss(model, inputs)


    return loss.detach(), predictions, labels