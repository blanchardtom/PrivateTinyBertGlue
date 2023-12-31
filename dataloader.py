from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import BertTokenizerFast, BertTokenizer


def get_dataloader(task:str, model_checkpoint:str, split:str, dataloader_drop_last:bool=True, shuffle:bool=False,
                   batch_size:int=16, dataloader_num_workers:int=0, dataloader_pin_memory:bool=True) -> DataLoader:
    """To create encoded dataset dataloader for a given GLUE task.

    Args:
        task (str): GLUE task.
        model_checkpoint (str): tokenizer restoring model_checkpoint.
        split (str): "train", "validation", "test".
        dataloader_drop_last (bool, optional): Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size) or not. Defaults to True.
        batch_size (int): Number of samples in each batch.
        dataloader_num_workers (int, optional): Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process. Defaults to 0.
        dataloader_pin_memory (bool, optional): Whether you want to pin memory in data loaders or not. Defaults to True.

    Returns:
        dataloader(DataLoader): A tokenized and encoded dataloader.
    """
    task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    }

    sentence1_key, sentence2_key = task_to_keys[task]

    # padding for max_length = 502 because of the max token limit of 512 of BERT models - num_worker_tokens 
    # used for prefix and soft-prompting
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, padding='max_length', max_length=502)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding='max_length', max_length=502)
    
    tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer, max_length=502)

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    
    columns_to_return = ['input_ids', 'label', 'attention_mask']
    encoded_dataset.set_format(type='torch', columns=columns_to_return)

    if (split == "validation" or split == "test") and task == "mnli":
        split = "validation_matched"
    if (split == "validation" or split == "test") and task == "mnli-mm":
        split = "validation_mismatched"
    
    dataloader = DataLoader(
                    encoded_dataset[split],
                    shuffle=shuffle,
                    batch_size=batch_size,
                    collate_fn=data_collator,
                    drop_last=dataloader_drop_last,
                    num_workers=dataloader_num_workers,
                    pin_memory=False,
    )
    
    return dataloader