# encoding = "utf-8"

import os
import torch
import transformers
# from torch.utils.data import Dataset
# from transformers import Trainer
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from chat_template import *
from iterative_trainer import IterativeTrainer
from data_utils import *


DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./Models/Meta-Llama-3-8B")
    prompt_template: str = field(default="llama-3")

@dataclass
class DataArguments:
    data_path: str = field(default="./data/alpaca_data.json", metadata={"help": "Path to the training data."})
    score_type: str = field(default="BOTH", metadata={"help": "IFD or BOTH"})
    diversity_weight_decay: float = field(default=0.1)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    output_dir: str = field(default="./outputs/sft")
    ratio: float = field(default=0.05, metadata={"help": "used to determine number of samples to be used in each epoch (len(initial_dataset) * ratio)"})
    update_ratio: int = field(default=1, metadata={"help": "used to determine the number of samples to be updated after each epoch, =1 refers to the whole dataset, >1 refers to the times of data used in each epoch"})
    update_strategy: str = field(default="iterative", metadata={"help": "iterative, baseline"})
    pre_computed_score_path: str = field(default=None)
    
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

        
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    if "wizard" in data_args.data_path or "dolly" in data_args.data_path:
        training_args.max_length=1024
    elif "mapie" in data_args.data_path:
        training_args.max_length = 1024
    else:
        training_args.max_length = model.config.max_position_embeddings

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.chat_template = chat_template_dict[model_args.prompt_template]

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, 
                                      score_type=data_args.score_type, 
                                      diversity_weight_decay = data_args.diversity_weight_decay)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    trainer = IterativeTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    # trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
