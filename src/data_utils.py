# encoding = "utf-8"
import torch
import copy
import json
import logging
import transformers
from tqdm import tqdm
import numpy as np
from scipy.sparse import find
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers.trainer_pt_utils import LabelSmoother
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Sequence


IGNORE_INDEX = LabelSmoother.ignore_index

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, score_type: str, diversity_weight_decay: float): 
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = self.load_data(data_path)
        self.score_type=score_type
        self.diversity_weight_decay = diversity_weight_decay
       
        logging.warning("Formatting inputs...")
        sources, targets = [], []
        direct_answers, whole_texts = [], []
        prior_start_idxes, cond_start_idxes, answer_lens = [], [], []
        scores = []

        for example in tqdm(list_data_dict):
            
            chat = [{"role": "system", "content": "You are a helpful assistant."}]

            if type(example["instruction"])==list:
                for turn in example["instruction"]:
                    if turn["from"]=="human":
                        chat.append({"role": "user", "content": turn["value"]})
                    else:
                        chat.append({"role": "assistant", "content": turn["value"]})
            else:
                if "input" not in example or example["input"]=="":
                    chat.append({"role": "user", "content": example["instruction"]})
                else:
                    chat.append({"role": "user", "content": example["instruction"]+"\nInput: " + example["input"]})
                
            sources.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
            targets.append(example["output"] + tokenizer.eos_token)

            chat.append({"role": "assistant", "content": example["output"]})
            whole_texts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False))
            chat = [{"role":"assistant", "content": example["output"]}]
            direct_answers.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False))

            prior_start_idxes.append(
                len(
                    tokenizer.tokenize(direct_answers[-1][:direct_answers[-1].rfind(example["output"])])
                )
            )

            cond_start_idxes.append(
                len(
                    tokenizer.tokenize(whole_texts[-1][:whole_texts[-1].rfind(example["output"])])
                )
            )
            
            answer_lens.append(
                max(
                    min(tokenizer.model_max_length, len(tokenizer.tokenize(whole_texts[-1]))) - cond_start_idxes[-1], # change encode to tokenize
                    0
                )
            )

            scores.append(0.0)


        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.original_idx = torch.tensor([i for i in range(len(list_data_dict))])
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.direct_answers_input_ids = _tokenize_fn(direct_answers, tokenizer)["input_ids"]
        self.whole_texts_input_ids = _tokenize_fn(whole_texts, tokenizer)["input_ids"]
        self.prior_start_idx = torch.tensor(prior_start_idxes)
        self.cond_start_idx = torch.tensor(cond_start_idxes)
        self.answer_len = torch.tensor(answer_lens)
        self.score = torch.tensor(scores, dtype=torch.bfloat16)
        self.source =  targets
        self.source_len = torch.tensor([len(tokenizer.tokenize(t)) for t in targets])
  
    def load_data(self, data_path):
        data = []
        with open(data_path, "r") as f:
            # for line in f:
            #     data.append(json.loads(line))
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], 
                    direct_answers_input_ids=self.direct_answers_input_ids[i],
                    whole_texts_input_ids=self.whole_texts_input_ids[i],
                    prior_start_idx=self.prior_start_idx[i],
                    cond_start_idx=self.cond_start_idx[i],
                    answer_len=self.answer_len[i],
                    original_idx=self.original_idx[i], 
                    score=self.score[i],
                    source = self.source[i],
                    source_len = self.source_len[i]
                    )

    def update(self, count, select_count):
        
        '''sort samples based on "IFD score" and "diversity"''' 
       
        indices = torch.nonzero(self.score[:count] < 1.0).squeeze()
        other_indices = torch.nonzero(self.score[:count] >= 1.0).squeeze()
        if other_indices.size()==torch.Size([]):
            other_indices = other_indices.unsqueeze(0)
        # indices = torch.nonzero(self.score[:count] != torch.nan).squeeze()

        score_filtered = self.score[:count][indices]
        IFD_sorted_indices = torch.argsort(score_filtered, descending=True)

        if self.score_type == "IFD": # only based on IFD
            sorted_indices = IFD_sorted_indices
            
        else: # self.score_type == "BOTH", balance between IFD and diversity
            score_filtered = score_filtered.cpu().to(torch.float32).numpy()        
            candidate_sources = [self.source[i] for i in indices]

            vectorizer = TfidfVectorizer(ngram_range=(1,3))
            tfidf_matrix = vectorizer.fit_transform(candidate_sources)
            tfidf_matrix = tfidf_matrix.tocsc()

            sorted_indices = []
            for i in tqdm(range(select_count)):
                tfidf_sums = np.array(tfidf_matrix.sum(axis=1)).flatten() #np.array(tfidf_matrix.sum(axis=1)).flatten()
                
                final_score = score_filtered * tfidf_sums
        
                max_doc_index = np.argmax(final_score)

                '''prepare for the next iteration'''
                _, col_indices, _ = find(tfidf_matrix[max_doc_index, :])
                for col_index in col_indices:
                    tfidf_matrix.data[tfidf_matrix.indptr[col_index]:tfidf_matrix.indptr[col_index + 1]] *= self.diversity_weight_decay #0.1
                    tfidf_matrix[max_doc_index, col_index] = 0.0


                sorted_indices.append(max_doc_index)
            
            sorted_indices += [i for i in IFD_sorted_indices.cpu().numpy() if i not in sorted_indices]
            sorted_indices = torch.tensor(sorted_indices).to(indices.device)

        final_indices = torch.cat((indices[sorted_indices][:select_count], other_indices, indices[sorted_indices][select_count:]))

        self.original_idx = torch.cat((self.original_idx[final_indices], self.original_idx[count:]))
        self.score = torch.cat((self.score[final_indices], self.score[count:]))
        self.prior_start_idx = torch.cat((self.prior_start_idx[final_indices], self.prior_start_idx[count:]))
        self.cond_start_idx = torch.cat((self.cond_start_idx[final_indices], self.cond_start_idx[count:]))
        self.answer_len = torch.cat((self.answer_len[final_indices], self.answer_len[count:]))
        self.source_len = torch.cat((self.source_len[final_indices], self.source_len[count:]))

        input_ids = []
        labels = []
        direct_answers_input_ids = []
        whole_texts_input_ids = []
        source = []

        for index in final_indices:
            input_ids.append(self.input_ids[index])
            labels.append(self.labels[index])
            direct_answers_input_ids.append(self.direct_answers_input_ids[index])
            whole_texts_input_ids.append(self.whole_texts_input_ids[index])
            source.append(self.source[index])

        self.input_ids = input_ids + self.input_ids[count:]
        self.labels = labels + self.labels[count:]
        self.direct_answers_input_ids = direct_answers_input_ids + self.direct_answers_input_ids[count:]
        self.whole_texts_input_ids = whole_texts_input_ids + self.whole_texts_input_ids[count:]
        self.source = source + self.source[count:]

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@dataclass
class DataCollatorForScoreDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        direct_answers_input_ids, whole_texts_input_ids = tuple([instance[key] for instance in instances] for key in ("direct_answers_input_ids", "whole_texts_input_ids"))
        
        direct_answers_input_ids = torch.nn.utils.rnn.pad_sequence(
                direct_answers_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        direct_answers_attention_mask=direct_answers_input_ids.ne(self.tokenizer.pad_token_id)
        direct_answers_label_mask = torch.zeros_like(direct_answers_input_ids, dtype=torch.bool)

    
        whole_texts_input_ids = torch.nn.utils.rnn.pad_sequence(
                whole_texts_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        whole_texts_attention_mask=whole_texts_input_ids.ne(self.tokenizer.pad_token_id)
        whole_texts_label_mask = torch.zeros_like(whole_texts_input_ids, dtype=torch.bool)
        # whole_texts_instruction_mask = torch.zeros_like(whole_texts_input_ids, dtype=torch.bool)

        for i, instance in enumerate(instances):
            prior = instance["prior_start_idx"]
            cond = instance["cond_start_idx"]
            ans = instance["answer_len"]

            direct_answers_label_mask[i, prior:prior+ans] = True
            whole_texts_label_mask[i, cond:cond+ans] = True
            # whole_texts_instruction_mask[i, :cond] = True

        return dict(
                direct_answers_input_ids=direct_answers_input_ids,
                direct_answers_attention_mask=direct_answers_attention_mask,
                direct_answers_label_mask=direct_answers_label_mask,
                whole_texts_input_ids=whole_texts_input_ids,
                whole_texts_attention_mask=whole_texts_attention_mask,
                whole_texts_label_mask=whole_texts_label_mask,
                # whole_texts_instruction_mask=whole_texts_instruction_mask,
            )

