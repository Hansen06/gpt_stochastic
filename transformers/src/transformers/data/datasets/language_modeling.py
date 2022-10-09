# Copyright 2020 The HuggingFace Team. All rights reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pickle
import random
import time
from typing import Dict, List, Optional
import copy

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock
import datasets
from tqdm import tqdm

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
from collections import defaultdict

from ....transformers import (
    GPT2Tokenizer,
    BertTokenizer
)

import constants

logger = logging.get_logger(__name__)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):

        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            f"cached_lm_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should look for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

class WikisectionDataset(TextDataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self,
                 cl_model,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 special_words: list,
                 use_section_null: bool,
                 overwrite_cache=False,
                 cache_dir: Optional[str] = None,
                 ):
        super(WikisectionDataset, self).__init__(
                 tokenizer=tokenizer,
                 file_path=file_path,
                 block_size=block_size,
                 overwrite_cache=overwrite_cache,
                 cache_dir=cache_dir,
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cl_model = cl_model
        self.use_section_null = use_section_null
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.examples = []
        self.raw_texts = []
        self.cl_texts = []
        self.section_ids = []
        self.cl_embeddings = []
        self.n_big = 0
        self.n_small = 0
        self.cpu_device = torch.device('cpu')
        self.cl_offset = 0
        self.lengths = defaultdict(lambda: [])
        self.section_idx_offset = 1
        self.special_words = special_words

        if 'long' in file_path:
            assert len(self.special_words) > 5
        else:
            assert len(self.special_words) == 5

        # string form of id's
        self.section_names = self.special_words[:-1]
        self.cl_eos_str = self.special_words[-1]
        assert self.cl_eos_str == ' . '
        # id token
        section_tokens = self.tokenizer(self.section_names)['input_ids']
        self.section_tokens = [tok[0] for tok in section_tokens]
        self.cl_eos_id = self.tokenizer(self.cl_eos_str)['input_ids'][0]
        assert self.cl_eos_id > 50000 # just checking its a new token

        self.set_cl_tokenizer()
        start = time.time()
        self.process_dataset()
        end = time.time()
        print("Processing dataset took {}".format(end-start))

    def process_dataset(self):
        with open(self.file_path, encoding="utf-8") as f:
            for idx, row in enumerate(f.readlines()):
                if row.strip():
                    # Text used for CL embeddings.
                    cl_text = self._clean2cltext(row)
                    # Text for GPT2
                    row = row.strip() # NOTE: remove break line
                    row = row.replace(". ", " . ") # NOTE marking end of sentence
                    row = f"{self.tokenizer.bos_token} {row} {self.tokenizer.eos_token}"
                    tokenized_text = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(row))

                    last_section_id = 0
                    if len(tokenized_text) >= self.block_size:
                        pass
                        # skip / filter large exmaples
                    else:
                        self.n_small += 1
                        example = self.tokenizer.build_inputs_with_special_tokens(tokenized_text)
                        self.examples.append(example)
                        section_ids, _ = self._determine_section_ids(example, last_section_id)
                        self.section_ids.append(section_ids)
                        self.raw_texts.append(self.tokenizer.decode(self.examples[-1]))
                        self.cl_texts.append(cl_text)
                        self._get_cl_embeddings(tokenized_example=example, gpt2_text=row,
                                                raw_text=self.raw_texts[-1],
                                                cl_text=self.cl_texts[-1])
                        if len(self.examples) > 1422:
                            break # same length as toy wikisection

        self.labels = copy.deepcopy(self.examples)
        print(f"big: {self.n_big} vs small: {self.n_small}")
        for k, v in self.lengths.items():
            print("[ {} ] {}+-{}".format(k, np.mean(v), np.std(v)/np.sqrt(len(v) )))


    def set_cl_tokenizer(self):
        self.cl_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.cl_tokenizer.pad_token = self.cl_tokenizer.eos_token
        self.cl_tokenizer.add_tokens(self.section_names)

    def cl_tokenize(self, text, device):
        output = self.cl_tokenizer(
            text,
            padding=True,
            return_tensors='pt',
        )
        input_ids = output['input_ids'].squeeze(0)
        attention_mask = output['attention_mask'].squeeze(0)
        eos_input_ids = torch.tensor([[self.cl_tokenizer.eos_token_id]*input_ids.shape[0]])
        eos_attention = torch.tensor([[0]*input_ids.shape[0]])
        input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
        attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)
        return input_ids.to(device), attention_mask.to(device)


    def _full_section_ids(self, tokenized_text, last_section_id):
        """output an array \in [0, 3]"""
        section_ids = np.zeros(len(tokenized_text))
        section_tokens = self.tokenizer(self.section_names)['input_ids']
        section_tokens = [tok[0] for tok in section_tokens]

        # Getting the first section token
        start_tok = None
        for tok in section_tokens:
            if tok in tokenized_text:
                start_tok = tok
                break

        if start_tok is None:
            start_tok = section_tokens[last_section_id]
            start_token_idx = 0
            pause = True
        else:
            start_token_idx = tokenized_text.index(start_tok)
            pause = False

        # Handle off-by-one (feed in new section before the section actually starts)
        start_token_idx -= self.section_idx_offset
        start_token_idx = max(0, start_token_idx)

        if start_tok != section_tokens[0]: # doesn't start with abstract
            section_ids[:start_token_idx] = section_tokens.index(start_tok - 1) # \in [0, 3]

        for next_tok in section_tokens[section_tokens.index(start_tok)+1:]:
            # if next_tok is not in text, then the rest is for start_tok
            if next_tok not in tokenized_text:
                section_ids[start_token_idx:] = section_tokens.index(start_tok) # \in [0, 3]
                break
            else:
                # Handle off-by-one
                next_tok_idx = tokenized_text.index(next_tok) - self.section_idx_offset
                next_tok_idx = max(0, next_tok_idx)
                section_ids[start_token_idx:next_tok_idx] = section_tokens.index(start_tok) # \in [0, 3]

            self.lengths[self.section_names[section_tokens.index(start_tok)]].append(
                next_tok_idx + 1 - start_token_idx)
            start_tok = next_tok
            start_token_idx = next_tok_idx

        if start_tok == section_tokens[-1]: # end section, rest of text is this token
            section_ids[start_token_idx:] = section_tokens.index(start_tok) # \in [0, 3]
            self.lengths[self.section_names[section_tokens.index(start_tok)]].append(
                len(section_ids) - start_token_idx)

        last_section_id = int(section_ids[-1])
        return section_ids, last_section_id

    def _null_section_id(self, tokenized_text, last_section_id):
        """output an array \in [0, 4] where 4 = null"""
        section_tokens = self.tokenizer(self.section_names)['input_ids']
        section_tokens = [tok[0] for tok in section_tokens]
        NULL_ID = len(section_tokens)
        section_ids = np.ones(len(tokenized_text)) * NULL_ID

        for section_id, section_tok in enumerate(section_tokens):
            if section_tok in tokenized_text:
                tok_idx = tokenized_text.index(section_tok) - self.section_idx_offset
                tok_idx = max(0, tok_idx)
                section_ids[tok_idx] = section_id
                last_section_id = section_id

        return section_ids, last_section_id


    def _determine_section_ids(self, tokenized_text, last_section_id):
        if self.use_section_null:
            section_ids, last_section_id = self._null_section_id(tokenized_text, last_section_id)
        else:
            section_ids, last_section_id = self._full_section_ids(tokenized_text, last_section_id)
        return section_ids, last_section_id

    def _clean2cltext(self, row):
        # Remove section tokens from text.
        for tok in self.section_names:
            row = row.replace(tok, "")
        cl_text = row.replace(".\n", ". ")
        return cl_text

    def _get_cl_embeddings(self, tokenized_example, gpt2_text, raw_text, cl_text):
        return self.get_cl_embeddings(tokenized_example, gpt2_text, raw_text, cl_text)

    def get_end_points(self, tokenized_example):
        eos_idxs = [i-1 for i, x in enumerate(tokenized_example) if x == self.cl_eos_id]
        eos_idxs += [len(tokenized_example)]
        return eos_idxs

    def get_cl_embeddings(self, tokenized_example, gpt2_text, raw_text, cl_text):
        split_pattern = " . "
        cl_embeddings = []
        eos_idxs = self.get_end_points(tokenized_example)
        split_sentences = gpt2_text.split(split_pattern)
        split_sentences = [_ + split_pattern for _ in split_sentences[:-1]] + [split_sentences[-1]]
        assert len(eos_idxs) == len(split_sentences)
        cl_input_ids, cl_attention_mask = self.cl_tokenize(split_sentences, self.device)
        cl_feats = self.cl_model.forward(input_ids=cl_input_ids, attention_mask=cl_attention_mask) # 1, feat_size
        # Align feats to the sentence length
        last_idx = 0
        for eos_idx, feat in zip(eos_idxs, cl_feats):
            cl_embeddings += [feat] * (eos_idx - last_idx)
            last_idx = eos_idx

        assert len(cl_embeddings) == len(tokenized_example)

        if self.cl_offset:
            cl_embeddings = cl_embeddings[self.cl_offset:] + [cl_embeddings[-1]] * self.cl_offset
        self.cl_embeddings.append(cl_embeddings)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.section_ids[i], dtype=torch.long),
                torch.stack(self.cl_embeddings[i]).to(self.cpu_device),
                self.cl_texts[i]
                )

class TaskmasterDataset(TextDataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self,
                 cl_model,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 use_section_null: bool,
                 special_words:list,
                 data_dir=os.path.join(constants.PATH2RECIPENLG, 'dataset'),
                 overwrite_cache=False,
                 cache_dir: Optional[str] = None,
                 name: str = 'wikihow'
                 ):
        super(TaskmasterDataset, self).__init__(
                tokenizer=tokenizer,
                 file_path=file_path,
                 block_size=block_size,
                 overwrite_cache=overwrite_cache,
                 cache_dir=cache_dir,)
        self.name = name
        self.train = True if 'train' in file_path else False
        self.cpu_device = torch.device('cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cl_model = cl_model
        self.lengths = defaultdict(lambda: [])
        self.special_words = special_words
        assert self.special_words  # should not be emtpy
        self.special_tokens = [_[0] for _ in tokenizer(self.special_words)['input_ids']]
        self.file_path = file_path
        self.data_dir = data_dir
        self.train = 'train' in self.file_path
        self.block_size = block_size
        self.cl_offset = 0
        self._set_indices()
        self.set_cl_tokenizer()

        self.use_section_null = use_section_null
        self.tokenizer = tokenizer
        self.examples = []
        self.cl_texts = []
        self.cl_embeddings = []
        self.section_ids = []
        self.raw_texts = []

        # string form of id's
        self.special_words = special_words
        self.section_names = self.special_words[:-1]
        self.cl_eos_str = self.special_words[-1]
        assert self.cl_eos_str == ' . '
        # id token
        section_tokens = self.tokenizer(self.section_names)['input_ids']
        self.section_tokens = [tok[0] for tok in section_tokens]
        self.cl_eos_id = self.tokenizer(self.cl_eos_str)['input_ids'][0]
        assert self.cl_eos_id > 50000  # just checking its a new token

        self._process_dataset()

    def set_cl_tokenizer(self):
        self.cl_tokenizer = GPT2Tokenizer.from_pretrained(constants.PATH2GPT)
        self.cl_tokenizer.pad_token = self.cl_tokenizer.eos_token
        self.cl_tokenizer.add_tokens(self.special_words)

    def cl_tokenize(self, text, device):
        output = self.cl_tokenizer(
            text,
            padding=True,
            return_tensors='pt',
        )
        input_ids = output['input_ids'].squeeze(0)
        attention_mask = output['attention_mask'].squeeze(0)
        eos_input_ids = torch.tensor([[self.cl_tokenizer.eos_token_id]*input_ids.shape[0]])
        eos_attention = torch.tensor([[0]*input_ids.shape[0]])
        input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
        attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)
        return input_ids.to(device), attention_mask.to(device)

    def _set_indices(self):
        print('LOADING MOVIE TM')
        self.data_dir = constants.PATH2TICKETTALK
        if self.train:
            self.data_files = ['data_0{}.json'.format(i) for i in range(0, 3)]
        else:
            self.data_files = ['data_{}.json'.format(i) for i in range(13, 14)]

    def _process_dataset(self):
        num_filtered = 0

        self.processed_data = []
        split_pattern = ".  "
        doc_counter = 0
        # self.lengths = defaultdict(lambda: [])
        for fname in self.data_files:
            data = json.load(open(os.path.join(self.data_dir, fname), 'rb'))
            if "restaurant" in self.name:
                data = data[self.start_conversation:self.end_conversation]
            for conversation in data:
                full_text = ""
                cl_text = []
                for sentence_counter, utterance in enumerate(conversation['utterances']):
                    text = "[ {} ] {}".format(utterance['speaker'].upper(), utterance['text'])
                    full_text += text + " "
                    cl_text.append(text)
                row = f"{self.tokenizer.bos_token} {full_text} {self.tokenizer.eos_token}"
                tokenized_text = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(row))
                if len(tokenized_text) >= self.block_size:
                    num_filtered+=1
                else:
                    example = self.tokenizer.build_inputs_with_special_tokens(tokenized_text)
                    self.examples.append(example)
                    self.cl_texts.append(full_text)
                    section_ids = [0]
                    self.get_cl_embeddings(example, full_text, cl_text, gpt2_text=row)
                    self.section_ids.append(section_ids)
                    self.raw_texts.append(row)
            if len(self.examples) > 1240:
                break

        self.labels = copy.deepcopy(self.examples)
        print("num examples {}".format(len(self.examples)))
        print(f"num filtered {num_filtered}")
        print("Lengths")
        for k, v in self.lengths.items():
            print("[ {} ] {}+-{}".format(k, np.mean(v), np.std(v)/np.sqrt(len(v))))

        print("examples")
        print(self.raw_texts[0])
        print(self.raw_texts[-1])

    def get_end_points(self, tokenized_example):
        eos_idxs = []
        for tok in self.special_tokens[:2]:
            eos_idxs += [i-1 for i, x in enumerate(tokenized_example) if x == tok]
        eos_idxs += [len(tokenized_example)]
        eos_idxs.sort()
        eos_idxs = eos_idxs[1:]
        return eos_idxs

    def get_cl_embeddings(self, tokenized_example, raw_text, cl_text, gpt2_text):

        cl_embeddings = []
        eos_idxs = self.get_end_points(tokenized_example)

        assert len(eos_idxs) == len(cl_text)

        cl_input_ids, cl_attention_mask = self.cl_tokenize(cl_text, self.device)
        cl_feats = self.cl_model.forward(
            input_ids=cl_input_ids, attention_mask=cl_attention_mask) # 1, feat_size
        # Align feats to the sentence length
        last_idx = 0
        for eos_idx, feat in zip(eos_idxs, cl_feats):
            cl_embeddings += [feat] * (eos_idx - last_idx)
            last_idx = eos_idx

        assert len(cl_embeddings) == len(tokenized_example)

        if self.cl_offset:
            cl_embeddings = cl_embeddings[self.cl_offset:] + [cl_embeddings[-1]] * self.cl_offset
        self.cl_embeddings.append(cl_embeddings)

    def __len__(self):
        return len(self.examples)


    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.section_ids[i], dtype=torch.long),
                torch.stack(self.cl_embeddings[i]).to(self.cpu_device),
                self.cl_texts[i]
                )

class EekeDataset(TextDataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self,
                 cl_model,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 use_section_null: bool,
                 special_words:list,
                 data_dir=constants.PATH2Erke,
                 overwrite_cache=False,
                 cache_dir: Optional[str] = None,
                 name: str = 'wikihow'
                 ):
        super(EekeDataset, self).__init__(
                tokenizer=tokenizer,
                 file_path=file_path,
                 block_size=block_size,
                 overwrite_cache=overwrite_cache,
                 cache_dir=cache_dir,)
        self.name = name
        self.cpu_device = torch.device('cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cl_model = cl_model
        self.lengths = defaultdict(lambda: [])
        self.special_words = special_words
        assert self.special_words  # should not be emtpy
        self.special_tokens = [_[0] for _ in tokenizer(self.special_words)['input_ids']]
        self.file_path = file_path
        self.data_dir = data_dir
        self.train = 'train' in self.file_path
        self.block_size = block_size
        self.cl_offset = 0

        print('==============LOADING ERKE DATA==============')
        if self.train:
            self.data_files = ['train-.json']
        else:
            self.data_files = ['valid-.json']

        # self.cl_tokenizer = GPT2Tokenizer.from_pretrained(constants.PATH2GPT)
        self.cl_tokenizer = BertTokenizer.from_pretrained(constants.PATH2GPT)
        self.cl_tokenizer.pad_token = self.cl_tokenizer.eos_token
        self.cl_tokenizer.add_tokens(self.special_words)

        self.use_section_null = use_section_null
        self.tokenizer = tokenizer
        self.examples = []
        self.cl_texts = []
        self.cl_embeddings = []
        self.section_ids = []
        self.raw_texts = []

        # string form of id's
        self.special_words = special_words
        self.section_names = self.special_words[:-1]
        self.cl_eos_str = self.special_words[-1]
        assert self.cl_eos_str == ' . '
        # id token
        section_tokens = self.tokenizer(self.section_names)['input_ids']
        self.section_tokens = [tok[0] for tok in section_tokens]
        self.cl_eos_id = self.tokenizer(self.cl_eos_str)['input_ids'][1]
        print('========self.cl_eos_id {}========'.format(self.cl_eos_id))
        assert self.cl_eos_id > 20000  # just checking its a new token

        self._process_dataset()

    def cl_tokenize(self, text, device):
        output = self.cl_tokenizer(
            text,
            padding=True,
            return_tensors='pt',
        )
        input_ids = output['input_ids'].squeeze(0)
        attention_mask = output['attention_mask'].squeeze(0)
        eos_input_ids = torch.tensor([[self.cl_tokenizer.eos_token_id]*input_ids.shape[0]])
        eos_attention = torch.tensor([[0]*input_ids.shape[0]])
        input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
        attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)
        return input_ids.to(device), attention_mask.to(device)


    def _process_dataset(self):
        num_filtered = 0

        self.processed_data = []
        split_pattern = ".  "
        doc_counter = 0
        # self.lengths = defaultdict(lambda: [])
        for fname in self.data_files:
            data = json.load(open(os.path.join(self.data_dir, fname), 'rb'))
            if "restaurant" in self.name:
                data = data[self.start_conversation:self.end_conversation]
            for conversation in data:
                full_text = ""
                cl_text = []
                for sentence_counter, utterance in enumerate(conversation['utterances']):
                    text = "[ {} ] {}".format(utterance['speaker'].upper(), utterance['text'])
                    full_text += text + " "
                    cl_text.append(text)
                row = f"{self.tokenizer.bos_token} {full_text} {self.tokenizer.eos_token}"
                tokenized_text = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(row))
                if len(tokenized_text) >= self.block_size:
                    num_filtered+=1
                else:
                    example = self.tokenizer.build_inputs_with_special_tokens(tokenized_text)
                    self.examples.append(example)
                    self.cl_texts.append(full_text)
                    section_ids = [0]
                    self.get_cl_embeddings(example, full_text, cl_text, gpt2_text=row)
                    self.section_ids.append(section_ids)
                    self.raw_texts.append(row)
            if len(self.examples) > 1240:
                break

        self.labels = copy.deepcopy(self.examples)
        print("num examples {}".format(len(self.examples)))
        print(f"num filtered {num_filtered}")
        print("Lengths")
        for k, v in self.lengths.items():
            print("[ {} ] {}+-{}".format(k, np.mean(v), np.std(v)/np.sqrt(len(v))))

        print("examples")
        print(self.raw_texts[0])
        print(self.raw_texts[-1])

    def get_end_points(self, tokenized_example):
        eos_idxs = []
        for tok in self.special_tokens[:2]:
            eos_idxs += [i-1 for i, x in enumerate(tokenized_example) if x == tok]
        eos_idxs += [len(tokenized_example)]
        eos_idxs.sort()
        eos_idxs = eos_idxs[1:]
        return eos_idxs

    def get_cl_embeddings(self, tokenized_example, raw_text, cl_text, gpt2_text):

        cl_embeddings = []
        eos_idxs = self.get_end_points(tokenized_example)

        assert len(eos_idxs) == len(cl_text)

        cl_input_ids, cl_attention_mask = self.cl_tokenize(cl_text, self.device)
        cl_feats = self.cl_model.forward(
            input_ids=cl_input_ids, attention_mask=cl_attention_mask) # 1, feat_size
        # Align feats to the sentence length
        last_idx = 0
        for eos_idx, feat in zip(eos_idxs, cl_feats):
            cl_embeddings += [feat] * (eos_idx - last_idx)
            last_idx = eos_idx

        assert len(cl_embeddings) == len(tokenized_example)

        if self.cl_offset:
            cl_embeddings = cl_embeddings[self.cl_offset:] + [cl_embeddings[-1]] * self.cl_offset
        self.cl_embeddings.append(cl_embeddings)

    def __len__(self):
        return len(self.examples)


    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.section_ids[i], dtype=torch.long),
                torch.stack(self.cl_embeddings[i]).to(self.cpu_device),
                self.cl_texts[i]
                )