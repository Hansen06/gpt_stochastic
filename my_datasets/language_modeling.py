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

from collections import defaultdict

import constants

class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer,
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
                print(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                print(f"Creating features from dataset file at {directory}")

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
                print(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

class EekeDataset():
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self,
                 cl_model,
                 tokenizer,
                 block_size: int,
                 special_words:list,
                 data_dir=constants.PATH2Erke,
                 data_names: str = 'wikihow'
                 ):
        self.data_names = data_names
        self.cpu_device = torch.device('cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cl_model = cl_model
        self.lengths = defaultdict(lambda: [])
        self.special_words = special_words
        assert self.special_words  # should not be emtpy
        self.special_tokens = [_[1] for _ in tokenizer(self.special_words)['input_ids']]
        self.data_dir = data_dir
        self.train = 'train' in self.data_names
        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        self.cl_offset = 0

        if self.train:
            print('==============loading erke train data==============')
            # self.data_files = ['valid-.json']
            self.data_files = ['train-.txt']
        else:
            print('==============loading erke valid data==============')
            # self.data_files = ['valid-.json']
            self.data_files = ['valid-.txt']

        self.tokenizer = tokenizer
        self.examples = []
        self.labels = []
        self.cl_texts = []
        self.cl_embeddings = []
        self.section_ids = []
        self.token_type_ids = []
        self.raw_texts = []

        self._process_dataset()

    def cl_tokenize(self, text, device):
        output = self.tokenizer(
            text,
            padding=True,
            return_tensors='pt',
        )
        input_ids = output['input_ids'].squeeze(0)
        attention_mask = output['attention_mask'].squeeze(0)
        eos_input_ids = torch.tensor([[self.tokenizer.eos_token_id]*input_ids.shape[0]])
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
            data = []
            with open(os.path.join(self.data_dir, fname), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    data.append(line.strip())

            for conversation in data:
                full_text = ""
                cl_text = []
                labels = []
                token_type_ids = []
                user_id = self.tokenizer.convert_tokens_to_ids('[ user ]')
                assistant_id = self.tokenizer.convert_tokens_to_ids('[ assistant ]')
                token_type_ids.append(self.tokenizer.bos_token_id)
                labels.append(-1)

                conversation = conversation.strip().split('\t')
                for sentence_counter, utterance in enumerate(conversation):
                    sp = utterance.split('[next]')
                    new_txt = []
                    for i, line in enumerate(sp):
                        if i != len(sp) - 1:
                            new_txt.append(line)
                            new_txt.append(' [ [next] ] ')
                        else:
                            new_txt.append(line)

                    if sentence_counter % 2 == 0:
                        text = "[ {} ] {}".format('user'.upper(), ''.join(new_txt))
                    else:
                        text = "[ {} ] {}".format('assistant'.upper(), ''.join(new_txt))

                    text_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
                    full_text += text + " "
                    cl_text.append(text)
                    if sentence_counter % 2 == 0: #患者
                        token_type_ids.extend([user_id] * len(self.tokenizer.tokenize(text)))
                        if sentence_counter != len(conversation) - 1:
                            labels.extend([-1] * len(self.tokenizer.tokenize(text)))
                        else:
                            labels.extend(text_ids)
                    else:
                        token_type_ids.extend([assistant_id] * len(self.tokenizer.tokenize(text)))
                        if sentence_counter != len(conversation) - 1:
                            labels.extend([-1] * len(self.tokenizer.tokenize(text)))
                        else:
                            labels.extend(text_ids)

                token_type_ids.append(self.tokenizer.eos_token_id)
                labels.append(self.tokenizer.eos_token_id)

                row = f"{self.tokenizer.bos_token} {full_text} {self.tokenizer.eos_token}"
                input_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(row))
                # print('row :{}'.format(row))
                # print('tokenized_text :{}'.format(tokenized_text))
                if len(input_ids) >= self.block_size:
                    num_filtered+=1
                else:
                    # example = self.tokenizer.build_inputs_with_special_tokens(tokenized_text) #在头尾分别添加[CLS]和[SEP] token
                    # print('example: {}'.format(example))
                    self.examples.append(input_ids)
                    self.labels.append(labels)
                    self.cl_texts.append(full_text)
                    self.get_cl_embeddings(input_ids, cl_text)
                    self.token_type_ids.append(token_type_ids)
                    self.raw_texts.append(row)
            if len(self.examples) > 1204:
                break

        # self.labels = copy.deepcopy(self.examples)
        print("num examples {}".format(len(self.examples)))
        print(f"num filtered {num_filtered}")
        print("Lengths")
        for k, v in self.lengths.items():
            print("[ {} ] {}+-{}".format(k, np.mean(v), np.std(v)/np.sqrt(len(v))))

        print("=====================================examples====================================")
        print(self.raw_texts[0])
        print(self.raw_texts[-1])
        print(self.examples[0])
        # print(self.cl_embeddings[0])
        print("=====================================examples====================================")

    def get_end_points(self, tokenized_example):
        '''
        获取每句话的最后一个token index
        '''
        eos_idxs = []
        for tok in self.special_tokens[:2]:
            eos_idxs += [i-1 for i, x in enumerate(tokenized_example) if x == tok]
        eos_idxs += [len(tokenized_example)]
        eos_idxs.sort()
        eos_idxs = eos_idxs[1:]
        return eos_idxs

    def get_cl_embeddings(self, tokenized_example, cl_text):

        cl_embeddings = []
        eos_idxs = self.get_end_points(tokenized_example)
        # print('tokenized_example : {}'.format(tokenized_example))
        # print('eos_idxs : {}'.format(eos_idxs))
        # print('=======len eos_idxs :{}============'.format(len(eos_idxs)))
        # print('=======len cl_text :{}============'.format(len(cl_text)))

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
                torch.tensor(self.token_type_ids[i], dtype=torch.long),
                torch.stack(self.cl_embeddings[i]).to(self.cpu_device),
                self.cl_texts[i]
                )