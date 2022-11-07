# language model 数据加载模块

import os
from collections import defaultdict

import torch
from torch.utils.data.dataset import Dataset

import constants

class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data_files = list()
        self.data_files_offset = list()
        self.data_len = 0
        self._check_files()

    def _check_files(self):
        if self.data_path is None:
            raise RuntimeError("Data path cannot be \
                empty at same time.")

        if self.data_path:
            if not os.path.exists(self.data_path):
                raise RuntimeError("Training files does not exist at " + self.data_path)
            prepare_files_offset(self.data_path, self.data_files,
                                 self.data_files_offset)
            # print(self.data_files_offset)
            self.data_len = len(self.data_files_offset)

    def __len__(self):
        return self.data_len

    def _get_line(self, index):
        tup = self.data_files_offset[index]
        target_file = self.data_files[tup[0]]
        with open(target_file, "r", encoding="utf-8") as f:
            f.seek(tup[1])
            line = f.readline()
        return line


class EekeDataset(TextDataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self,
                 cl_model,
                 tokenizer,
                 block_size: int,
                 use_section_null: bool,
                 special_words: list,
                 *inputs, **kwargs
                 ):
        super(EekeDataset, self).__init__(*inputs, **kwargs)
        self.cpu_device = torch.device('cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cl_model = cl_model
        self.lengths = defaultdict(lambda: [])
        self.special_words = special_words
        assert self.special_words  # should not be emtpy
        self.special_tokens = [_[1] for _ in tokenizer(self.special_words)['input_ids']]
        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        self.cl_offset = 0

        self.use_section_null = use_section_null
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        conversation = self._get_line(index)
        full_text = ""
        cl_text = []
        sen_len = []
        token_type_ids = []
        labels = []
        user_id = self.tokenizer.convert_tokens_to_ids('[ user ]')
        assistant_id = self.tokenizer.convert_tokens_to_ids('[ assistant ]')
        token_type_ids.append(self.tokenizer.bos_token_id)
        labels.append(-1)
        # print('user_id :{}'.format(user_id))
        # print('assistant_id :{}'.format(assistant_id))
        conversation = conversation.strip().split('\t')
        for sen_count, utterance in enumerate(conversation):
            sp = utterance.split('[next]')
            new_txt = []
            for i, line in enumerate(sp):
                if i != len(sp) - 1:
                    new_txt.append(line)
                    new_txt.append(' [ [next] ] ')
                else:
                    new_txt.append(line)

            if sen_count % 2 == 0:#患者
                # text = "[ {} ] {}".format('user'.upper(), ''.join(new_txt))
                text = ''.join(new_txt)
                text_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
                token_type_ids.extend([user_id] * len(self.tokenizer.tokenize(text)))
                if sen_count != len(conversation) - 1:
                    labels.extend([-1] * len(self.tokenizer.tokenize(text)))
                else:
                    labels.extend(text_ids)
                if sen_count == 0 or sen_count == len(conversation)-1:
                    sen_len.append(len(self.tokenizer.tokenize(text))+1)
                else:
                    sen_len.append(len(self.tokenizer.tokenize(text)))
            else:
                # text = "[ {} ] {}".format('assistant'.upper(), ''.join(new_txt))
                text = ''.join(new_txt)
                text_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
                token_type_ids.extend([assistant_id] * len(self.tokenizer.tokenize(text)))
                if sen_count != len(conversation) - 1:
                    labels.extend([-1] * len(self.tokenizer.tokenize(text)))
                else:
                    labels.extend(text_ids)

                if sen_count == 0 or sen_count == len(conversation)-1:
                    sen_len.append(len(self.tokenizer.tokenize(text))+1)
                else:
                    sen_len.append(len(self.tokenizer.tokenize(text)))

            full_text += text + " "
            cl_text.append(text)

        token_type_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)

        # print('labels : {}'.format(labels))
        # print('labels.len : {}'.format(len(labels)))
        # print('token_type_ids.len : {}'.format(len(token_type_ids)))

        row = f"{self.tokenizer.bos_token} {full_text} {self.tokenizer.eos_token}"
        input_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(row))

        # print('row :{}'.format(row))
        # print('tokenized_text :{}'.format(tokenized_text))
        # print('section_ids :{}'.format(section_ids))

        if len(input_ids) >= self.block_size:
            print('--------------跳过case too long----------------')
            return self.__getitem__(index + 1)
        else:
            cl_embeddings = self.get_cl_embeddings(sen_len, cl_text)
            if cl_embeddings == 0:
                print('--------------跳过case----------------')
                return self.__getitem__(index + 1)
            # labels = copy.deepcopy(input_ids)

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(token_type_ids, dtype=torch.long),
                torch.stack(cl_embeddings).to(self.cpu_device),
                full_text,
                )

    def cl_tokenize(self, text, device):
        output = self.tokenizer(
            text,
            padding=True,
            return_tensors='pt',
        )
        input_ids = output['input_ids'].squeeze(0)
        attention_mask = output['attention_mask'].squeeze(0)
        eos_input_ids = torch.tensor([[self.tokenizer.eos_token_id] * input_ids.shape[0]])
        eos_attention = torch.tensor([[0] * input_ids.shape[0]])
        input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
        attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)
        return input_ids.to(device), attention_mask.to(device)

    def get_cl_embeddings(self, sen_len, cl_text):
        cl_embeddings = []

        # print('len(sen_len) : {}'.format(len(sen_len)))
        # print('len(cl_text) : {}'.format(len(cl_text)))

        if len(sen_len) != len(cl_text):
            return 0

        cl_input_ids, cl_attention_mask = self.cl_tokenize(cl_text, self.device)
        cl_feats = self.cl_model.forward(input_ids=cl_input_ids, attention_mask=cl_attention_mask)  # 1, feat_size
        # Align feats to the sentence length
        for s_len, feat in zip(sen_len, cl_feats):
            cl_embeddings += [feat] * s_len

        assert len(cl_embeddings) == sum(sen_len)

        if self.cl_offset:
            cl_embeddings = cl_embeddings[self.cl_offset:] + [cl_embeddings[-1]] * self.cl_offset

        return cl_embeddings


def prepare_files_offset(path, files_list, offset_list):
    """Fill the file index and offsets of each line in files_list in offset_list
    Args:
        path: string of file path, support single file or file dir
        files_list: the list contains file names
        offset_list: the list contains the tuple of file name index and offset
    """
    if os.path.isdir(path):  # for multi-file, its input is a dir
        files_list.extend([os.path.join(path, f) for f in os.listdir(path)])
    elif os.path.isfile(path):  # for single file, its input is a file
        files_list.append(path)
    else:
        raise RuntimeError(path + " is not a normal file.")
    print(files_list)
    for i, f in enumerate(files_list):
        offset = 0
        with open(f, "r", encoding="utf-8") as single_file:
            for line in single_file:
                tup = (i, offset)
                offset_list.append(tup)
                offset += len(bytes(line, encoding='utf-8'))