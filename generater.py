#!/usr/bin/env python
# coding=utf-8
"""
Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

import argparse
import logging

import os
import numpy as np
import torch
import tqdm
import torch.nn as nn
from model import language
import transformers.src.transformers

from my_datasets.language_modeling import EekeDataset
from transformers.src.transformers import (
    GPT2TimeLMHeadModel,
    GPT2Tokenizer,
    BertTokenizer
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    # "gpt2": (GPT2TimeLMHeadModel, GPT2Tokenizer),
    "gpt2": (GPT2TimeLMHeadModel, BertTokenizer),
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def simulate_brownian_bridge(B_0, B_T, num_samples, sentence_lengths, dt=0.05, mu=0.0, sigma=1.0):
    """Run bridge forward pinned at B_0 and B_T"""
    if isinstance(B_0, torch.Tensor):
        B_0 = B_0.cpu().detach().numpy()
    if isinstance(B_T, torch.Tensor):
        B_T = B_T.cpu().detach().numpy()

    bridge = [B_0]
    x_t = np.copy(B_0)
    for step in range(num_samples - 2):  # number of sentences
        dim = B_0.shape[-1]
        noise = np.sqrt(dt) * sigma * np.random.normal(mu, sigma, dim)
        t = step / num_samples
        x_tp1 = x_t * (1 - dt / (1 - t)) + (dt / (1 - t)) * B_T + noise
        length_idx = step % len(sentence_lengths)
        bridge += [x_tp1] * sentence_lengths[length_idx]
        x_t = x_tp1

    length_idx = step % len(sentence_lengths)
    bridge += [B_T] * sentence_lengths[length_idx]

    return bridge


def split_text(raw_text):
    split_pattern = ". "
    split_raw_text = [_ + split_pattern for _ in raw_text.split(split_pattern)]
    split_raw_text[-1] = split_raw_text[-1].rstrip(split_pattern)
    return split_raw_text


def get_density(dataset, lm, cl_model):
    """Estimate density of last latent"""
    first_latents = []
    last_latents = []
    length = len(dataset)
    for text_i in range(length):
        first_latents.append(dataset.cl_embeddings[text_i][0].detach().cpu().numpy())
        last_latents.append(dataset.cl_embeddings[text_i][-1].detach().cpu().numpy())
    first_latents = np.array(first_latents)
    last_latents = np.array(last_latents)
    return first_latents.mean(0), first_latents.std(0), last_latents.mean(0), last_latents.std(0)


def get_special_tokens(dataset_name, tokenizer, add_tokens=True):
    SPECIAL_TOKENS = []
    if 'erke' in dataset_name:
        SPECIAL_TOKENS = [
            '[ user ]', #21128
            '[ assistant ]', #21129
            '[ <|endoftext|> ]', #21130
            '[ [next] ]' #21131
        ]

    SPECIAL_TOKENS += [' . '] #21132
    if add_tokens:
        # NOTE loading previous tokenizer sometimes already includes the new tokens
        eos = tokenizer(' . ')['input_ids']
        print("Old tokenizer size: ", len(tokenizer))
        if len(eos) == 1 and eos[0] == 21128 + len(SPECIAL_TOKENS):
            print('========================================================================')
            print("Not adding because it's already contained")
            pass  # don't add cause it's already contained
        else:
            print("Adding tokens, ", SPECIAL_TOKENS)
            tokenizer.add_tokens(SPECIAL_TOKENS)
        print("New tokenizer size: ", len(tokenizer))
    SECTION_IDS = [_[1] for _ in tokenizer(SPECIAL_TOKENS)['input_ids']]
    return SECTION_IDS, SPECIAL_TOKENS, tokenizer


def get_checkpoint(latent_dim, use_section_ids=False, token_size=None,
                   filepath=None):
    '''
    加载布朗模型
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = language.GPT2OUEncoder(
        hidden_dim=128,
        latent_dim=latent_dim,
        finetune_gpt2=False)
    if use_section_ids:
        model.model.resize_token_embeddings(token_size)

    transformers.__spec__ = 'gpt2'  # Avoid bug
    state_dict = torch.load(filepath)
    new_dict = {}
    for k, v in state_dict['state_dict'].items():
        if any([i in k for i in ['model.model.g_ar', 'model.model.W_k']]):
            new_dict[k[6:]] = v
        elif any([i in k for i in ['model.g_ar', 'model.W_k', 'time_model']]):
            continue
        elif "model." in k:
            new_dict[k[6:]] = v
        else:
            new_dict[k] = v

    if any(['g_ar' in k for k in new_dict.keys()]):
        model.g_ar = nn.GRU(input_size=latent_dim,
                            hidden_size=2400,  # default number in infoNCE for langauge
                            num_layers=3,
                            batch_first=True
                            )
        model.W_k = nn.Linear(2400, latent_dim)
    elif any(['time_model' in k for k in state_dict['state_dict'].keys()]):
        model.fc_mu = nn.Linear(latent_dim, latent_dim)
        model.fc_var = nn.Linear(latent_dim, latent_dim)

    model.load_state_dict(new_dict)
    for p in model.parameters():
        p.requires_grad = False

    model.to(device)
    model = model.eval()
    return model


def cl_tokenize(tokenizer, text, device):
    output = tokenizer(
        text,
        padding=True,
        return_tensors='pt',
    )
    input_ids = output['input_ids']

    print('output:{}'.format(output))
    print('cl input_ids:{}'.format(input_ids))

    attention_mask = output['attention_mask']
    eos_input_ids = torch.tensor([[tokenizer.eos_token_id]*input_ids.shape[0]])
    eos_attention = torch.tensor([[0]*input_ids.shape[0]])
    input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
    attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)
    return input_ids.to(device), attention_mask.to(device)

def get_cl_embeddings(history, cl_model, tokenizer, special_ids, device):
    full_text = ""
    cl_text = []

    token_type_ids = []
    user_id = tokenizer.convert_tokens_to_ids('[ user ]')
    assistant_id = tokenizer.convert_tokens_to_ids('[ assistant ]')
    token_type_ids.append(tokenizer.bos_token_id)

    for i, utterance in enumerate(history):
        sp = utterance.split('[next]')
        new_txt = []
        for i, line in enumerate(sp):
            if i != len(sp) - 1:
                new_txt.append(line)
                new_txt.append(' [ [next] ] ')
            else:
                new_txt.append(line)
        if i % 2 == 0: #患者
            text = "[ {} ] {}".format('USER', ''.join(new_txt))
            full_text += text + " "
            cl_text.append(text)
            token_type_ids.extend([user_id] * len(tokenizer.tokenize(text)))
        else:
            text = "[ {} ] {}".format('ASSISTANT', ''.join(new_txt))
            full_text += text + " "
            cl_text.append(text)
            token_type_ids.extend([assistant_id] * len(tokenizer.tokenize(text)))

    token_type_ids.append(tokenizer.eos_token_id)

    print('full_text :{}'.format(full_text))

    row = f"{tokenizer.bos_token} {full_text} {tokenizer.eos_token}"
    print('row text:{}'.format(row))
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(row))
    # input_ids = tokenizer.build_inputs_with_special_tokens(tokenized_text)

    print('special_tokens:{}'.format(special_ids))

    cl_embeddings = []
    eos_idxs = []
    for tok in special_ids[:2]:
        eos_idxs += [i - 1 for i, x in enumerate(input_ids) if x == tok]
    eos_idxs += [len(input_ids)]
    eos_idxs.sort()
    eos_idxs = eos_idxs[1:]

    print('eos_idxs :{}\tlenght:{}'.format(eos_idxs,len(eos_idxs)))
    print('cl_text :{}\tlenght：{}'.format(cl_text,len(cl_text)))
    assert len(eos_idxs) == len(cl_text)

    cl_input_ids, cl_attention_mask = cl_tokenize(tokenizer, cl_text, device)
    cl_feats = cl_model.forward(
        input_ids=cl_input_ids, attention_mask=cl_attention_mask)  # 1, feat_size
    last_idx = 0
    for eos_idx, feat in zip(eos_idxs, cl_feats):
        cl_embeddings += [feat] * (eos_idx - last_idx)
        last_idx = eos_idx

    return torch.tensor([input_ids]).to(device), cl_embeddings, eos_idxs, torch.tensor([token_type_ids]).to(device)

def generter(model, history, cl_model, tokenizer, special_tokens, args, last_latent_mu, last_latent_std):
    # Get all the CL feats
    input_ids, cl_embeddings, end, token_type_ids = get_cl_embeddings(history, cl_model, tokenizer, special_tokens, args.device)
    # print(cl_embeddings)
    true_cl_feats = torch.stack(cl_embeddings)
    # print(true_cl_feats)
    print('input_ids:{}'.format(input_ids))
    print('input_ids shape:{}'.format(input_ids.shape))
    print('token_type_ids:{}'.format(token_type_ids))
    print('token_type_ids shape:{}'.format(token_type_ids.shape))

    LABELS = ['TRUE CL', 'BRIDGE CL (DE)', 'RANDOM CL']
    # INTERPOLATION - BRIDGE
    B_T = np.random.normal(loc=last_latent_mu, scale=last_latent_std)

    num_sentences = len(true_cl_feats)

    print("Original num sentences: {}".format(len(end)))
    print("Target num sentences: {}".format(num_sentences))
    end_lengths = np.ones(len(end))
    end_lengths = end_lengths.astype(np.int)
    print("end_lengths :{}".format(end_lengths))

    bridge_feats = simulate_brownian_bridge(
        B_0=true_cl_feats[0], B_T=B_T, num_samples=num_sentences,
        sentence_lengths=end_lengths
    ) #根据起始状态和最终状态生成整个布朗桥

    bridge_feats = torch.tensor(
        bridge_feats, dtype=true_cl_feats.dtype).to(args.device)
    # RANDOM
    random_feats = torch.rand(true_cl_feats.shape).to(args.device) # 随机生成布朗桥
    feats = [true_cl_feats, bridge_feats, random_feats]

    for seq_i, seq_cl_feats in enumerate(feats[1:2]):
        cl_feats = seq_cl_feats[0]  # Get the first sentence feat

        # RESET THE CL INDEX
        model.transformer._cur_cl_idx = 0
        model.transformer._has_reset = False

        if args.method == "sample":
            output_sequences = model.generate(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                cl_feats=cl_feats,  # .to(args.device),
                seq_cl_feats=seq_cl_feats,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                bad_words_ids=args.bad_words_ids,
                # List of token ids that are not allowed to be generated. In order to get the
                # tokens of the words that should not appear in the generated text,
                # use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`
                min_length=args.min_length
            )

        # # NOTE GREEDY
        elif args.method == "greedy":
            output_sequences = model.generate(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                cl_feats=cl_feats,  # .to(args.device),
                seq_cl_feats=seq_cl_feats,
                max_length=args.max_length,
                num_return_sequences=args.num_return_sequences,
            )

        # # NOTE Beam search
        elif args.method == "beam":
            output_sequences = model.generate(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                cl_feats=cl_feats,  # .to(args.device),
                seq_cl_feats=seq_cl_feats,
                max_length=args.max_length,
                num_beams=5,
                early_stopping=True,
                num_return_sequences=args.num_return_sequences,
                # no_repeat_ngram_size=2, # To avoid repetition
            )
        else:
            raise ValueError("need to specify --method")

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generate_res = []
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):

            print('output_sequences.shape :{}'.format(output_sequences.shape))

            generated_sequence = generated_sequence.tolist()

            # Decode text
            # text = tokenizer.decode(generated_sequence,
            # =True)
            text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            print('generated_sequence: {}'.format(generated_sequence))
            print('before generate text: {}'.format(text))

            # Remove all text after the stop token
            stop_idx = []
            for stop in args.stop_token:
                if text.find(stop) != -1:
                    if text.find(stop) != 0:
                        stop_idx.append(text.find(stop))
                    else:
                        text = text[len(stop)+1:]
                        stop_idx.append(text.find(stop))
            stop_idx.sort()
            print('stop_idx :{}'.format(stop_idx))
            text = text[:stop_idx[0]]

            generate_res.append(text)
        return generate_res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 1.0 has no effect, lower tend toward greedy sampling", )
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--num-sentences", type=int, default=0)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--multiply-sentences", type=int, default=1)
    parser.add_argument("--p", type=float, default=0.99)
    parser.add_argument("--block_size", type=int, default=1024)

    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")

    parser.add_argument("--suppress_eos", action="store_true", default=False, help="Text added prior to input.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--dataset_name", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--fixed_prompt", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--num_intervals", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--encoder_filepath", type=str, required=True, default="", help="Text added prior to input.")
    parser.add_argument("--latent_dim", type=int, default=3, help="random seed for initialization")
    parser.add_argument("--use_random_embs", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--use_true_end_latent", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--method", type=str, default="", help="Text added prior to input.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.use_section_null = 0

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}")

    set_seed(args)

    # Initialize the model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = GPT2TimeLMHeadModel.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    model.transformer._config.use_contrastive_embeddings = True

    SECTION_IDS, SPECIAL_TOKENS, tokenizer = get_special_tokens(
        dataset_name=args.dataset_name, tokenizer=tokenizer)

    tokenizer.eos_token = '[ <|endoftext|> ]'
    tokenizer.bos_token = '[ <|endoftext|> ]'
    tokenizer.pad_token = tokenizer.eos_token

    print('tokenizer.eos_token_id:{}'.format(tokenizer.eos_token_id))
    print('tokenizer.bos_token_id:{}'.format(tokenizer.bos_token_id))

    if args.suppress_eos:
        args.bad_words_ids = [[tokenizer.eos_token_id]]  # 指定那些id不生成
    else:
        args.bad_words_ids = None

    model.transformer.special_tokens = SECTION_IDS
    cl_model = get_checkpoint(
        latent_dim=args.latent_dim,
        use_section_ids=True,
        token_size=len(tokenizer),
        filepath=args.encoder_filepath
    )
    cl_model.to(args.device)
    cl_model.eval()

    model.transformer._config.use_noisy_embeddings = False
    logger.info(args)

    args.prompt_text = args.prompt if args.prompt else ""  # input("Model prompt >>> ")
    args.stop_token = [
            '[ user ]',
            '[ assistant ]',
            '[ <|endoftext|> ]'
        ]

    print(f'Args: {args}')

    train_dataset = EekeDataset(
        tokenizer=tokenizer,
        special_words=SPECIAL_TOKENS,
        block_size=args.block_size,
        cl_model=cl_model,
        data_names='train'
    )

    # Estimate dnesity for last sentence
    _, _, last_latent_mu, last_latent_std = get_density(dataset=train_dataset, lm=model, cl_model=cl_model)

    print("last latent mu", last_latent_mu)
    print("last latent std", last_latent_std)

    history = []
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        if raw_text == 'exit':
            break
        history.append(raw_text)
        out_text = generter(model, history, cl_model, tokenizer, SECTION_IDS, args, last_latent_mu, last_latent_std)
        print('out_text: {}'.format(out_text))
        history.append(out_text[0])



if __name__ == "__main__":
    main()
