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
    SECTION_IDS = []
    if 'erke' in dataset_name:
        SECTION_IDS = [
            '[ user ]',
            '[ assistant ]',
            '<|endoftext|>'
        ]

    SECTION_IDS += [' . ']
    if add_tokens:
        # NOTE loading previous tokenizer sometimes already includes the new tokens
        eos = tokenizer(' . ')['input_ids']
        print("Old tokenizer size: ", len(tokenizer))
        if len(eos) == 1 and eos[0] == 21128 + len(SECTION_IDS):
            print('========================================================================')
            print("Not adding because it's already contained")
            pass  # don't add cause it's already contained
        else:
            print("Adding tokens, ", SECTION_IDS)
            tokenizer.add_tokens(SECTION_IDS)
        print("New tokenizer size: ", len(tokenizer))
    SPECIAL_TOKENS = [_[1] for _ in tokenizer(SECTION_IDS)['input_ids']]
    return SECTION_IDS, SPECIAL_TOKENS, tokenizer


def get_checkpoint(dataset_name, latent_dim, base_model="gpt2",
                   use_section_ids=False, token_size=None,
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

def get_cl_embeddings(history, cl_model, tokenizer, special_tokens, device):
    full_text = ""
    cl_text = []
    for i, utterance in enumerate(history):
        if i % 2 == 0:
            text = "[ {} ] {}".format('USER', utterance)
            full_text += text + " "
            cl_text.append(text)
        else:
            text = "[ {} ] {}".format('ASSISTANT', utterance)
            full_text += text + " "
            cl_text.append(text)

    row = f"{tokenizer.bos_token} {full_text} {tokenizer.eos_token}"
    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(row))
    example = tokenizer.build_inputs_with_special_tokens(tokenized_text)

    cl_embeddings = []
    eos_idxs = []
    for tok in special_tokens[:2]:
        eos_idxs += [i - 1 for i, x in enumerate(example) if x == tok]
    eos_idxs += [len(example)]
    eos_idxs.sort()
    eos_idxs = eos_idxs[1:]

    assert len(eos_idxs) == len(cl_text)

    cl_input_ids, cl_attention_mask = cl_tokenize(cl_text, device)
    cl_feats = cl_model.forward(
        input_ids=cl_input_ids, attention_mask=cl_attention_mask)  # 1, feat_size
    last_idx = 0
    for eos_idx, feat in zip(eos_idxs, cl_feats):
        cl_embeddings += [feat] * (eos_idx - last_idx)
        last_idx = eos_idx

    return cl_embeddings, eos_idxs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model")
    parser.add_argument("--train_path", default=None, type=str, required=True, help="Path train data")
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
    parser.add_argument("--split-sentences", type=int, default=1)
    parser.add_argument("--multiply-sentences", type=int, default=1)
    parser.add_argument("--p", type=float, default=0.99)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
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
    parser.add_argument("--label", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--method", type=str, default="", help="Text added prior to input.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.use_section_null = 0

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    # Initialize the model and tokenizer
    model_class = GPT2TimeLMHeadModel
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    model.transformer._config.use_contrastive_embeddings = True

    SECTION_WOEDS, SPECIAL_TOKENS, tokenizer = get_special_tokens(
        dataset_name=args.dataset_name, tokenizer=tokenizer)

    tokenizer.eos_token = '<|endoftext|>'
    tokenizer.bos_token = '<|endoftext|>'
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(
        '<|endoftext|>')  # 修改id 50256->21130  hugging默认是50256，即英文模型大小
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')  # 修改id 50256->21130

    if args.suppress_eos:
        bad_words_ids = [[tokenizer.eos_token_id]]  # 指定那些id不生成
    else:
        bad_words_ids = None

    model.transformer.special_tokens = SPECIAL_TOKENS
    base_model = 'gpt2'
    CL_MODEL = get_checkpoint(
        dataset_name=args.dataset_name,
        latent_dim=args.latent_dim,
        use_section_ids=True,
        token_size=len(tokenizer),
        base_model=base_model,
        filepath=args.encoder_filepath
    )
    CL_MODEL.to(args.device)
    CL_MODEL.eval()

    model.transformer._config.use_noisy_embeddings = False
    logger.info(args)

    prompt_text = args.prompt if args.prompt else ""  # input("Model prompt >>> ")

    print(f'Args: {args}')

    train_dataset = EekeDataset(
        tokenizer=tokenizer,
        special_words=SECTION_WOEDS,
        block_size=args.block_size,
        cl_model=CL_MODEL,
        data_names='train'
    )

    max_length = args.max_length
    min_length = args.min_length

    # Estimate dnesity for last sentence
    _, _, last_latent_mu, last_latent_std = get_density(dataset=train_dataset, lm=model, cl_model=CL_MODEL)

    print("last latent mu", last_latent_mu)
    print("last latent std", last_latent_std)

    history = []
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history.append(raw_text)

        # Get all the CL feats
        cl_embeddings, end = get_cl_embeddings(history, CL_MODEL, tokenizer, SECTION_WOEDS, args.device)
        true_cl_feats = torch.stack(cl_embeddings)
        true_cl_feats = true_cl_feats[::args.split_sentences]

        LABELS = ['TRUE CL', 'BRIDGE CL (DE)', 'RANDOM CL']
        # INTERPOLATION - BRIDGE
        B_T = np.random.normal(loc=last_latent_mu, scale=last_latent_std)

        num_sentences = len(true_cl_feats)

        print("Original num sentences: {}".format(len(end)))
        print("Target num sentences: {}".format(num_sentences))
        end_lengths = [end[i] if i == 0 else end[i + 1] - end[i] for i in range(len(end) - 1)] # [19,49,89,102,112] -> [19,40,13,10]
        end_lengths = (np.array(end_lengths) * (num_sentences / len(end)))
        end_lengths = np.ones(end_lengths.shape)
        end_lengths = end_lengths.astype(np.int)

        bridge_feats = simulate_brownian_bridge(
            B_0=true_cl_feats[0], B_T=B_T, num_samples=num_sentences,
            sentence_lengths=end_lengths
        )

        bridge_feats = torch.tensor(
            bridge_feats, dtype=true_cl_feats.dtype).to(args.device)
        # RANDOM
        random_feats = torch.rand(true_cl_feats.shape).to(args.device)
        feats = [true_cl_feats, bridge_feats, random_feats]

        for seq_i, seq_cl_feats in enumerate(feats):
            cl_feats = seq_cl_feats[0]  # Get the first sentence feat
            prefix = args.prefix if args.prefix else args.padding_text
            encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=True, return_tensors="pt")
            encoded_prompt = encoded_prompt.to(args.device)

            if encoded_prompt.size()[-1] == 0:
                input_ids = None
            else:
                input_ids = encoded_prompt

            # RESET THE CL INDEX
            model.transformer._cur_cl_idx = 0
            model.transformer._has_reset = False

            if args.method == "sample":
                output_sequences = model.generate(
                    input_ids=input_ids,
                    section_ids=None,
                    cl_feats=cl_feats,  # .to(args.device),
                    seq_cl_feats=seq_cl_feats,
                    max_length=max_length,
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                    bad_words_ids=bad_words_ids,
                    # List of token ids that are not allowed to be generated. In order to get the
                    # tokens of the words that should not appear in the generated text,
                    # use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`
                    min_length=min_length
                )

            # # NOTE GREEDY
            elif args.method == "greedy":
                output_sequences = model.generate(
                    input_ids=input_ids,
                    section_ids=None,
                    cl_feats=cl_feats,  # .to(args.device),
                    seq_cl_feats=seq_cl_feats,
                    max_length=max_length,
                    num_return_sequences=args.num_return_sequences,
                )

            # # NOTE Beam search
            elif args.method == "beam":
                output_sequences = model.generate(
                    input_ids=input_ids,
                    section_ids=None,
                    cl_feats=cl_feats,  # .to(args.device),
                    seq_cl_feats=seq_cl_feats,
                    max_length=max_length,
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

            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                # print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
                generated_sequence = generated_sequence.tolist()

                print("Generated length: {}".format(len(generated_sequence)))

                # Decode text
                # text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                text = tokenizer.decode(generated_sequence, skip_special_tokens=True)

                # Remove all text after the stop token
                text = text[: text.find(args.stop_token) if args.stop_token else None]

                print('==============generated sequence: {}================='.format(text))

                # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                total_sequence = (
                        prompt_text + text[len(tokenizer.decode(encoded_prompt[0], skip_special_tokens=True)):]
                )

                print("[ GENERATED FOR {} ]: {}".format(LABELS[seq_i], total_sequence))


if __name__ == "__main__":
    main()
