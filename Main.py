import torch
import numpy as np
import random
import argparse
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

from SVD4LLAMA import SVD4LLAMAMain

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='PIQA', help='The sampled dataset')
    parser.add_argument('--sampled_size', type=int, default=128, help='The sampled data number')
    parser.add_argument('--compress_ratio', type=float, default=0.25, help='total compress ratio of model, including sparsity_ratio')
    parser.add_argument('--sparsity_ratio', type=float, default=0.05, help='ratio of delta matrix')
    parser.add_argument('--not_delta', action='store_false', default=True, help='whether to use delta matrix')
    parser.add_argument('--not_eval_uncompressed_model', action='store_false', default=True, help='whether to evaluate the uncompressed model')
    parser.add_argument('--not_eval_compressed_model', action='store_false', default=True, help='whether to evaluate the compressed model')
    parser.add_argument('--data_save_path', type=str, default='./models/save')
    parser.add_argument("--data_root", type=str, default="",help="the path to load data")
    parser.add_argument("--model_path", type=str, default="",help="the path to load model")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()

def get_model_and_tokenizer(model_path):
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False,padding_side="left",trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    special_tokens = tokenizer.all_special_tokens
    print(f"Special tokens: {special_tokens[:10]}{'...' if len(special_tokens) > 10 else ''}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    print("Model loading complete")
    return model, tokenizer

def check_data_exist(data_name):
    file_path = os.path.join("./DataLoad", f"{data_name}.py")
    return os.path.exists(file_path)

def main(args):
    if hasattr(args, 'seed'):
        set_seed(args.seed)
    else:
        print("=" * 10 + "warning: There is no seed variable to control the random number seed!" + "=" * 10)

    try:
        model, tokenizer = get_model_and_tokenizer(args.model_path)
    except:
        print("Can't load model!")
        raise

    if args.data in {'MATHQA','ARCE','WINOG','PIQA','OPENB','HELLAS'}:
        print(f"Loading {args.data}")
        SVD4LLAMAMain(args, model, tokenizer)
    elif check_data_exist(args.data):
        print(f"Loading {args.data}")
        SVD4LLAMAMain(args, model, tokenizer)
    else:
        print("Data not support!")

class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

if __name__ == '__main__':
    args = parse_args()
    if args.not_delta:
        args.compress_ratio += args.sparsity_ratio
        assert args.compress_ratio < 1
    args.data_save_path += '/' + args.data + '_' + str(args.compress_ratio*100) + '_' + str(args.sparsity_ratio*100)
    sys.stdout = Tee(args.data_save_path + '/output.txt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    torch.set_printoptions(precision=8)
    main(args)
