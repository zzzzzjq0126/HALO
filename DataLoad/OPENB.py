import pandas as pd
import torch
import numpy as np

def load_dataset():
    data_path = "/cephfs/shared/wjj2/openb/test-00000-of-00001.parquet"
    df = pd.read_parquet(data_path)
    return df

def build_prompt(item):
    return item['question_stem']

def doc_to_choice(item):
    return item["choices"]["text"]

def generate_output(model, tokenizer, prompt, choices, args):
    probs = []
    for choice in choices:
        probs.append(loglikelihood(model, tokenizer, prompt, choice, args))
    pred = np.argmax(probs)
    print(pred, probs)
    return pred

def loglikelihood(model, tokenizer, prompt, choice, args):
    input_ids = tokenizer.encode(prompt + " " + choice, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    continuation_start = len(tokenizer.encode(prompt))
    continuation_ids = input_ids[:,continuation_start:]
    selected_log_probs = log_probs[:, continuation_start - 1:-1, :].gather(-1, continuation_ids.unsqueeze(-1)).squeeze(-1)
    total_log_prob = selected_log_probs.sum().item()
    return total_log_prob

def evalutate_tatal_dataset(dataset, model, tokenizer, args):
    corr = 0
    total = dataset.shape[0]

    for _, item in dataset.iterrows():
        try:
            prompt = build_prompt(item)
            choices = doc_to_choice(item)
            if args.use_logits_to_generete_output:
                pred = generate_output(model, tokenizer, prompt, choices, args)
            else:
                pass
            if item["choices"]["label"][pred] == item["answerKey"].lstrip():
                corr += 1
        except Exception as e:
            print(f"\nError processing sample: {str(e)}")
            print("item",item)
            import traceback
            traceback.print_exc()
            raise e
    accuracy = corr / total if total > 0 else 0.0
    print(f'Correct-{corr}, Total-{total}, Accuracy-{100 * accuracy:.4f}%')


def sample_dataset_for_svd(args):
    path = "/cephfs/shared/wjj2/openb/train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    sample_dataset = []

    for _ in range(args.sampled_size):
        try:
            sampled_rows = df.sample(n=30)

            prompts = []
            for _, item in sampled_rows.iterrows():
                prompt = build_prompt(item)
                prompts.append(prompt)
            combined_prompt = "\n".join(prompts)
            sample_dataset.append(combined_prompt)

        except Exception as e:
            print(f"\nError processing sample: {str(e)}")
            import traceback
            traceback.print_exc()

    return sample_dataset