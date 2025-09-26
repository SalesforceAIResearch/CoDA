import torch
from attention_patch import replace_attention_mask

replace_attention_mask()

from llamafactory.train.ddm.trainer import eval_forward, generate_samples, generate_samples_v2
from model import DiscreteDiffusionModel
from argparse import ArgumentParser

from transformers import AutoConfig, AutoTokenizer

import torch.distributions as dists
import torch.nn.functional as F
from f1 import compute_f1, normalize_answer

def get_anneal_attn_mask(seq_len, bsz, dtype, device, attn_mask_ratio):
    mask = torch.full((seq_len, seq_len), 0, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
    causal_mask = mask.to(dtype)
    
    random_mask = torch.bernoulli(torch.full((seq_len, seq_len), 0.0, device=device) + attn_mask_ratio)

    anneal_mask = torch.logical_or(causal_mask, random_mask)
    expanded_mask = anneal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.to(dtype)

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def top_p_logits(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # import pdb; pdb.set_trace();
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def eval_Lambada(model, tokenizer, args):
    total_cnt = 0
    cor = 0
    with open('evaluation/lambada_test_plain_text.txt', 'r', encoding='utf-8') as file:
        for line in file:
            total_cnt += 1
            line = line.strip()
            # import pdb; pdb.set_trace();
            x0 = tokenizer.encode(line)
            prefix = tokenizer.encode(' '.join(line.split()[:-1]))
            # attention_mask = get_anneal_attn_mask(len(xt), 1, dtype=model.lm_head.weight.dtype, device=model.device, attn_mask_ratio=1.0)
            # masked_nums = len(xt)-len(inputs)
            # xt[-masked_nums:] = [tokenizer.mask_token_id] * masked_nums
            # xt = torch.tensor([xt]).to(model.device)
            # logits = model(xt, attention_mask=attention_mask)
            # filter_logits = top_p_logits(logits/0.8, p=0.8)
            # scores = torch.log_softmax(filter_logits, dim=-1)
            # # x0_scores, x0 = scores.max(-1)
            # x0 = dists.Categorical(logits=scores).sample()
            # pred = tokenizer.decode(x0.tolist()[0][-masked_nums-1:-1])

            masked_nums = len(x0)-len(prefix)
            src_mask = [1]*len(prefix)+[0]*masked_nums
            inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
            args.diffusion_steps = masked_nums
            args.logits_temp = 1.0
            res = generate_samples(model, args, tokenizer, inputs, eval=True)
            pred = tokenizer.decode(res.tolist()[0][-masked_nums:])
            # import pdb; pdb.set_trace();
            if pred.strip() == line.split()[-1].strip():
                cor += 1
            # print(total_cnt, cor/total_cnt)
    print('acc:', cor/total_cnt)

import re
import numpy as np

def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def eval_hellaswag(model, tokenizer, args):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split='validation')

    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()

        query = preprocess(doc["activity_label"] + ": " + ctx)
        choices = [preprocess(ending) for ending in doc["endings"]]
        gold = int(doc["label"])

        score_list = []
        prefix = tokenizer.encode(query)

        for choice in choices:

            x0 = prefix + tokenizer.encode(choice)
            src_mask = [1]*len(prefix)+[0]*(len(x0)-len(prefix))
            inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
            score = eval_forward(model, inputs, args, tokenizer)
            # import pdb; pdb.set_trace();
            score_list.append(score.tolist())
        pred = np.argmin(np.array(score_list))

        if pred == gold:
            cor += 1
        # print(total_cnt, cor/total_cnt)
        
    print('acc:', cor/total_cnt)  


def eval_wino(model, tokenizer, args):
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split='validation', trust_remote_code=True)

    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        
        idx = doc["sentence"].index("_")
        
        options = [doc["option1"], doc["option2"]]

        answer_to_num = {"1": 0, "2": 1}
        gold = answer_to_num[doc["answer"]]

        score_list = []
        
        for opt in options:
            target = opt 
            suffix = doc["sentence"][idx+1:].strip()
            target_id = tokenizer.encode(target, add_special_tokens=False)
            suffix_id = tokenizer.encode(suffix, add_special_tokens=False)
            prefix = doc["sentence"][:idx]
            prefix_id = tokenizer.encode(prefix, add_special_tokens=False)

            x0 = prefix_id + target_id + suffix_id
            src_mask = [1]*len(prefix_id)+[0]*(len(x0)-len(prefix_id))
            inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
            score = eval_forward(model, inputs, args, tokenizer)
            import pdb; pdb.set_trace();
            score_list.append(score.tolist())
        pred = np.argmin(np.array(score_list))

        if pred == gold:
            cor += 1
        # print(total_cnt, cor/total_cnt)
        
    print('acc:', cor/total_cnt)  

def eval_piqa(model, tokenizer, args):
    from datasets import load_dataset
    ds = load_dataset("ybisk/piqa", split='validation', trust_remote_code=True)
    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        
        query = f"Question: {doc['goal']}\nAnswer: "
        choices = [doc["sol1"], doc["sol2"]]
        gold = doc["label"]

        score_list = []
        prefix = tokenizer.encode(query)

        for choice in choices:

            x0 = prefix + tokenizer.encode(" " + choice)
            src_mask = [1]*len(prefix)+[0]*(len(x0)-len(prefix))
            inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
            score = eval_forward(model, inputs, args, tokenizer)
            # import pdb; pdb.set_trace();
            score_list.append(score.tolist())
        pred = np.argmin(np.array(score_list))

        if pred == gold:
            cor += 1
        # print(total_cnt, cor/total_cnt)
        
    print('acc:', cor/total_cnt)  

def eval_siqa(model, tokenizer, args):
    from datasets import load_dataset
    ds = load_dataset("allenai/social_i_qa", split='validation', trust_remote_code=True)
    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        
        query = f"Question: {doc['context']} {doc['question']}\nAnswer: "
        choices = [doc['answerA'], doc['answerB'], doc['answerC']]
        gold = int(doc["label"]) - 1

        score_list = []
        prefix = tokenizer.encode(query, add_special_tokens=False)

        for choice in choices:

            x0 = prefix + tokenizer.encode(choice, add_special_tokens=False)
            src_mask = [1]*len(prefix)+[0]*(len(x0)-len(prefix))
            inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
            score = eval_forward(model, inputs, args, tokenizer)
            # import pdb; pdb.set_trace();
            score_list.append(score.tolist())
        pred = np.argmin(np.array(score_list))

        if pred == gold:
            cor += 1
        # print(total_cnt, cor/total_cnt)
        
    print('acc:', cor/total_cnt)

import csv, json
import evaluate

def eval_infilling(model, tokenizer, args):
    problems = []
    with open(f"evaluation/cloze_test_val__spring2016.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            sents = row[1:-3] + [row[-3] if row[-1] == "1" else row[-2]]
            # sents = [s if i == 0 else " " + s for i, s in enumerate(sents)]
            problems.append(sents)
    
    samples = []
    total_cnt = 0
    gens = []
    refs = []

    for stories in problems:
        total_cnt += 1
        # import pdb; pdb.set_trace();
        prompt = stories[0] + " " + stories[1]
        suffix = stories[3] + " " + stories[4]
        middle = stories[2]

        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        suff = tokenizer.encode(suffix, add_special_tokens=False)
        x0 = prefix + tokenizer.encode(middle, add_special_tokens=False) + suff
        src_mask = [1]*len(prefix)+[0]*(len(x0)-len(prefix)-len(suff))+[1]*len(suff)
        inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
        res = generate_samples(model, args, tokenizer, inputs, eval=True)
        pred = tokenizer.decode(res.tolist()[0][len(prefix)-1:len(x0)-len(suff)-1])
    
        samples.append(dict(pred=pred, label=middle, prefix=prompt, suffix=suffix))
        gens.append(pred)
        refs.append(middle)

        if total_cnt == 1000:
            break

    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=gens, references=refs)
    for key in results.keys():
        results[key] *= 100
    results["rougeAvg"] = (results["rouge1"] + results["rouge2"] + results["rougeL"]) / 3
    print(f"rouge1={results['rouge1']:.2f}, rouge2={results['rouge2']:.2f}, rougeL={results['rougeL']:.2f}, rougeAvg={results['rougeAvg']:.2f}")


    with open(f'ROCInfill_medium_t{args.diffusion_steps}_tmp{args.logits_temp}.jsonl', 'w') as f:
        for json_obj in samples:
            f.write(json.dumps(json_obj) + '\n')

def humaneval_infill(model, tokenizer, args):
    from human_eval_infilling.data import write_jsonl, read_problems

    subtasks = "single-line"
    problems = read_problems(benchmark_name=subtasks)
    samples = []
    for task_id in problems:
        # import pdb; pdb.set_trace();
        prompt = problems[task_id]["prompt"]
        suffix = problems[task_id]["suffix"]
        middle = problems[task_id]["canonical_solution"]

        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        suff = tokenizer.encode(suffix, add_special_tokens=False)
        x0 = prefix + tokenizer.encode(middle, add_special_tokens=False) + suff
        src_mask = [1]*len(prefix)+[0]*(len(x0)-len(prefix)-len(suff))+[1]*len(suff)
        if len(x0) > 1000:
            print(task_id)
            continue
        inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
        res = generate_samples(model, args, tokenizer, inputs, eval=True)
        pred = tokenizer.decode(res.tolist()[0][len(prefix)-1:len(x0)-len(suff)-1])
    
        samples.append(dict(task_id=task_id, completion=pred))

    write_jsonl(f"humaneval_medium_samplingv1_{subtasks}.jsonl", samples)
    
def eval_triva(model, tokenizer, args):
    from datasets import load_dataset
    # ds = load_dataset("mandarjoshi/trivia_qa", "rc", split='validation')
    ds = load_dataset("rajpurkar/squad", split='validation')
    gens = []
    refs = []
    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        # import pdb; pdb.set_trace();
        query = f"{doc['context']}\nQuesion{doc['question']}?\nAnswer: "
        labels = doc["answers"]["text"]
        encoded_labels = [tokenizer.encode(l, add_special_tokens=False) for l in labels]
        long_gold = max(encoded_labels, key=len)

        input_ids = tokenizer.encode(query)
        full = long_gold
        tokens = len(full)
        
        x0 = input_ids + [0]*(tokens)
        src_mask = [1]*len(input_ids)+[0]*(tokens)
        args.diffusion_steps = tokens

        inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
        res = generate_samples(model, args, tokenizer, inputs, eval=True)
        pred = tokenizer.decode(res.tolist()[0][len(input_ids)-1:])
        

        for l in labels:
            if normalize_answer(l) in normalize_answer(pred.strip()):
                cor += 1
                break
                
        gens.append(pred)
        refs.append(labels)

        if total_cnt == 2000:
            break

        print(pred, labels)

    print('em acc:', cor/total_cnt)
    print(compute_f1(gens, refs))

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default='diffusionfamily/diffugpt-s')
    parser.add_argument("--base_model_name", type=str, default='gpt2')
    parser.add_argument("--shift", type=bool, default=True) # do not change this
    parser.add_argument("--diffusion_steps", type=int, default=32)
    parser.add_argument("--logits_temp", type=float, default=0.95)
    parser.add_argument("--topp_temp", type=float, default=0.9)
    parser.add_argument("--verbose", type=bool, default=False) # print middle state

    args = parser.parse_args()

    # model_name = 'gpt2'  # 'gpt2-medium', 'gpt2-large'
    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # model = DiscreteDiffusionModel(args.base_model_name, config, tokenizer)
    model = DiscreteDiffusionModel.from_pretrained(
        model_name, 
        model=args.base_model_name, 
        config=config, 
        tokenizer=tokenizer,
        device='cuda'
    ).to('cuda')

    eval_Lambada(model, tokenizer, args)
    eval_hellaswag(model, tokenizer, args)
    humaneval_infill(model, tokenizer, args)
    eval_infilling(model, tokenizer, args)
    eval_wino(model, tokenizer, args)
    eval_siqa(model, tokenizer, args)
    eval_piqa(model, tokenizer, args)
    eval_triva(model, tokenizer, args)

if __name__ == "__main__":
    main()