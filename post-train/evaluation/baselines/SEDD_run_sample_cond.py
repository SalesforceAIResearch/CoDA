import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
import sampling
from sampling import AnalyticPredictor
from model import utils as mutils

import re
import numpy as np
import csv, json
import evaluate
from f1 import compute_f1, normalize_answer


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-small", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--prefix", type=str, default="Hi, my name is")
    parser.add_argument("--suffix", type=str, default=" and that's why I'm late.")
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, graph, noise = load_model(args.model_path, device)

    mask_id = graph.dim - 1 

    def eval_lambada():
        total_cnt = 0
        cor = 0
        with open('../evaluation/lambada_test_plain_text.txt', 'r', encoding='utf-8') as file:
            for line in file:
                total_cnt += 1
                line = line.strip()
                
                xt = tokenizer(line).input_ids
                input_ids = tokenizer(' '.join(line.split()[:-1])).input_ids
                masked_nums = len(xt)-len(input_ids)
                # xt[-masked_nums:] = [mask_id] * masked_nums
                # xt = torch.tensor([xt]).to(device)

                # denoiser = AnalyticPredictor(graph, noise)
                # sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)

                # t = (masked_nums/xt.shape[1]) * torch.ones(xt.shape[0], 1, device=device)
                # x0 = denoiser.update_fn(sampling_score_fn, xt, t, 1/xt.shape[1])

                input_locs = list(range(len(input_ids)))
                input_ids = torch.tensor(input_ids, device="cuda")[None]

                def proj_fun(x):
                    x[:, input_locs] = input_ids
                    return x
                sampling_fn = sampling.get_pc_sampler(
                    graph, noise, (1, len(xt)), 'analytic', masked_nums, device=device, proj_fun=proj_fun
                )
                samples = proj_fun(sampling_fn(model))
                # import pdb; pdb.set_trace();
                pred = tokenizer.decode(samples.tolist()[0][-masked_nums:])

                if pred.strip() == line.split()[-1].strip():
                    cor += 1
        print('acc:', cor/total_cnt)

    def uncon_gen():
        prefix_ids = tokenizer(args.prefix).input_ids
        suffix_ids = tokenizer(args.suffix).input_ids
        input_ids = prefix_ids + suffix_ids
        input_locs = list(range(len(prefix_ids))) + list(range(1024-len(suffix_ids), 1024))

        # more generaly commands can be defined with something like below:
        # input_ids = [0, 1, 512, 8080, 50256, 20000]
        # input_locs = [5, 6, 19, 20, 1000, 10001]


        input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(args.batch_size, 1)

        def proj_fun(x):
            x[:, input_locs] = input_ids
            return x
        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device, proj_fun=proj_fun
        )

        samples = proj_fun(sampling_fn(model))

        text_samples = tokenizer.batch_decode(samples)
        for i in text_samples:
            print(i)
            print("=================================================")

    eps = 1e-5
    steps = 32

    def evalHellaSwag():
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
            prefix = tokenizer(query).input_ids

            for choice in choices:

                x0 = prefix + tokenizer(choice).input_ids
                # masked_nums = len(x0) - len(prefix)
                # x0[-masked_nums:] = [mask_id] * masked_nums
                x0 = torch.tensor([x0]).to(device)
                timesteps = torch.linspace(1, eps, steps + 1, device=device)
                score = 0

                for i in range(steps):
                    t = timesteps[i] * torch.ones(x0.shape[0], device=device)

                    sigma, dsigma = noise(t)
        
                    xt = graph.sample_transition(x0, sigma[:, None])
                    for i, token in enumerate(prefix):
                        xt[:,i] = token

                    # import pdb; pdb.set_trace();

                    log_score_fn = mutils.get_score_fn(model, train=False, sampling=False)
                    log_score = log_score_fn(xt, sigma)
                    loss = graph.score_entropy(log_score, sigma[:, None], xt, x0)
                    score += (dsigma[:, None] * loss).sum(dim=-1).tolist()[0]

                # import pdb; pdb.set_trace();
                score_list.append(score/steps/len(tokenizer(choice).input_ids))
            pred = np.argmin(np.array(score_list))

            if pred == gold:
                cor += 1
            # print(total_cnt, cor/total_cnt)
            
        print('acc:', cor/total_cnt) 

    def evalWino():
        from datasets import load_dataset
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split='validation', trust_remote_code=True)

        total_cnt = 0
        cor = 0

        for doc in ds:
            total_cnt += 1

            idx = doc["sentence"].index("_")
            target = " " + doc["sentence"][idx+1:].strip()
            target_id = tokenizer.encode(target, add_special_tokens=False)
            options = [doc["option1"], doc["option2"]]
            
            answer_to_num = {"1": 0, "2": 1}
            gold = answer_to_num[doc["answer"]]

            score_list = []
            

            for opt in options:

                prefix = doc["sentence"][:idx] + opt
                prefix_id = tokenizer.encode(prefix, add_special_tokens=False)

                x0 = prefix_id + target_id
                # masked_nums = len(x0) - len(prefix)
                # x0[-masked_nums:] = [mask_id] * masked_nums
                x0 = torch.tensor([x0]).to(device)
                timesteps = torch.linspace(1, eps, steps + 1, device=device)
                score = 0

                for i in range(steps):
                    t = timesteps[i] * torch.ones(x0.shape[0], device=device)

                    sigma, dsigma = noise(t)
        
                    xt = graph.sample_transition(x0, sigma[:, None])
                    for i, token in enumerate(prefix_id):
                        xt[:,i] = token

                    # import pdb; pdb.set_trace();

                    log_score_fn = mutils.get_score_fn(model, train=False, sampling=False)
                    log_score = log_score_fn(xt, sigma)
                    loss = graph.score_entropy(log_score, sigma[:, None], xt, x0)
                    score += (dsigma[:, None] * loss).sum(dim=-1).tolist()[0]

                # import pdb; pdb.set_trace();
                score_list.append(score/steps)
            pred = np.argmin(np.array(score_list))

            if pred == gold:
                cor += 1
            # print(total_cnt, cor/total_cnt)
            
        print('acc:', cor/total_cnt) 

    def evalPIQA():
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
            prefix = tokenizer(query).input_ids

            for choice in choices:

                x0 = prefix + tokenizer(choice).input_ids
                # masked_nums = len(x0) - len(prefix)
                # x0[-masked_nums:] = [mask_id] * masked_nums
                x0 = torch.tensor([x0]).to(device)
                timesteps = torch.linspace(1, eps, steps + 1, device=device)
                score = 0

                for i in range(steps):
                    t = timesteps[i] * torch.ones(x0.shape[0], device=device)

                    sigma, dsigma = noise(t)
        
                    xt = graph.sample_transition(x0, sigma[:, None])
                    for i, token in enumerate(prefix):
                        xt[:,i] = token

                    # import pdb; pdb.set_trace();

                    log_score_fn = mutils.get_score_fn(model, train=False, sampling=False)
                    log_score = log_score_fn(xt, sigma)
                    loss = graph.score_entropy(log_score, sigma[:, None], xt, x0)
                    score += (dsigma[:, None] * loss).sum(dim=-1).tolist()[0]

                # import pdb; pdb.set_trace();
                score_list.append(score/steps)
            pred = np.argmin(np.array(score_list))

            if pred == gold:
                cor += 1
            # print(total_cnt, cor/total_cnt)
            
        print('acc:', cor/total_cnt) 

    def evalSIQA():
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
            prefix = tokenizer(query).input_ids

            for choice in choices:

                x0 = prefix + tokenizer(choice).input_ids
                # masked_nums = len(x0) - len(prefix)
                # x0[-masked_nums:] = [mask_id] * masked_nums
                x0 = torch.tensor([x0]).to(device)
                timesteps = torch.linspace(1, eps, steps + 1, device=device)
                score = 0

                for i in range(steps):
                    t = timesteps[i] * torch.ones(x0.shape[0], device=device)

                    sigma, dsigma = noise(t)
        
                    xt = graph.sample_transition(x0, sigma[:, None])
                    for i, token in enumerate(prefix):
                        xt[:,i] = token

                    # import pdb; pdb.set_trace();

                    log_score_fn = mutils.get_score_fn(model, train=False, sampling=False)
                    log_score = log_score_fn(xt, sigma)
                    loss = graph.score_entropy(log_score, sigma[:, None], xt, x0)
                    score += (dsigma[:, None] * loss).sum(dim=-1).tolist()[0]

                # import pdb; pdb.set_trace();
                score_list.append(score/steps)
            pred = np.argmin(np.array(score_list))

            if pred == gold:
                cor += 1
            # print(total_cnt, cor/total_cnt)
            
        print('acc:', cor/total_cnt) 


    def eval_infilling():
        problems = []
        with open(f"../evaluation/cloze_test_val__spring2016.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                sents = row[1:-3] + [row[-3] if row[-1] == "1" else row[-2]]
                # sents = [s if i == 0 else " " + s for i, s in enumerate(sents)]
                problems.append(sents)
        
        sample_res = []
        total_cnt = 0
        gens = []
        refs = []

        for stories in problems:
            total_cnt += 1
            # import pdb; pdb.set_trace();
            prompt = stories[0] + " " + stories[1]
            suffix = stories[3] + " " + stories[4]
            middle = stories[2]

            prefix_ids = tokenizer(prompt).input_ids
            suffix_ids = tokenizer(suffix).input_ids
            middle_ids = tokenizer(middle).input_ids
            input_ids = prefix_ids + suffix_ids
            total_len = len(prefix_ids)+len(middle_ids)+len(suffix_ids)
            input_locs = list(range(len(prefix_ids))) + list(range(len(prefix_ids)+len(middle_ids), total_len))

            input_ids = torch.tensor(input_ids, device="cuda")[None]

            def proj_fun(x):
                x[:, input_locs] = input_ids
                return x
            
            sampling_fn = sampling.get_pc_sampler(
                graph, noise, (1, total_len), 'analytic', steps, device=device, proj_fun=proj_fun
            )

            samples = proj_fun(sampling_fn(model))

            
            pred = tokenizer.decode(samples.tolist()[0][len(prefix_ids):len(prefix_ids)+len(middle_ids)])

            sample_res.append(dict(pred=pred, label=middle, prefix=prompt, suffix=suffix))

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


        with open(f'ROCInfill_small_t{steps}.jsonl', 'w') as f:
            for json_obj in sample_res:
                f.write(json.dumps(json_obj) + '\n')

    def eval_triva():
        from datasets import load_dataset
        ds = load_dataset("mandarjoshi/trivia_qa", "rc", split='validation')
        
        gens = []
        refs = []
        total_cnt = 0
        cor = 0

        for doc in ds:
            total_cnt += 1
            # import pdb; pdb.set_trace();
            query = f"Question: {doc['question']}?\nAnswer: "
            labels = doc["answer"]["aliases"]
            normal_labels = doc["answer"]["normalized_aliases"]
            encoded_labels = [tokenizer(l).input_ids for l in labels]
            long_gold = max(encoded_labels, key=len)

            full = long_gold
            
            input_ids = tokenizer(query).input_ids
            masked_nums = len(full) + 10
            
            input_locs = list(range(len(input_ids)))
            input_ids = torch.tensor(input_ids, device="cuda")[None]

            def proj_fun(x):
                x[:, input_locs] = input_ids
                return x
            sampling_fn = sampling.get_pc_sampler(
                graph, noise, (1, input_ids.shape[1]+masked_nums), 'analytic', masked_nums, device=device, proj_fun=proj_fun
            )
            samples = proj_fun(sampling_fn(model))
            # import pdb; pdb.set_trace();
            pred = tokenizer.decode(samples.tolist()[0][-masked_nums:])

            if normalize_answer(pred.strip()) in normal_labels:
                cor += 1
            else:
                for l in normal_labels:
                    if l in normalize_answer(pred.strip()):
                        cor += 1
                        break
                    
            gens.append(pred)
            refs.append(normal_labels)

            if total_cnt == 2000:
                break

            print(pred, labels)

        print('em acc:', cor/total_cnt)
        print(compute_f1(gens, refs))


    def humaneval_infill():
        from human_eval_infilling.data import write_jsonl, read_problems

        subtasks = "single-line"
        problems = read_problems(benchmark_name=subtasks)
        samples_res = []
        for task_id in problems:
            # import pdb; pdb.set_trace();
            prompt = problems[task_id]["prompt"]
            suffix = problems[task_id]["suffix"]
            middle = problems[task_id]["canonical_solution"]

            if "81" in task_id:
                samples_res.append(dict(task_id=task_id, completion=""))
                continue

            prefix_ids = tokenizer(prompt).input_ids
            suffix_ids = tokenizer(suffix).input_ids
            middle_ids = tokenizer(middle).input_ids
            input_ids = prefix_ids + suffix_ids
            total_len = len(prefix_ids)+len(middle_ids)+len(suffix_ids)
            input_locs = list(range(len(prefix_ids))) + list(range(len(prefix_ids)+len(middle_ids), total_len))

            input_ids = torch.tensor(input_ids, device="cuda")[None]

            def proj_fun(x):
                x[:, input_locs] = input_ids
                return x
            
            sampling_fn = sampling.get_pc_sampler(
                graph, noise, (1, total_len), 'analytic', steps, device=device, proj_fun=proj_fun
            )

            samples = proj_fun(sampling_fn(model))

            pred = tokenizer.decode(samples.tolist()[0][len(prefix_ids):len(prefix_ids)+len(middle_ids)])
        
            samples_res.append(dict(task_id=task_id, completion=pred))

        write_jsonl(f"humaneval_small_samplingv1_{subtasks}.jsonl", samples_res)
    
    evalHellaSwag()
    # eval_lambada()
    # evalWino()
    # evalSIQA()
    # eval_infilling()
    # eval_triva()
    # humaneval_infill()


if __name__=="__main__":
    main()