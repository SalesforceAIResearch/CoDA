import torch
from transformers import AutoModelForCausalLM, GPT2Tokenizer, AutoTokenizer
import torch.nn.functional as F

from datasets import load_dataset
import re
import numpy as np
from f1 import compute_f1, normalize_answer

model_name = '/app/qi/backup/data/Megatron/LLaMA-Factory-clean/output/gpt2-pt-2/checkpoint-1000'  # 'gpt2-medium', 'gpt2-large'
# model_name = 'meta-llama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained('/app/qi/backup/data/Megatron/vlm/huggingface/cache/models--gpt2-medium/snapshots/6dcaa7a952f72f9298047fd5137cd6e4f05f41da')
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', _attn_implementation="sdpa", torch_dtype=torch.bfloat16, eos_token_id=None)
model.eval()
# input_ids = tokenizer.encode("", return_tensors='pt').to(model.device)
# import time, json

# start_time = time.time()
# outputs = model.generate(input_ids, max_length=512, num_return_sequences=1)
# end_time = time.time()
# print(len(outputs[0].tolist()))
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# with open(f"ar_un_gen_{model_name.replace('/', '-')}.jsonl", 'a') as f:
#     f.write(json.dumps({"predict": generated_text}) + '\n')
# gpu_memory_allocated = torch.cuda.memory_allocated()
# gpu_memory_reserved = torch.cuda.memory_reserved()
# iteration_time = end_time - start_time

# print(f"Time = {iteration_time:.4f} seconds, "
#     f"GPU Memory Allocated = {gpu_memory_allocated / 1e6:.2f} MB, "
#     f"GPU Memory Reserved = {gpu_memory_reserved / 1e6:.2f} MB")

def eval_lambada():

    total_cnt = 0
    cor = 0
    with open('../evaluation/lambada_test_plain_text.txt', 'r', encoding='utf-8') as file:
        for line in file:
            total_cnt += 1
            line = line.strip()
            # import pdb; pdb.set_trace();
            input_prefix = ' '.join(line.split()[:-1])
            input_ids = tokenizer.encode(input_prefix, return_tensors='pt').to(model.device)
            full = tokenizer.encode(line, return_tensors='pt')
            tokens = full.shape[1] - input_ids.shape[1]
            outputs = model.generate(input_ids, max_length=input_ids.shape[1] + tokens, num_return_sequences=1, do_sample=False)
            generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            pred = generated_text

            if pred.strip() == line.split()[-1].strip():
                cor += 1
            # print(total_cnt, cor/total_cnt)

    print('acc:', cor/total_cnt)




def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def evalhellaSwag():
    ds = load_dataset("Rowan/hellaswag", split='validation')

    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()

        query = preprocess(doc["activity_label"] + ": " + ctx)
        choices = [preprocess(ending) for ending in doc["endings"]]
        gold = int(doc["label"])

        scores = []
        question = tokenizer.encode(query, add_special_tokens=False)

        for c in choices:

            # encode full sentences, questions, and answers, no padding for simplicity
            sentence = question + tokenizer.encode(c, add_special_tokens=False)
            
            sentence = torch.tensor([sentence]).to(model.device)
            # import pdb; pdb.set_trace(); 
            # run the model on full sentences and get the log probabilities
            logprobs = F.log_softmax(model(sentence)['logits'], dim=-1).cpu()

            # take log probabilities corresponding to possible answer tokens
            logprobs = logprobs[0][len(question) - 1 : -1, :]

            # get the scores by summing log probabilities corresponding to correct answer tokens, unvectorized

            answer = sentence[0][len(question):].cpu()
            # guess = logprobs.argmax(dim=-1)
            # print(tokenizer.decode(guess), bool((guess == answer).all()))
            scores.append(float(torch.gather(logprobs, 1, answer.unsqueeze(-1)).mean()))

        # predict the answer
        pred = np.argmax(np.array(scores))
        if pred == gold:
            cor += 1

    print('acc:', cor/total_cnt)


def evalWino():
    
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split='validation', trust_remote_code=True)

    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        # import pdb; pdb.set_trace();
        idx = doc["sentence"].index("_")
        options = [doc["option1"], doc["option2"]]
        
        answer_to_num = {"1": 0, "2": 1}
        gold = answer_to_num[doc["answer"]]

        scores = []

        for opt in options:
            target = opt + " " + doc["sentence"][idx+1:].strip()
            target_id = tokenizer.encode(target, add_special_tokens=False)

            prefix = doc["sentence"][:idx]
            prefix_id = tokenizer.encode(prefix, add_special_tokens=False)

            # encode full sentences, questions, and answers, no padding for simplicity
            sentence = prefix_id + target_id
            
            sentence = torch.tensor([sentence]).to(model.device)
            # import pdb; pdb.set_trace(); 
            # run the model on full sentences and get the log probabilities
            logprobs = F.log_softmax(model(sentence)['logits'], dim=-1).cpu()

            # take log probabilities corresponding to possible answer tokens
            logprobs = logprobs[0][len(prefix_id) - 1 : -1, :]

            # get the scores by summing log probabilities corresponding to correct answer tokens, unvectorized

            answer = sentence[0][len(prefix_id):].cpu()
            # guess = logprobs.argmax(dim=-1)
            # print(tokenizer.decode(guess), bool((guess == answer).all()))
            scores.append(float(torch.gather(logprobs, 1, answer.unsqueeze(-1)).mean()))

        # predict the answer
        pred = np.argmax(np.array(scores))
        if pred == gold:
            cor += 1

    print('acc:', cor/total_cnt, total_cnt)


def evalPIQA():
    
    ds = load_dataset("ybisk/piqa", split='validation', trust_remote_code=True)

    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        # import pdb; pdb.set_trace();
        query = f"Question: {doc['goal']}\nAnswer: "
        choices = [doc["sol1"], doc["sol2"]]
        gold = doc["label"]

        scores = []
        question = tokenizer.encode(query, add_special_tokens=False)

        for c in choices:

            # encode full sentences, questions, and answers, no padding for simplicity
            sentence = question + tokenizer.encode(c, add_special_tokens=False)
            
            sentence = torch.tensor([sentence]).to(model.device)
            # import pdb; pdb.set_trace(); 
            # run the model on full sentences and get the log probabilities
            logprobs = F.log_softmax(model(sentence)['logits'], dim=-1).cpu()

            # take log probabilities corresponding to possible answer tokens
            logprobs = logprobs[0][len(question) - 1 : -1, :]

            # get the scores by summing log probabilities corresponding to correct answer tokens, unvectorized

            answer = sentence[0][len(question):].cpu()
            # guess = logprobs.argmax(dim=-1)
            # print(tokenizer.decode(guess), bool((guess == answer).all()))
            scores.append(float(torch.gather(logprobs, 1, answer.unsqueeze(-1)).sum()))

        # predict the answer
        pred = np.argmax(np.array(scores))
        if pred == gold:
            cor += 1

    print('acc:', cor/total_cnt, total_cnt)


def evalSIQA():
    
    ds = load_dataset("allenai/social_i_qa", split='validation', trust_remote_code=True)
    
    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        # import pdb; pdb.set_trace();
        query = f"Question: {doc['context']} {doc['question']}\nAnswer: "
        choices = [doc['answerA'], doc['answerB'], doc['answerC']]
        gold = int(doc["label"]) - 1

        scores = []
        question = tokenizer.encode(query, add_special_tokens=False)

        for c in choices:

            # encode full sentences, questions, and answers, no padding for simplicity
            sentence = question + tokenizer.encode(c, add_special_tokens=False)
            
            sentence = torch.tensor([sentence]).to(model.device)
            # import pdb; pdb.set_trace(); 
            # run the model on full sentences and get the log probabilities
            logprobs = F.log_softmax(model(sentence)['logits'], dim=-1).cpu()

            # take log probabilities corresponding to possible answer tokens
            logprobs = logprobs[0][len(question) - 1 : -1, :]

            # get the scores by summing log probabilities corresponding to correct answer tokens, unvectorized

            answer = sentence[0][len(question):].cpu()
            # guess = logprobs.argmax(dim=-1)
            # print(tokenizer.decode(guess), bool((guess == answer).all()))
            scores.append(float(torch.gather(logprobs, 1, answer.unsqueeze(-1)).sum()))

        # predict the answer
        pred = np.argmax(np.array(scores))
        if pred == gold:
            cor += 1

    print('acc:', cor/total_cnt, total_cnt)

import csv, json
import evaluate

def eval_infilling():
    problems = []
    with open(f"../evaluation/cloze_test_val__spring2016.csv") as f:
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
        x0 = prefix
        outputs = model.generate(torch.tensor([x0]).to(model.device), max_length=len(prefix)+len(middle), num_return_sequences=1, do_sample=False)
        pred = tokenizer.decode(outputs.tolist()[0][len(prefix):])
    
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

    with open(f"ROCInfill-{model_name.replace('/', '-')}.jsonl", 'w') as f:
        for json_obj in samples:
            f.write(json.dumps(json_obj) + '\n')

def eval_humaneval():

    from human_eval_infilling.data import write_jsonl, read_problems

    subtasks = "single-line"
    problems = read_problems(benchmark_name=subtasks)
    samples_res = []
    for task_id in problems:
        # import pdb; pdb.set_trace();
        prompt = problems[task_id]["prompt"]
        suffix = problems[task_id]["suffix"]
        middle = problems[task_id]["canonical_solution"]
        print(task_id, prompt)

        prompt = f"The prefix is: {prompt} and the suffix is: {suffix}, please return the middle code: "

        # if "81" in task_id:
        #     samples_res.append(dict(task_id=task_id, completion=""))
        #     continue

        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        suff = tokenizer.encode(suffix, add_special_tokens=False)
        x0 = prefix
        outputs = model.generate(torch.tensor([x0]).to(model.device), max_length=len(prefix)+len(middle), num_return_sequences=1, do_sample=False)
        pred = tokenizer.decode(outputs.tolist()[0][len(prefix):])
    
        samples_res.append(dict(task_id=task_id, completion=pred))

    write_jsonl(f"instruct-humaneval_{model_name.replace('/', '-')}_{subtasks}.jsonl", samples_res)
    

def eval_trivaqa():
    total_cnt = 0
    cor = 0

    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split='validation')
    predictions = []
    references = []

    for doc in ds:
        total_cnt += 1
        # import pdb; pdb.set_trace();
        query = f"Question: {doc['question']}?\nAnswer: "
        labels = doc["answer"]["aliases"]
        normal_labels = doc["answer"]["normalized_aliases"]
        encoded_labels = [tokenizer.encode(l, add_special_tokens=False) for l in labels]
        long_gold = max(encoded_labels, key=len)

        input_ids = tokenizer.encode(query, return_tensors='pt').to(model.device)
        full = long_gold
        tokens = len(full)
        outputs = model.generate(input_ids, max_length=input_ids.shape[1] + tokens, num_return_sequences=1, do_sample=False)
        generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        pred = generated_text

        if normalize_answer(pred.strip()) in normal_labels:
            cor += 1
        else:
            for l in normal_labels:
                if l in normalize_answer(pred.strip()):
                    cor += 1
                    break
        
        predictions.append(pred)
        references.append(labels)

        if total_cnt == 2000:
            break

        print(pred, labels)

    print('em acc:', cor/total_cnt)
    print(compute_f1(predictions, references))

# eval_lambada()
evalhellaSwag()
evalWino()
evalSIQA()
evalPIQA()
# eval_infilling()
# eval_trivaqa()
# eval_humaneval()