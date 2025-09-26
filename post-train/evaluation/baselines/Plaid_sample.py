import contextlib
import fire
import mup
import numpy as np
import lib.datasets
import lib.models
import lib.utils
import os, json, re
from datasets import load_dataset
import time
import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim, autograd
from f1 import compute_f1, normalize_answer

def main(**args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = lib.utils.AttributeDict(args)
    args.setdefault('seq_len', 256)
    args.setdefault('vocab_size', 32768)
    args.setdefault('weights_path', None)
    args.setdefault('dim', 2048)
    args.setdefault('n_blocks', 24)
    args.setdefault('n_heads', 32)
    args.setdefault('gamma_0', -3.)
    args.setdefault('gamma_1', 6.)
    args.setdefault('embed_dim', 16)
    args.setdefault('initial_noise_scale', 1.0)
    args.setdefault('n_samples', 8)
    args.setdefault('sampling_timesteps', 16)
    args.setdefault('score_temp', 0.9)
    args.setdefault('output_scale', 1.)
    args.setdefault('owt2_tokenizer', True)
    args.setdefault('ddim_sampler', False)
    args.setdefault('guidance_weight', 2.)

    lib.utils.print_args(args)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_default_device('cuda')

    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp64 by default and explicitly switch to fp32/bf16 where
    # appropriate.
    torch.set_default_dtype(torch.float64)

    def log1mexp(x):
        # Computes log(1-exp(-|x|))
        x = -x.abs()
        return torch.where(
            x > -0.693,
            torch.log(-torch.expm1(x)),
            torch.log1p(-torch.exp(x))
        )

    def create_modules(dim, n_heads):
        return {
            'noise_schedule': lib.models.NoiseSchedule().float(),
            'gamma_bounds': lib.models.GammaBounds(args.gamma_0, args.gamma_1).float(),
            'embedding_matrix': lib.models.EmbeddingMatrix(args.vocab_size, args.embed_dim).float(),
            'model': lib.models.DiffusionModel(dim, args.embed_dim, args.n_blocks, n_heads, args.vocab_size).float()
        }
    modules = create_modules(args.dim, args.n_heads)
    base_modules = create_modules(256, 4)
    delta_modules = create_modules(128, 2)
    for key in modules:
        main, base, delta = modules[key], base_modules[key], delta_modules[key]
        mup.set_base_shapes(main, base, delta=delta)
        main.cuda()

    print(f'Loading weights from {args.weights_path}')
    for name, module in modules.items():
        module.load_state_dict(torch.load(
            os.path.join(args.weights_path, f'{name}.pt'),
            map_location=torch.device('cuda')
        ))

    for key in modules:
        print(key+':')
        lib.utils.print_model(modules[key])


    def generate_samples(guidance_tokens, seq_len=args.seq_len):
        """
        Sampling (implements Appendix A.4 eqn 33 in VDM). Needs float64 to work.
        guidance_tokens: [(token, weight, position, complement), ...]
            token: vocab index of token
            weight: guidance weight
            position: sequence index, or 'any', or 'all'
            complement: if True, do guidance on log(1-p(y|x))
        """
        with torch.no_grad():
            embedding_matrix = modules['embedding_matrix']()

            gamma_0, gamma_1 = modules['gamma_bounds']()
            alpha_0 = torch.sigmoid(-gamma_0).sqrt()
            sigma_0 = torch.sigmoid(gamma_0).sqrt()

            z = torch.randn((args.n_samples, seq_len, args.embed_dim), device='cuda') * args.initial_noise_scale
            x_selfcond = torch.zeros_like(z).float()
            for i, t in enumerate(tqdm.tqdm(torch.linspace(1., 0., args.sampling_timesteps))):
                t = t[None].cuda()
                s = t - 1. / args.sampling_timesteps
                gamma_s = modules['noise_schedule'](s).double()
                gamma_t = modules['noise_schedule'](t).double()
                gamma_s = gamma_0 + (gamma_1 - gamma_0) * gamma_s
                gamma_t = gamma_0 + (gamma_1 - gamma_0) * gamma_t
                alpha_squared_s = torch.sigmoid(-gamma_s)
                alpha_squared_t = torch.sigmoid(-gamma_t)
                alpha_s = alpha_squared_s.sqrt()
                alpha_t = alpha_squared_t.sqrt()
                sigma_squared_s = torch.sigmoid(gamma_s)
                sigma_squared_t = torch.sigmoid(gamma_t)
                sigma_s = sigma_squared_s.sqrt()
                sigma_t = sigma_squared_t.sqrt()

                if len(guidance_tokens) > 0:
                    with torch.enable_grad():
                        z.requires_grad = True
                        logits, x_reconst = modules['model'](
                            z=z.to(torch.float32, copy=True),
                            gamma=gamma_t.float(),
                            embedding_matrix=embedding_matrix,
                            bias_scale=1.,
                            x_selfcond=x_selfcond
                        )

                        logprobs = F.log_softmax(logits.float(), dim=2)
                        logprobs_any = logprobs.logsumexp(dim=1)-float(seq_len)

                        sum_logp = 0.
                        for token, weight, position, complement in guidance_tokens:
                            if position == 'any':
                                logp = logprobs_any[:, token]
                            elif position == 'all':
                                logp = logprobs[:, :, token]
                            else:
                                logp = logprobs[:, position, token]
                            if complement:
                                logp = log1mexp(logp)
                            sum_logp += weight * logp.sum()

                        guidance_grad = autograd.grad(sum_logp, [z])[0]
                        z.requires_grad = False
                    x_selfcond = x_reconst.clone().detach()
                    x_reconst = x_reconst.double()
                    epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
                    epsilon_pred /= args.score_temp
                    x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t
                    x_reconst += guidance_grad.double() * sigma_squared_t / alpha_squared_t.sqrt()
                    epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
                else:
                    _, x_reconst = modules['model'](
                        z=z.to(torch.float32, copy=True),
                        gamma=gamma_t.float(),
                        embedding_matrix=embedding_matrix,
                        bias_scale=1.,
                        x_selfcond=x_selfcond
                    )
                    x_selfcond = x_reconst.clone().detach()
                    x_reconst = x_reconst.double()
                    epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
                    epsilon_pred /= args.score_temp
                    x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t
                if t > 0:
                    if args.ddim_sampler:
                        z = (alpha_s * x_reconst) + (sigma_s * epsilon_pred)
                    else:
                        c = -torch.expm1(gamma_s - gamma_t)
                        z *= (1 - c) * alpha_squared_s.sqrt() / alpha_squared_t.sqrt()
                        z += c * (alpha_squared_s.sqrt() * x_reconst.double())
                        z += (c * (1 - alpha_squared_s)).sqrt() * torch.randn_like(z)

            logits, _ = modules['model'](
                z=z.float(),
                gamma=gamma_t.float(),
                embedding_matrix=embedding_matrix,
                bias_scale=1.,
                x_selfcond=x_selfcond
            )
            x_samples = logits.argmax(dim=-1)

            return x_samples


    def eval_score(seq_id, src_mask):
        with torch.no_grad():
            embedding_matrix = modules['embedding_matrix']()
            x_embed = embedding_matrix[torch.tensor([seq_id])]

            gamma_0, gamma_1 = modules['gamma_bounds']()
            alpha_0 = torch.sigmoid(-gamma_0).sqrt()
            sigma_0 = torch.sigmoid(gamma_0).sqrt()
            seq_len = len(seq_id)
            score = 0.

            noise = torch.randn((1, seq_len, args.embed_dim), device='cuda') * args.initial_noise_scale
            x_selfcond = torch.zeros_like(noise).float()
            for i, t in enumerate(tqdm.tqdm(torch.linspace(1., 0., args.sampling_timesteps))):
                # import pdb; pdb.set_trace();
                t = t[None].cuda()
                s = t - 1. / args.sampling_timesteps
                gamma_s = modules['noise_schedule'](s).double()
                gamma_t = modules['noise_schedule'](t).double()
                gamma_s = gamma_0 + (gamma_1 - gamma_0) * gamma_s
                gamma_t = gamma_0 + (gamma_1 - gamma_0) * gamma_t
                alpha_squared_s = torch.sigmoid(-gamma_s)
                alpha_squared_t = torch.sigmoid(-gamma_t)
                alpha_s = alpha_squared_s.sqrt()
                alpha_t = alpha_squared_t.sqrt()
                sigma_squared_s = torch.sigmoid(gamma_s)
                sigma_squared_t = torch.sigmoid(gamma_t)
                sigma_s = sigma_squared_s.sqrt()
                sigma_t = sigma_squared_t.sqrt()

                z = torch.add(torch.mul(noise, sigma_t[:,None,None]), alpha_t[:,None,None] * x_embed)
                src_mask_tensor = torch.tensor([src_mask], dtype=torch.bool).unsqueeze(-1)
                z = torch.where(src_mask_tensor, x_embed, z)

                logits, x_reconst = modules['model'](
                    z=z.to(torch.float32, copy=True),
                    gamma=gamma_t.float(),
                    embedding_matrix=embedding_matrix,
                    bias_scale=1.,
                    x_selfcond=x_selfcond
                )
                x_selfcond = x_reconst.clone().detach()
                x_reconst = x_reconst.double()
                epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
                epsilon_pred /= args.score_temp
                x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t

                logprobs = F.log_softmax(logits.float(), dim=2)

                sum_logp = 0.
                position = 0
                cnt = 0
                for token, mask in zip(seq_id, src_mask):
                    if mask == 0:
                        logp = logprobs[:, position, token]
                        sum_logp += logp.sum()
                        cnt += 1
                    position += 1
                sum_logp /= cnt
                score += sum_logp
            return sum_logp/args.sampling_timesteps

    def print_samples(x_samples):
        if args.owt2_tokenizer:
            owt2_tokenizer = lib.datasets.openwebtext2_tokenizer()
            with open(f'un-generation-t{args.sampling_timesteps}.jsonl', "a", encoding="utf-8") as writer:
                for x in x_samples:
                    x = owt2_tokenizer.decode(x.tolist(), skip_special_tokens=False)
                    # print(x.replace("\n", "↵"))
                    writer.write(json.dumps({"predict": x.replace("\n", "↵")}, ensure_ascii=False) + '\n')
        else:
            for x in x_samples:
                x = x.tolist()
                x = [idx2word[i].decode('utf-8', 'ignore') for i in x]
                x = ' '.join(x)
                x = x.replace('START','')
                x = x.replace('END','')
                x = x.replace('PAD','')
                x = x.replace(' .', '.')
                x = x.replace(' !', '!')
                x = x.replace(' ,', ',')
                x = x.replace(' \' ', '\'')
                x = x.strip()
                # replace newlines with '↵' symbol for cleaner printing
                print(x.replace("\n", "↵"))

    tokenizer = lib.datasets.openwebtext2_tokenizer()

    def un_gen():
        print('Unconditional:')
        print_samples(generate_samples([], seq_len=1024))
        print("\n"*10)

    def eval_lamb():
        total_cnt = 0
        cor = 0
        
        with open('lambada_test_plain_text.txt', 'r', encoding='utf-8') as file:
            for line in file:
                total_cnt += 1
                line = line.strip()
                prefix = ' '.join(line.split()[:-1])
                x0 = tokenizer.encode(line).ids
                print('Prefix completion: ', prefix)
                prefix = tokenizer.encode(prefix).ids
                masked_nums = len(x0)-len(prefix)
                x_samples = generate_samples(
                    [(token, args.guidance_weight, position, False) for position, token in enumerate(prefix)], seq_len=len(prefix)+masked_nums
                )
                owt2_tokenizer = lib.datasets.openwebtext2_tokenizer()
                pred = owt2_tokenizer.decode(x_samples[0].tolist()[len(prefix):], skip_special_tokens=False)
                
                if pred.strip() == line.split()[-1].strip():
                    cor += 1
                print(total_cnt, cor/total_cnt) 
                # print_samples(x_samples)
        print("\n"*10)
        print('acc:', cor/total_cnt)

    import csv, json, evaluate

    def eval_infilling():
        problems = []
        with open(f"cloze_test_val__spring2016.csv") as f:
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
        owt2_tokenizer = lib.datasets.openwebtext2_tokenizer()

        for stories in problems:
            total_cnt += 1
            # import pdb; pdb.set_trace();
            prompt = stories[0] + " " + stories[1]
            suffix = stories[3] + " " + stories[4]
            middle = stories[2]

            prefix = tokenizer.encode(prompt).ids
            suff = tokenizer.encode(suffix).ids
            midd = tokenizer.encode(middle).ids
            
            x_samples = generate_samples(
                [(token, args.guidance_weight, position, False) for position, token in enumerate(prefix)]
                + [(token, args.guidance_weight, position + len(prefix) + len(midd), False) for position, token in enumerate(suff)],
                seq_len=len(prefix)+len(midd)+len(suff)
            )
            
            pred = owt2_tokenizer.decode(x_samples[0].tolist()[len(prefix):len(prefix) + len(midd)], skip_special_tokens=False)
                
            # pred = tokenizer.decode(res.tolist()[0][len(prefix)-1:len(x0)-len(suff)-1])
        
            samples.append(dict(pred=pred, label=middle, prefix=prompt, suffix=suffix))
            gens.append(pred)
            refs.append(middle)

            if total_cnt == 1000:
                break

        # rouge = evaluate.load("rouge", cache_dir='/apdcephfs_qy3/share_733425/victoriabi/sansagong/huggingface/cache/')
        # results = rouge.compute(predictions=gens, references=refs)
        # for key in results.keys():
        #     results[key] *= 100
        # results["rougeAvg"] = (results["rouge1"] + results["rouge2"] + results["rougeL"]) / 3
        # print(f"rouge1={results['rouge1']:.2f}, rouge2={results['rouge2']:.2f}, rougeL={results['rougeL']:.2f}, rougeAvg={results['rougeAvg']:.2f}")


        with open(f'ROCInfill_t{args.sampling_timesteps}.jsonl', 'w') as f:
            for json_obj in samples:
                f.write(json.dumps(json_obj) + '\n')

    def preprocess(text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def evalhellaSwag():
        ds = load_dataset("../../huggingface/cache/Rowan___hellaswag/default/0.1.0/6002345709e0801764318f06bf06ce1e7d1a1fe3/", split='validation', cache_dir='/apdcephfs_qy3/share_733425/victoriabi/sansagong/huggingface/cache/')

        total_cnt = 0
        cor = 0

        owt2_tokenizer = lib.datasets.openwebtext2_tokenizer()

        for doc in ds:
            total_cnt += 1
            ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()

            query = preprocess(doc["activity_label"] + ": " + ctx)
            choices = [preprocess(ending) for ending in doc["endings"]]
            gold = int(doc["label"])

            score_list = []
            prefix = owt2_tokenizer.encode(query).ids
            # import pdb; pdb.set_trace();

            for c in choices:
                x0 = prefix + tokenizer.encode(c).ids
                src_mask = [1]*len(prefix)+[0]*(len(x0)-len(prefix))
                score = eval_score(x0, src_mask)
                # import pdb; pdb.set_trace();
                score_list.append(score.tolist())
            pred = np.argmax(np.array(score_list))
            if pred == gold:
                cor += 1
            print(query, choices, pred, gold)

        print('acc:', cor/total_cnt)


    def evalWino():
        
        ds = load_dataset("../../huggingface/cache/allenai___winogrande/winogrande_xl/1.1.0/a826c3d3506aefe0e9e9390dcb53271070536586bab95849876b2c1743df56e2/", "winogrande_xl", split='validation', cache_dir='/apdcephfs_qy3/share_733425/victoriabi/sansagong/huggingface/cache/')

        total_cnt = 0
        cor = 0
        tokenizer = lib.datasets.openwebtext2_tokenizer()

        for doc in ds:
            total_cnt += 1
            # import pdb; pdb.set_trace();
            idx = doc["sentence"].index("_")
        
            options = [doc["option1"], doc["option2"]]
            
            target = " " + doc["sentence"][idx+1:].strip()
            target_id = tokenizer.encode(target).ids

            answer_to_num = {"1": 0, "2": 1}
            gold = answer_to_num[doc["answer"]]

            score_list = []
            
            for opt in options:
                prefix = doc["sentence"][:idx] + opt
                prefix_id = tokenizer.encode(prefix).ids

                x0 = prefix_id + target_id
                src_mask = [1]*len(prefix_id)+[0]*(len(x0)-len(prefix_id))
                score = eval_score(x0, src_mask)
                # import pdb; pdb.set_trace();
                score_list.append(score.tolist())
            pred = np.argmax(np.array(score_list))
            if pred == gold:
                cor += 1
            print(doc["sentence"], options, pred, gold)

        print('acc:', cor/total_cnt, total_cnt)

    def eval_piqa():
        from datasets import load_dataset
        ds = load_dataset("../../huggingface/cache/ybisk___piqa/plain_text/1.1.0/6c611c1a9bf220943c4174e117d3b660859665baf1d43156230116185312d011/", split='validation', cache_dir='/apdcephfs_qy3/share_733425/victoriabi/sansagong/huggingface/cache/')
        total_cnt = 0
        cor = 0
        tokenizer = lib.datasets.openwebtext2_tokenizer()

        for doc in ds:
            total_cnt += 1
            
            query = f"Question: {doc['goal']}\nAnswer: "
            choices = [doc["sol1"], doc["sol2"]]
            gold = doc["label"]

            score_list = []
            prefix = tokenizer.encode(query).ids

            for choice in choices:

                x0 = prefix + tokenizer.encode(" " + choice).ids
                src_mask = [1]*len(prefix)+[0]*(len(x0)-len(prefix))
                score = eval_score(x0, src_mask)
                # import pdb; pdb.set_trace();
                score_list.append(score.tolist())
            pred = np.argmax(np.array(score_list))

            if pred == gold:
                cor += 1
            # print(total_cnt, cor/total_cnt)
            
        print('acc:', cor/total_cnt)  

    def eval_triva():
        ds = load_dataset("mandarjoshi/trivia_qa", "rc", split='validation', cache_dir='/apdcephfs_qy3/share_733425/victoriabi/sansagong/huggingface/cache/')
        gens = []
        refs = []
        total_cnt = 0
        cor = 0
        tokenizer = lib.datasets.openwebtext2_tokenizer()

        for doc in ds:
            total_cnt += 1
            # import pdb; pdb.set_trace();
            total_cnt += 1
            # import pdb; pdb.set_trace();
            query = f"Question: {doc['question']}?\nAnswer: "
            labels = doc["answer"]["aliases"]
            normal_labels = doc["answer"]["normalized_aliases"]
            encoded_labels = [tokenizer.encode(l, add_special_tokens=False).ids for l in labels]
            long_gold = max(encoded_labels, key=len)

            input_ids = tokenizer.encode(query).ids
            full = long_gold
            tokens = len(full)
            
            x_samples = generate_samples(
                [(token, args.guidance_weight, position, False) for position, token in enumerate(input_ids)],
                seq_len=len(input_ids)+tokens
            )
            
            pred = owt2_tokenizer.decode(x_samples[0].tolist()[len(input_ids):], skip_special_tokens=False)
            
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

    # eval_lamb()
    # eval_infilling()
    # un_gen()
    # evalhellaSwag()
    # evalWino()
    # eval_piqa()
    eval_triva()


    # print('Infilling: A year ago in Paris, [...] Wow, what a great day!')
    # tokenizer = lib.datasets.openwebtext2_tokenizer()
    # prefix = tokenizer.encode(' A year ago in Paris,').ids
    # suffix = tokenizer.encode('. Wow, what a great day!').ids
    # infill_len = 40
    # print_samples(generate_samples(
    #     [(token, args.guidance_weight, position, False) for position, token in enumerate(prefix)]
    #     + [(token, args.guidance_weight, position + len(prefix) + infill_len, False) for position, token in enumerate(suffix)]
    # ))
    # print("\n"*10)

    # print('Word-level weights: Let\'s talk about law[10] and medicine[1].')
    # guidance = [
    #     (tokenizer.encode(' Let').ids,      args.guidance_weight,   0,  False),
    #     (tokenizer.encode('\'s').ids,       args.guidance_weight,   1,  False),
    #     (tokenizer.encode(' talk').ids,     args.guidance_weight,   2,  False),
    #     (tokenizer.encode(' about').ids,    args.guidance_weight,   3,  False),
    #     (tokenizer.encode(' law').ids,      10.,                    4,  False),
    #     (tokenizer.encode(' and').ids,      args.guidance_weight,   5,  False),
    #     (tokenizer.encode(' medicine').ids, args.guidance_weight,   6,  False),
    #     (tokenizer.encode('.').ids,         args.guidance_weight,   7,  False),
    # ]
    # assert(all(len(a) == 1 for a,_,_,_ in guidance))
    # guidance = [(a[0], b, c, d) for a,b,c,d in guidance]
    # print_samples(generate_samples(guidance))
    # print('\n'*10)

    # print('Word-level weights: Let\'s talk about law[1] and medicine[10].')
    # guidance = [
    #     (tokenizer.encode(' Let').ids,      args.guidance_weight,   0,  False),
    #     (tokenizer.encode('\'s').ids,       args.guidance_weight,   1,  False),
    #     (tokenizer.encode(' talk').ids,     args.guidance_weight,   2,  False),
    #     (tokenizer.encode(' about').ids,    args.guidance_weight,   3,  False),
    #     (tokenizer.encode(' law').ids,      args.guidance_weight,   4,  False),
    #     (tokenizer.encode(' and').ids,      args.guidance_weight,   5,  False),
    #     (tokenizer.encode(' medicine').ids, 10.,                    6,  False),
    #     (tokenizer.encode('.').ids,         args.guidance_weight,   7,  False),
    # ]
    # assert(all(len(a) == 1 for a,_,_,_ in guidance))
    # guidance = [(a[0], b, c, d) for a,b,c,d in guidance]
    # print_samples(generate_samples(guidance))
    # print('\n'*10)

    # print(f'Lexically constrained generation: Donald')
    # guidance = [
    #     (tokenizer.encode(' Donald').ids, 3., 'any', False),
    # ]
    # assert(all(len(a) == 1 for a,_,_,_ in guidance))
    # guidance = [(a[0], b, c, d) for a,b,c,d in guidance]
    # print_samples(generate_samples(guidance))
    # print("\n"*10)

    # print(f'Negation: Donald but not Trump')
    # guidance = [
    #     (tokenizer.encode(' Donald').ids, 3., 'any', False),
    #     (tokenizer.encode(' Trump').ids, 10., 'all', True),
    # ]
    # assert(all(len(a) == 1 for a,_,_,_ in guidance))
    # guidance = [(a[0], b, c, d) for a,b,c,d in guidance]
    # print_samples(generate_samples(guidance))
    # print("\n"*10)


if __name__ == '__main__':
    fire.Fire(main)