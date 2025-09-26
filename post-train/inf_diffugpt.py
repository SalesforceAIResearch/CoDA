import torch
from transformers import AutoConfig, AutoTokenizer
import torch.nn.functional as F
from argparse import ArgumentParser


from model import DiscreteDiffusionModel, generate_samples

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default='diffusionfamily/diffugpt-s')
    parser.add_argument("--base_model_name", type=str, default='gpt2')
    parser.add_argument("--shift", type=bool, default=True) # do not change this
    parser.add_argument("--diffusion_steps", type=int, default=64)
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

    # import pdb; pdb.set_trace();

    gen_len = config.task_specific_params['text-generation']['max_length']
    print("="*20, "Generating...", gen_len)
    # un-conditional generation
    print("="*20, "Unconditional gen...")
    x0 = [0] * gen_len
    inputs = {
        "input_ids": torch.tensor([x0])
    }
    res = generate_samples(model, args, tokenizer, inputs, verbose=args.verbose)
    pred = tokenizer.decode(res.tolist()[0])
    print(pred)

    # conditional generation
    print("="*20, "Prefix gen...")
    prefix = [tokenizer.bos_token_id] + tokenizer.encode("Today is a wonderful day,")

    src_mask = [1]*len(prefix)+[0]*(gen_len-len(prefix))
    x0 = prefix + [0]*(gen_len-len(prefix))

    inputs = {
        "input_ids": torch.tensor([x0]), 
        "src_mask": torch.tensor([src_mask])
    }
    res = generate_samples(model, args, tokenizer, inputs, verbose=args.verbose)
    pred = tokenizer.decode(res.tolist()[0])
    print(pred)