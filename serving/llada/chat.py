# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import argparse

from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

def chat(args):
    device = 'cuda'
    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    gen_length = args.gen_length
    steps = args.steps
    print('*' * 66)
    print(f'**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **')
    print('*' * 66)

    conversation_num = 0
    while True:
        user_input = input("Enter your question: ")

        m = [{"role": "user", "content": user_input}]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        if conversation_num == 0:
            prompt = input_ids
        else:
            prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)
        print(f'use cache: {args.use_cache} use cache position: {args.if_cache_position} threshold: {args.threshold} block size: {args.block_size}')
        if args.use_cache:
            if args.if_cache_position:
                out, nfe = generate_with_dual_cache(model, prompt, steps=steps, gen_length=gen_length, block_length=args.block_size, temperature=0., remasking='low_confidence', threshold=args.threshold)
            else:
                out, nfe = generate_with_prefix_cache(model, prompt, steps=steps, gen_length=gen_length, block_length=args.block_size, temperature=0., remasking='low_confidence', threshold=args.threshold)
        else:
            out, nfe = generate(model, prompt, steps=steps, gen_length=gen_length, block_length=args.block_size, temperature=0., remasking='low_confidence', threshold=args.threshold)

        answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
        print(f"Bot's reply: {answer}")
        print(f"Number of forward passes: {nfe}")

        # remove the <EOS>
        prompt = out[out != 126081].unsqueeze(0)
        conversation_num += 1
        print('-----------------------------------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--if_cache_position", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)

    args = parser.parse_args()
    chat(args)

