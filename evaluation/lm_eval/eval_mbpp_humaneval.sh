#!/bin/bash
MODEL_DIR=${1}

HF_ALLOW_CODE_EVAL=1 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=$MODEL_DIR,trust_remote_code=True,max_new_tokens=768,diffusion_steps=768,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks humaneval_instruct \
    --device cuda \
    --batch_size 32 \
    --num_fewshot 0 \
    --output_path output_reproduce/humaneval \
    --log_samples \
    --confirm_run_unsafe_code \
    --apply_chat_template true

HF_ALLOW_CODE_EVAL=1 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=$MODEL_DIR,trust_remote_code=True,max_new_tokens=768,diffusion_steps=768,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks humaneval_plus \
    --device cuda \
    --batch_size 32 \
    --num_fewshot 0 \
    --output_path output_reproduce/humaneval_plus \
    --log_samples \
    --confirm_run_unsafe_code \
    --apply_chat_template true

HF_ALLOW_CODE_EVAL=1 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=$MODEL_DIR,trust_remote_code=True,max_new_tokens=768,diffusion_steps=768,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks mbpp_instruct \
    --device cuda \
    --batch_size 32 \
    --num_fewshot 0 \
    --output_path output_reproduce/mbpp \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template true

HF_ALLOW_CODE_EVAL=1 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=$MODEL_DIR,trust_remote_code=True,max_new_tokens=768,diffusion_steps=768,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks mbpp_plus_instruct \
    --device cuda \
    --batch_size 32 \
    --num_fewshot 0 \
    --output_path output_reproduce/mbpp_plus \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template true
