# LLaDA Model Evaluation Guide

This document provides detailed instructions for evaluating the LLaDA model on GSM8K math problem solving and HumanEval code generation tasks.

## Environment Setup

Before running any evaluation, set the following environment variables:
```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```

## GSM8K Evaluation

GSM8K is a dataset of 8,000 grade school math problems designed to evaluate mathematical reasoning capabilities.

### Common Parameters

```bash
task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
```

### Evaluation Methods

1. **Baseline**
```bash
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},show_speed=True
```

2. **Prefix Cache**
```bash
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True
```

3. **Parallel Generation**
```bash
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.9,show_speed=True
```

4. **Prefix Cache + Parallel**
```bash
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True
```

5. **Dual Cache + Parallel**
```bash
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True
```

### Parameter Descriptions

- `task`: Evaluation task (gsm8k)
- `length`: Generation length
- `block_length`: Block size for parallel generation
- `num_fewshot`: Number of few-shot examples
- `steps`: Number of generation steps
- `use_cache`: Enable prefix cache
- `dual_cache`: Enable dual cache
- `threshold`: Confidence threshold for parallel generation
- `show_speed`: Display speed metrics

## HumanEval Evaluation

HumanEval is a dataset of 164 Python programming problems designed to evaluate code generation capabilities.

### Common Parameters

```bash
task=humaneval
length=256
block_length=32
steps=$((length / block_length))
```

### Evaluation Methods

1. **Baseline**
```bash
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},show_speed=True \
--output_path evals_results/baseline/humaneval-ns0-${length} --log_samples
```

2. **Prefix Cache**
```bash
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True \
--output_path evals_results/prefix_cache/humaneval-ns0-${length} --log_samples
```

3. **Parallel Generation**
```bash
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.9,show_speed=True \
--output_path evals_results/parallel/humaneval-ns0-${length} --log_samples
```

4. **Prefix Cache + Parallel**
```bash
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True \
--output_path evals_results/cache_parallel/humaneval-ns0-${length} --log_samples
```

5. **Dual Cache + Parallel**
```bash
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True \
--output_path evals_results/dual_cache_parallel/humaneval-ns0-${length} --log_samples
```

### Post-processing

For HumanEval evaluation, post-processing is required:
```bash
python postprocess_code.py {the samples_xxx.jsonl file under output_path}
```

## Notes

1. All evaluations use the LLaDA-8B-Instruct model
2. Results are saved in the `evals_results` directory
3. For HumanEval, samples are logged for post-processing
4. Speed metrics are shown for all evaluations
5. Different optimization strategies can be combined: