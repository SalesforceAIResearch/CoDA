"""
Data collator for SFT (Supervised Fine-Tuning) that handles instruction-response pairs.
"""
import os
import torch
import logging

from typing import Dict, List, Optional, Union

import torch_xla.runtime as xr
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from datasets import Dataset, load_dataset, concatenate_datasets

def is_main_process():
    """Check if this is the main process (rank 0)."""
    return xr.process_index() == 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SFT_DATASET_MAP = {
    "opencoder_filtered_infinity_instruct": {
        "hf_hub_url": "OpenCoder-LLM/opc-sft-stage1",
        "subset": "filtered_infinity_instruct",
        "columns": {"prompt": "instruction", "response": "output"},
    },
    "opencoder_realuser_instruct": {
        "hf_hub_url": "OpenCoder-LLM/opc-sft-stage1",
        "subset": "realuser_instruct",
        "columns": {"prompt": "instruction", "response": "output"},
    },
    "opencoder_largescale_diverse_instruct": {
        "hf_hub_url": "OpenCoder-LLM/opc-sft-stage1",
        "subset": "largescale_diverse_instruct",
        "columns": {"prompt": "instruction", "response": "output"},
    },
    "opencoder_educational_instruct": {
        "hf_hub_url": "OpenCoder-LLM/opc-sft-stage2",
        "subset": "educational_instruct",
        "columns": {"prompt": "instruction", "response": "output"},
    },
    "opencoder_evol_instruct": {
        "hf_hub_url": "OpenCoder-LLM/opc-sft-stage2",
        "subset": "evol_instruct",
        "columns": {"prompt": "instruction", "response": "output"},
    },
    "opencoder_package_instruct": {
        "hf_hub_url": "OpenCoder-LLM/opc-sft-stage2",
        "subset": "package_instruct",
        "columns": {"prompt": "instruction", "response": "output"},
    },
    "opencoder_mceval_instruct": {
        "hf_hub_url": "OpenCoder-LLM/opc-sft-stage2",
        "subset": "mceval_instruct",
        "columns": {"prompt": "instruction", "response": "output"},
    },
}


class SFTDataCollator(DataCollatorMixin):
    """
    Data collator for SFT training that creates instruction-response pairs
    and generates src_masks to indicate which tokens are instruction vs response.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        format: str = "alpaca",
        include_system_prompt: bool = True,
        instruction_response_separator: str = "\n\n### Response:\n",
        custom_format: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.format = format
        self.include_system_prompt = include_system_prompt
        self.instruction_response_separator = instruction_response_separator
        self.custom_format = custom_format or {}

        # Get separator token IDs
        self.separator_token_ids = self.tokenizer.encode(
            instruction_response_separator, add_special_tokens=False
        )

    def _extract_instruction_response(self, example: Dict) -> tuple[str, str]:
        """Extract instruction and response from example based on format."""
        if self.format == "alpaca":
            instruction = example.get("instruction", "")
            response = example.get("output", "")

            # Include system prompt if available and enabled
            if self.include_system_prompt and "system" in example:
                system = example["system"]
                if system and system.strip():
                    instruction = f"{system}\n\n{instruction}"

        elif self.format == "sharegpt":
            # ShareGPT format typically has conversations
            conversations = example.get("conversations", [])
            if len(conversations) >= 2:
                # Take the first human message as instruction, first assistant as response
                instruction = conversations[0].get("value", "")
                response = conversations[1].get("value", "")
            else:
                instruction = ""
                response = ""

        elif self.format == "custom":
            instruction = example.get(
                self.custom_format.get("instruction_field", "instruction"), ""
            )
            response = example.get(
                self.custom_format.get("response_field", "response"), ""
            )

            # Include system prompt if available and enabled
            if self.include_system_prompt and "system_field" in self.custom_format:
                system = example.get(self.custom_format["system_field"], "")
                if system and system.strip():
                    instruction = f"{system}\n\n{instruction}"
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        return instruction, response

    def _create_instruction_response_sequence(
        self, instruction: str, response: str
    ) -> tuple[List[int], int]:
        """
        Create a tokenized sequence with instruction and response,
        and return the length of the instruction part.
        """
        # Tokenize instruction
        instruction_tokens = self.tokenizer.encode(
            instruction, add_special_tokens=False
        )

        # Add separator
        full_instruction_tokens = instruction_tokens + self.separator_token_ids

        # Tokenize response
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

        # Combine instruction + separator + response
        full_sequence = full_instruction_tokens + response_tokens

        # Return sequence and instruction length (including separator)
        return full_sequence, len(full_instruction_tokens)

    def __call__(
        self, features: List[Dict[str, Union[str, List[int]]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate features into a batch for SFT training.

        Args:
            features: List of examples, each containing instruction-response data

        Returns:
            Dictionary with:
            - input_ids: Tokenized sequences
            - attention_mask: Attention masks
            - instruction_lengths: Length of instruction part for each sequence
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_instruction_lengths = []

        for feature in features:
            # Extract instruction and response
            instruction, response = self._extract_instruction_response(feature)

            # Create sequence and get instruction length
            sequence, instruction_length = self._create_instruction_response_sequence(
                instruction, response
            )

            batch_input_ids.append(sequence)
            batch_instruction_lengths.append(instruction_length)

            # Create attention mask (all tokens are attended to)
            attention_mask = [1] * len(sequence)
            batch_attention_mask.append(attention_mask)

        # Pad sequences to the same length
        max_length = max(len(seq) for seq in batch_input_ids)

        padded_input_ids = []
        padded_attention_mask = []

        for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):
            # Pad with tokenizer's pad token
            padding_length = max_length - len(input_ids)
            padded_input_ids.append(
                input_ids + [self.tokenizer.pad_token_id] * padding_length
            )
            padded_attention_mask.append(attention_mask + [0] * padding_length)

            # Ensure we have at least one valid token in each sequence
            if len(input_ids) == 0:
                padded_input_ids[-1][0] = self.tokenizer.eos_token_id or 1
                padded_attention_mask[-1][0] = 1

        # Create src_mask for SFT training
        src_mask = torch.zeros(len(padded_input_ids), max_length, dtype=torch.bool)
        for i, length in enumerate(batch_instruction_lengths):
            # Ensure length is valid and within bounds
            valid_length = min(length, max_length)
            if valid_length > 0:
                src_mask[i, :valid_length] = True

        # Ensure we don't have completely empty sequences
        # If any sequence has zero instruction length, give it at least 1 token
        for i, length in enumerate(batch_instruction_lengths):
            if length == 0:
                src_mask[i, 0] = True
                batch_instruction_lengths[i] = 1

        result = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "instruction_lengths": torch.tensor(
                batch_instruction_lengths, dtype=torch.long
            ),
            "src_mask": src_mask,
        }

        return result

def pad_sequence(lists, padding_value, cut_len):
    new_lists = []
    for l in lists:
        if len(l) >= cut_len:
            lnew = l[:cut_len]
        else:
            lnew = l+[padding_value]*(cut_len-len(l))

        new_lists.append(lnew)
        
    return new_lists


def create_sft_dataset(
    name: str,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int = 8192,
) -> Dataset:
    """
    Create an SFT dataset from a raw dataset that:
    1. Assembles conversations
    2. Applies chat template (if available and enabled)
    3. Tokenizes the text
    4. Creates src_mask where context tokens = 1, response tokens = 0
    5. Filters out sequences exceeding block_size
    6. Pads remaining sequences to block_size

    Args:
        dataset: Raw dataset with instruction-response pairs
        tokenizer: Tokenizer for processing text
        format: Format of the data ("alpaca", "sharegpt", "custom")
        include_system_prompt: Whether to include system prompts
        instruction_response_separator: Separator between instruction and response
        custom_format: Custom format configuration
        block_size: Maximum sequence length

    Returns:
        Processed dataset ready for SFT training with padded sequences
    """
    if name not in SFT_DATASET_MAP:
        raise ValueError(
            f"Dataset {name} not found in SFT_DATASET_MAP. Available datasets: {list(SFT_DATASET_MAP.keys())}"
        )

    dataset_info = SFT_DATASET_MAP[name]
    dataset = load_dataset(
        dataset_info["hf_hub_url"],
        dataset_info["subset"],
        split="train",
        num_proc=min(64, os.cpu_count() - 4),
    )

    # Map columns to standard names if needed
    prompt_column = dataset_info["columns"]["prompt"]
    response_column = dataset_info["columns"]["response"]

    def process_example(
        examples: Dict[str, List],
        prompt_column: str = "prompt",
        response_column: str = "response",
    ):
        model_inputs = {"input_ids": [], "src_mask": []}

        for prompt, response in zip(examples[prompt_column], examples[response_column]):
            # Create conversation format for chat template
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            # Apply chat template
            input_sequence = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, return_dict=False
            )
            input_ids = tokenizer.encode(input_sequence, add_special_tokens=True)

            if len(input_ids) > block_size:
                continue

            assistant_tokens = [151644, 77091]
            # Search backwards for the pattern
            src_mask = None
            for i in range(len(input_ids) - len(assistant_tokens), -1, -1):
                if input_ids[i : i + len(assistant_tokens)] == assistant_tokens:
                    # Set mask to 1 up to and including the assistant tokens, 0 afterwards
                    src_mask = [1] * (i + len(assistant_tokens)) + [0] * (
                        len(input_ids) - i - len(assistant_tokens)
                    )
                    break
            if src_mask is None:
                raise ValueError(
                    "Ids of `<|im_start|>assistant` not found in input_ids, please check the preprocessing."
                )

            model_inputs["input_ids"].append(input_ids)
            model_inputs["src_mask"].append(src_mask)
        
        model_inputs["input_ids"] = pad_sequence(model_inputs["input_ids"], tokenizer.pad_token_id, block_size)
        model_inputs["src_mask"] = pad_sequence(model_inputs["src_mask"], 0, block_size)

        return model_inputs

    # Process the dataset
    processed_dataset = dataset.map(
        process_example,
        fn_kwargs={"prompt_column": prompt_column, "response_column": response_column},
        remove_columns=dataset.column_names,
        desc="Processing SFT dataset",
        batched=True,
        batch_size=1000,
        num_proc=min(64, os.cpu_count() - 4),
    )

    return processed_dataset


def concat_and_shuffle_sft_datasets(
    datasets: List[Dataset],
    seed: Optional[int] = None,
) -> Dataset:
    """
    Concatenate multiple SFT datasets and shuffle them.

    Args:
        datasets: List of processed SFT datasets to concatenate
        seed: Random seed for shuffling (for reproducibility)

    Returns:
        Concatenated and shuffled dataset
    """
    if not datasets:
        raise ValueError("At least one dataset must be provided")

    if len(datasets) == 1:
        # If only one dataset, just shuffle it
        if seed is not None:
            return datasets[0].shuffle(seed=seed)
        else:
            return datasets[0].shuffle()

    # Concatenate all datasets
    concatenated_dataset = concatenate_datasets(datasets)

    # Shuffle the concatenated dataset
    if seed is not None:
        shuffled_dataset = concatenated_dataset.shuffle(seed=seed)
    else:
        shuffled_dataset = concatenated_dataset.shuffle()

    return shuffled_dataset


def make_sft_dataset(
    dataset_names: List[str] | str,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int = 8192,
    seed: Optional[int] = None,
) -> Dataset:
    """
    Create a combined SFT dataset from multiple dataset configurations.

    Args:
        dataset_configs: List of dataset configurations, each containing:
            - name: Dataset name from SFT_DATASET_MAP
            - config_name: Configuration name (optional)
            - max_samples: Maximum number of samples to take from this dataset (optional)
        tokenizer: Tokenizer for processing text
        cache_dir: Directory to cache datasets
        block_size: Maximum sequence length
        data_format: Format of the data ("alpaca", "sharegpt", "custom")
        include_system_prompt: Whether to include system prompts
        instruction_response_separator: Separator between instruction and response
        custom_format: Custom format configuration
        use_chat_template: Whether to use tokenizer's chat template if available
        shuffle_seed: Random seed for shuffling (for reproducibility)

    Returns:
        Combined, processed, and shuffled SFT dataset

    Example:
        dataset_configs = [
            {"name": "opencoder_filtered_infinity_instruct", "max_samples": 10000},
            {"name": "opencoder_realuser_instruct", "max_samples": 5000},
        ]
        combined_dataset = create_multi_sft_dataset(
            dataset_configs=dataset_configs,
            tokenizer=tokenizer,
            cache_dir="./cache",
            block_size=8192
        )
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    processed_datasets = []

    for name in dataset_names:
        if is_main_process():
            logger.info(f"Processing dataset: {name}")
        # Process the dataset
        processed_dataset = create_sft_dataset(
            name=name,
            tokenizer=tokenizer,
            block_size=block_size,
        )

        processed_datasets.append(processed_dataset)

    # Concatenate and shuffle all datasets
    final_dataset = concat_and_shuffle_sft_datasets(
        datasets=processed_datasets, seed=seed
    )

    if is_main_process():
        logger.info(f"Final combined dataset: {len(final_dataset)} samples")
    return final_dataset

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from datasets import load_from_disk, DatasetDict
    # dataset_names = ["opencoder_filtered_infinity_instruct", "opencoder_realuser_instruct", "opencoder_largescale_diverse_instruct"]
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    # block_size = 1024
    # seed = 42
    # dataset = make_sft_dataset(dataset_names, tokenizer, block_size, seed)
    # dataset.save_to_disk("/export/agentstudio-family-2/haolin/data/sft_opc_stage1", max_shard_size="2GB", num_proc=64)
    dataset = load_from_disk("/export/agentstudio-family-2/haolin/data/sft_opc_stage1")
