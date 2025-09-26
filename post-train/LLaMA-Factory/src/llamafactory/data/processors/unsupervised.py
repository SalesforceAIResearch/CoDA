# Copyright 2024 the LlamaFactory team.
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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras.logging import get_logger
from ..data_utils import Role
from .processor_utils import infer_seqlen


if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


logger = get_logger(__name__)


def _encode_unsupervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["Image"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    if len(response) == 1:
        messages = prompt + response
    else:
        messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]

    messages = template.mm_plugin.process_messages(messages, images, processor)
    input_ids, labels = template.encode_oneturn(tokenizer, messages, system, tools)
    
    # Debug: Check where None values are coming from
    if input_ids is None or any(id is None for id in input_ids):
        logger.warning("Invalid input_ids detected!")
        logger.warning(f"Messages: {messages}")
        logger.warning(f"System: {system}")
        logger.warning(f"Tools: {tools}")
        logger.warning(f"Raw input_ids: {input_ids}")
        logger.warning(f"Raw labels: {labels}")
        
        # Try to fix the None values immediately
        if input_ids:
            input_ids = [token_id for token_id in input_ids if token_id is not None]
        else:
            input_ids = []
            
    if labels is None or any(id is None for id in labels):
        logger.warning("Invalid labels detected, using empty list")
        if labels:
            labels = [token_id for token_id in labels if token_id is not None]
        else:
            labels = []
    
    if template.efficient_eos:
        labels += [tokenizer.eos_token_id]

    # Aggressive fix for None tokens - remove them before any other processing
    if input_ids:
        input_ids = [token_id for token_id in input_ids if token_id is not None]
    if labels:
        labels = [token_id for token_id in labels if token_id is not None]

    input_ids, _ = template.mm_plugin.process_token_ids(input_ids, None, images, tokenizer, processor)
    
    # # HACK: Remove first None token and append padding token at the end
    # if input_ids and input_ids[0] is None:
    #     input_ids = input_ids[1:]
    #     labels = labels[1:]
    #     input_ids.append(tokenizer.pad_token_id)
    #     labels.append(tokenizer.pad_token_id)  # Also append pad token to labels
    #     logger.debug("Removed first None token from input_ids and appended pad tokens")
    
    # # Remove any remaining None values
    # input_ids = [token_id for token_id in input_ids if token_id is not None]
    # labels = [token_id for token_id in labels if token_id is not None]
    
    # Append padding token at the end
    # if input_ids:
    #     if tokenizer.pad_token_id is not None:
    #         input_ids.append(tokenizer.pad_token_id)
    #     elif tokenizer.eos_token_id is not None:
    #         input_ids.append(tokenizer.eos_token_id)  # Use EOS as pad
    #     else:
    #         input_ids.append(0)  # Fallback default
    #     logger.debug("Appended padding token at the end")
    
    # Hack for TorchPrime models: remove first None token and add pad token
    # Check if this is a TorchPrime model by looking for Qwen3 tokenizer characteristics
    # is_torchprime = (hasattr(tokenizer, 'model_max_length') and 
    #                 hasattr(tokenizer, 'pad_token_id') and 
    #                 tokenizer.pad_token_id is None and
    #                 'qwen' in tokenizer.name_or_path.lower())

    # HACK: remove first None token and add pad token at the end
    # is_torchprime = True
    # if is_torchprime:
    #     print("is_torchprime: ", is_torchprime)
    #     # Hack for TorchPrime models: remove first None token and add pad token
    #     if input_ids and input_ids[0] is None:
    #         input_ids = input_ids[1:]
    #         labels = labels[1:]
    #         input_ids.append(tokenizer.pad_token_id)
    #         labels.append(tokenizer.pad_token_id)  # Also append pad token to labels
    #         logger.debug("Removed None token from beginning for TorchPrime model")
        
    #     # Remove any remaining None tokens
    #     # input_ids = [token_id for token_id in input_ids if token_id is not None]
    #     # labels = [token_id for token_id in labels if token_id is not None]
        
    #     # Add pad token at the end for TorchPrime models
    #     # if input_ids and tokenizer.eos_token_id is not None:
    #     #     input_ids.append(tokenizer.eos_token_id)  # Use EOS as pad for Qwen models
    # else:
    #     print("is_torchprime: False")
    
    source_len, target_len = infer_seqlen(len(input_ids), len(labels), cutoff_len)
    input_ids = input_ids[:source_len]
    labels = labels[:target_len]
    extra_inputs = template.mm_plugin.get_mm_inputs(
        images=images, feature_seqlens={"token_type_ids": len(input_ids)}, processor=processor
    )
    return input_ids, labels, extra_inputs


def preprocess_unsupervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X` and labels with format `Y <eos>`
    model_inputs = defaultdict(list)
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        input_ids, labels, extra_inputs = _encode_unsupervised_example(
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            images=examples["images"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
        )
        
        # Skip examples with empty or invalid input_ids
        if not input_ids or len(input_ids) == 0:
            logger.warning(f"Skipping example {i} with empty input_ids")
            continue
            
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        for key, value in extra_inputs.items():
            model_inputs[key].append(value)

    return model_inputs


def print_unsupervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    input_ids = example["input_ids"]
    if input_ids is None or any(id is None for id in input_ids):
        print("Warning: input_ids contains None values, attempting to fix...")
        print("Original input_ids:\n{}".format(input_ids))
        
        # Try to fix by removing None values
        if input_ids:
            input_ids = [token_id for token_id in input_ids if token_id is not None]
            print("Fixed input_ids:\n{}".format(input_ids))
            
            if input_ids:
                try:
                    print("inputs:\n{}".format(tokenizer.decode(input_ids, skip_special_tokens=False)))
                except Exception as e:
                    print(f"Failed to decode even after fixing: {e}")
            else:
                print("No valid tokens after fixing")
        else:
            print("input_ids is None, cannot fix")
        return
    
    print("input_ids:\n{}".format(input_ids))
    print("inputs:\n{}".format(tokenizer.decode(input_ids, skip_special_tokens=False)))
