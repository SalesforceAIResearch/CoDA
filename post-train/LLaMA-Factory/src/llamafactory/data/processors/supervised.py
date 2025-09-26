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

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .processor_utils import greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


logger = get_logger(__name__)


def _encode_supervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["Image"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    train_on_prompt: bool,
    mask_history: bool,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    messages = template.mm_plugin.process_messages(prompt + response, images, processor)
    input_ids, labels = template.mm_plugin.process_token_ids([], [], images, tokenizer, processor)
    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    total_length = len(input_ids) + (1 if template.efficient_eos else 0)
    if mask_history:
        encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= cutoff_len:
            break

        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len

        if train_on_prompt:
            source_label = source_ids
        elif template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_history and turn_idx != 0:  # train on the last turn only
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        if mask_history:  # reversed sequences
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    extra_inputs = template.mm_plugin.get_mm_inputs(
        images=images, feature_seqlens={"token_type_ids": len(input_ids)}, processor=processor
    )
    return input_ids, labels, extra_inputs


def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = defaultdict(list)
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        input_ids, labels, extra_inputs = _encode_supervised_example(
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            images=examples["images"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        for key, value in extra_inputs.items():
            model_inputs[key].append(value)

    return model_inputs


def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # TODO: use `position_ids` to achieve packing
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    if processor is not None:
        raise NotImplementedError("`packing` have not been implemented for multimodal datasets.")

    valid_num = 0
    batch_input_ids, batch_labels = [], []
    lengths = []
    length2indexes = defaultdict(list)
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            images=examples["images"][i],
            template=template,
            tokenizer=tokenizer,
            processor=None,
            cutoff_len=data_args.cutoff_len - 1,  # reserved for the padding token
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        length = len(input_ids)
        if length > data_args.cutoff_len:
            logger.warning("Dropped lengthy example with length {} > {}.".format(length, data_args.cutoff_len))
        else:
            lengths.append(length)
            length2indexes[length].append(valid_num)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            valid_num += 1

    model_inputs = defaultdict(list)
    knapsacks = greedy_knapsack(lengths, data_args.cutoff_len - 1)  # reserved for the padding token
    for knapsack in knapsacks:
        packed_input_ids, packed_attention_masks, packed_labels = [], [], []
        for i, length in enumerate(knapsack):
            index = length2indexes[length].pop()
            packed_input_ids += batch_input_ids[index]
            packed_labels += batch_labels[index]
            if data_args.neat_packing:
                packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
            else:
                packed_attention_masks += [1] * len(batch_input_ids[index])

        if len(packed_input_ids) < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - len(packed_input_ids)
            packed_input_ids += [tokenizer.pad_token_id] * pad_length
            packed_labels += [IGNORE_INDEX] * pad_length
            if data_args.neat_packing:
                packed_attention_masks += [0] * pad_length
            else:
                packed_attention_masks += [1] * pad_length  # more efficient flash_attn

        if len(packed_input_ids) != data_args.cutoff_len:
            raise ValueError("The length of packed example should be identical to the cutoff length.")

        model_inputs["input_ids"].append(packed_input_ids)
        model_inputs["attention_mask"].append(packed_attention_masks)
        model_inputs["labels"].append(packed_labels)

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print("labels:\n{}".format(tokenizer.decode(valid_labels, skip_special_tokens=False)))

def preprocess_partical_dataset(examples: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer", data_args: "DataArguments"
    ) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and masks with format `<ignore> ... <ignore> Y <eos>`
    # single turn
    model_inputs = {"input_ids": [], "attention_mask": [], "src_mask": []}

    # sep_token_id = 9166 # ====== for llama
    # sep_token_id = 1647 # ====== for mistral
    # pad_token_id = 258 # <0xFF> for llama
    # sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    assert pad_token_id is not None, "pad_token_id is None"
    # assert sep_token_id is not None, "sep_token_id is None"

    examples_prompt = [messages[0]["content"] for messages in examples["prompt"]]
    examples_output = [messages[0]["content"] for messages in examples["response"]]

    for src, tgt in zip(examples_prompt, examples_output):
        src_ids = tokenizer.encode(src, add_special_tokens=False)
        tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)

        # First apply chat template on original conversation to see total length
        conversation = [
            {"role": "user", "content": src},
            {"role": "assistant", "content": tgt},
        ]
        input_ids = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        # If exceeds cutoff, intelligently truncate/just discard the example
        if data_args.cutoff_len is not None and len(input_ids) > data_args.cutoff_len:
            print(f"Discarding example due to cutoff length, length: {len(input_ids)}, cutoff length: {data_args.cutoff_len}")
            continue
            # excess_tokens = len(input_ids) - data_args.cutoff_len

            # # First compute how many target tokens we can preserve
            # # We can remove at most len(src_ids) from src, rest must come from target
            # tokens_to_remove_from_target = max(0, excess_tokens - len(src_ids))
            # target_tokens_to_keep = max(
            #     1, len(tgt_ids) - tokens_to_remove_from_target
            # )  # Keep at least 1 target token

            # # Keep target as much as possible
            # tgt_ids_cut = tgt_ids[:target_tokens_to_keep]

            # # Remove remaining excess from src (keep end of src)
            # tokens_to_remove_from_src = excess_tokens - (
            #     len(tgt_ids) - target_tokens_to_keep
            # )
            # src_tokens_to_keep = max(
            #     1, len(src_ids) - tokens_to_remove_from_src
            # )  # Keep at least 1 src token
            # src_ids_cut = src_ids[-src_tokens_to_keep:]

            # if tokens_to_remove_from_target > 0:
            #     print(
            #         f"Truncating target from {len(tgt_ids)} tokens to {target_tokens_to_keep} tokens due to cutoff length, please check if the input is too long or cutoff length is too short"
            #     )
            # if tokens_to_remove_from_src > 0:
            #     print(
            #         f"Truncating source from {len(src_ids)} tokens to {src_tokens_to_keep} tokens due to cutoff length, please check if the input is too long or cutoff length is too short"
            #     )

            # # Rebuild conversation with truncated content
            # src_cutoff, tgt_cutoff = tokenizer.decode(src_ids_cut), tokenizer.decode(
            #     tgt_ids_cut
            # )
            # conversation = [
            #     {"role": "user", "content": src_cutoff},
            #     {"role": "assistant", "content": tgt_cutoff},
            # ]
            # input_ids = tokenizer.apply_chat_template(
            #     conversation,
            #     tokenize=True,
            #     return_dict=False,
            #     add_generation_prompt=False,
            #     enable_thinking=False,
            # )

            # # Final check: if still exceeds cutoff, truncate from the end
            # if len(input_ids) > data_args.cutoff_len:
            #     input_ids = input_ids[:data_args.cutoff_len]
            #     print(f"Final input_ids length {len(input_ids)} after chat template still exceeded cutoff, truncated to {data_args.cutoff_len}")

        # NOTE: Here the src_mask should be 1 for all tokens before the last occurrence of '<|im_start|>assistant'(generation prompt) for Qwen3 models, and 0 for the rest
        # NOTE 2: The token ids of '<|im_start|>assistant' is [151644, 77091] for Qwen3 models
        # NOTE 3: Currently the dataset doesn't contain any reasoning tokens, if that is the case, we should wrap the reasoning tokens in <think></think> tags in the chat template

        # Find the last occurrence of [151644, 77091] in input_ids using reverse search
        assistant_tokens = [151644, 77091]
        # Search backwards for the pattern
        src_mask = None
        for i in range(len(input_ids) - len(assistant_tokens), -1, -1):
            if input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
                # Set mask to 1 up to and including the assistant tokens, 0 afterwards
                src_mask = [1] * (i + len(assistant_tokens)) + [0] * (len(input_ids) - i - len(assistant_tokens))
                break
        if src_mask is None:
            raise ValueError("Ids of `<|im_start|>assistant` not found in input_ids, please check the preprocessing.")

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["src_mask"].append(src_mask)

    model_inputs["input_ids"] = pad_sequence(model_inputs["input_ids"], padding_value=pad_token_id, cut_len=data_args.cutoff_len)
    model_inputs["labels"] = model_inputs["input_ids"]
    model_inputs["attention_mask"] = pad_sequence(model_inputs["attention_mask"], padding_value=1, cut_len=data_args.cutoff_len)
    model_inputs["src_mask"] = pad_sequence(model_inputs["src_mask"], padding_value=0, cut_len=data_args.cutoff_len)

    return model_inputs

def pad_sequence(lists, padding_value, cut_len):
    new_lists = []
    for l in lists:
        if len(l) >= cut_len:
            lnew = l[:cut_len]
        else:
            lnew = l+[padding_value]*(cut_len-len(l))

        new_lists.append(lnew)
        
    return new_lists
