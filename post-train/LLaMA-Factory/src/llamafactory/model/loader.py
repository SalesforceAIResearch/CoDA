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

from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from ..extras.logging import get_logger
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_ms
from .adapter import init_adapter
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .model_utils.visual import get_image_seqlen
from .patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model
from safetensors.torch import load_file

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)

from transformers import GPT2TokenizerFast, GPT2Tokenizer
from typing import Dict
from itertools import chain
from tokenizers.pre_tokenizers import Digits


class MaskTokenWrapper(GPT2TokenizerFast):
    def __init__(self, tokenizer):
        EOS_TOKEN = tokenizer.eos_token
        PAD_TOKEN = "¨"
        SEP_TOKEN = "======"
        MASK_TOKEN_ID = tokenizer.vocab_size
        self.tokenizer = tokenizer
        self.digit_tokenizer = Digits(individual_digits=True)
        self.token2id = {token:id for token, id in self.tokenizer.vocab.items()}
        self.tokenizer.add_special_tokens({
            'pad_token': PAD_TOKEN, 
            'eos_token': EOS_TOKEN, 
            'sep_token': SEP_TOKEN,
            'mask_token': "[¨M¨]"
        })
        self.__dict__.update(self.tokenizer.__dict__.items())
        self.eos_token_id = self.token2id[EOS_TOKEN]
        self.pad_token_id = self.token2id[PAD_TOKEN]
        self.sep_token_id = self.token2id[SEP_TOKEN]
        self.mask_token_id = MASK_TOKEN_ID

    def encode(self, text, digit=True, **kwargs):
        if digit:
            chunks = self.digit_tokenizer.pre_tokenize_str(text)
            res = self.encode_batch([i[0] for i in chunks], digit=False, **kwargs)
            return res
        return self.tokenizer(text)


    def encode_batch(self, texts, digit=True, **kwargs):
        if digit:
            return [self.encode(text, digit=True, **kwargs) for text in texts]
        return list(chain.from_iterable([self.tokenizer.encode(text, **kwargs) for text in texts]))


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_ms(model_args)
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")

    patch_tokenizer(tokenizer)
    if isinstance(tokenizer, GPT2TokenizerFast) or isinstance(tokenizer, GPT2Tokenizer):
        if "medium" in model_args.model_name_or_path:
            # import pdb; pdb.set_trace();
            tokenizer = MaskTokenWrapper(tokenizer)
            model_args.resize_vocab = True
            # logger.info("Current vocab size:", len(tokenizer))
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")
        else:
            tokenizer.mask_token_id = 10541 # for diffu-gpt2-small
            tokenizer.sep_token_id = 50155
            # EOS_TOKEN = 50256
            # PAD_TOKEN = 101
            # SEP_TOKEN_ID = 50155

    if tokenizer.mask_token_id is None: # for diffu-llama
        tokenizer.mask_token_id = 811
        tokenizer.pad_token_id = 30399
        tokenizer.sep_token_id = 4936
    try:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        setattr(processor, "tokenizer", tokenizer)
        setattr(processor, "image_seqlen", get_image_seqlen(config))
        setattr(processor, "image_resolution", model_args.image_resolution)
    except Exception:
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if "Processor" not in processor.__class__.__name__:
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    # Check if this is a torchprime model first
    if hasattr(model_args, 'torchprime_config') and model_args.torchprime_config:
        from .torchprime_loader import TorchPrimeModelLoader
        from ..train.ddm.model import DiscreteDiffusionModel
        
        # Load torchprime model directly
        loader = TorchPrimeModelLoader(
            model_config_path=model_args.torchprime_config,
            checkpoint_dir=model_args.model_name_or_path
        )
        model = loader.load_model()
        
        # # Update tokenizer if needed
        # if hasattr(model_args, 'update_tokenizer') and model_args.update_tokenizer:
        #     tokenizer = loader.load_tokenizer()
        tokenizer = loader.load_tokenizer()
        # Create a minimal config for DDM
        from transformers import PretrainedConfig
        config = PretrainedConfig()
        setattr(config, 'model_type', 'qwen3')
        setattr(config, 'hidden_size', 8192)  
        setattr(config, 'vocab_size', 151937) 
        
        # For DDM stage, wrap with DiscreteDiffusionModel
        if "ddm" in finetuning_args.stage:
            model = DiscreteDiffusionModel(model, config, model_args)
        
        # Initialize adapter if needed
        model = init_adapter(config, model, model_args, finetuning_args, is_trainable)
        
        return model
    
    # Regular HuggingFace model loading
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
                load_class = AutoModelForVision2Seq
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config)
            else:
                model = load_class.from_pretrained(**init_kwargs)

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    if "ddm" in finetuning_args.stage:
        from ..train.ddm.model import DiscreteDiffusionModel
        import os
        
        model = DiscreteDiffusionModel(model, config, model_args)
        # import pdb; pdb.set_trace();
        checkpoint_dir = model_args.checkpoint_dir
        loaded = None
        if checkpoint_dir is not None: # for sampling
            if os.path.exists(os.path.join(checkpoint_dir, 'model.safetensors')):
                loaded = load_file(
                    os.path.join(checkpoint_dir, 'model.safetensors')
                )
            elif os.path.exists(os.path.join(checkpoint_dir, 'pytorch_model.bin')):
                loaded = torch.load(
                    os.path.join(checkpoint_dir, 'pytorch_model.bin'),
                    map_location=torch.device('cuda')
                )
                loaded = {k: v for k, v in loaded.items() if not k.startswith('model.')}
            # printparam(model)
            # print('='*30)
            # print(loaded)
            if loaded:
                model.load_state_dict(loaded, strict=False)
            # model.to(torch.device('cuda'))
    # import pdb; pdb.set_trace();
    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)
    # import pdb; pdb.set_trace();

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:,}".format(all_param)

    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, shape: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.shape, param.device, param.requires_grad
                )
            )

    return model

def printparam(model):
    for name, param in model.named_parameters():
        print(
            "name: {}, dtype: {}, shape: {}, device: {}, trainable: {}".format(
                name, param.dtype, param.shape, param.device, param.requires_grad
            )
        )
