from transformers import AutoConfig, AutoModelForCausalLM, AutoModel
import torch

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError: # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import torch.distributed as dist
from transformers import PreTrainedModel

class DiscreteDiffusionModel(PreTrainedModel):
    """
    diffusion model
    """
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def __init__(
        self,
        model,
        config,
        model_args
    ):
        super().__init__(config)
        self.model = model
        self.config = config
        self.embed_dim = self.config.hidden_size
        self.hidden_dim = self.config.hidden_size
        # Get vocab size from model or config
        if hasattr(model, 'get_input_embeddings'):
            self.vocab_size = model.get_input_embeddings().weight.size(0)
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            self.vocab_size = model.model.embed_tokens.weight.size(0)
        elif getattr(self.config, "model_type", None) == "qwen3":
            self.vocab_size = getattr(self.config, 'vocab_size', 151936)
        else:
            raise ValueError(f"Unsupported model type: {getattr(self.config, 'model_type', None)}")
            
        if getattr(self.config, "model_type", None) == "gpt2":
            self.embed_tokens = self.model.transformer.wte
            self.denoise_model = self.model.transformer # use inputs_embeds instead of input_ids in forward function
            for gpt2block in self.model.transformer.h:
                gpt2block.attn.bias.fill_(True)  # remove causal mask
            self.lm_head = self.model.lm_head
            del self.denoise_model.wte
        elif getattr(self.config, "model_type", None) == "llama":
            self.embed_tokens = self.model.model.embed_tokens
            self.denoise_model = self.model.model
            self.lm_head = self.model.lm_head
            del self.denoise_model.embed_tokens
        elif getattr(self.config, "model_type", None) == "qwen3" or hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            # Handle torchprime Qwen3 models
            self.embed_tokens = self.model.model.embed_tokens
            self.denoise_model = self.model.model
            self.lm_head = self.model.lm_head
            del self.denoise_model.embed_tokens
        del self.model

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def get_embeds(self, input_ids):
        return self.embed_tokens(input_ids)
    
    def forward(self, input_ids, attention_mask, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        denoise the input
        """
        x_embed = self.get_embeds(input_ids)

        x = self.denoise_model(inputs_embeds = x_embed, attention_mask=attention_mask, return_dict = False)[0]

        logits = self.get_logits(x)

        return logits
