"""
Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
"""

from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils import logging
from .model_config import CoDAConfig
from .attention import AttentionModule
from .modeling_utils import (
    HomogeneousSequential,
    RopeScaling,
    default_rope_frequencies,
    apply_rotary_pos_emb,
    transition,
    prefix_input_ids,
    truncate_input_ids,
)
from .generation_utils import DLMGenerationMixin, DLMGenerationConfig



logger = logging.get_logger(__name__)


class CoDARMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class CoDAMLP(nn.Module):
    def __init__(self, config: CoDAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class CoDAAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: CoDAConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.attention_block = AttentionModule(config)
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        # weiran: diffullama
        self.is_causal = False

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=getattr(config, "attention_bias", False),
        )
        self.q_norm = CoDARMSNorm(
            self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self.k_norm = CoDARMSNorm(
            self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.FloatTensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Apply q_norm and k_norm to the head dimension
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        # Apply normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose to get the right shape for attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        attn_output = self.attention_block(
            query_states, key_states, value_states, attention_mask
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class CoDARotaryEmbedding(nn.Module):
    inv_freq: nn.Buffer

    def __init__(
        self,
        head_dim,
        rope_theta,
        scaling: RopeScaling | None = None,
    ):
        super().__init__()
        if scaling is None:
            inv_freq = default_rope_frequencies(head_dim, theta=rope_theta)
        else:
            raise NotImplementedError("Scaling is not implemented")
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class CoDADecoderLayer(nn.Module):
    def __init__(self, config: CoDAConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = CoDAAttention(config=config, layer_idx=layer_idx)

        self.mlp = CoDAMLP(config)
        self.input_layernorm = CoDARMSNorm(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self.post_attention_layernorm = CoDARMSNorm(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_embeddings: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None,  # necessary, but kept here for BC
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        """
        # This gives the `hidden_states` tensor a name so that we can layer specify
        # to offload this tensor to host RAM to save memory. This is not a standard
        # torch API because there is no such feature in PyTorch. Instead, the name
        # becomes node metadata during FX graph capture.

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CoDAModel(PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers.

    Args:
      config: FlexConfig
    """
    config_class = CoDAConfig

    def __init__(self, config: CoDAConfig):
        super().__init__(config=config)
        self.vocab_size = config.vocab_size
        if "pad_token_id" not in config:
            self.padding_idx = None
        else:
            self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        # `HomogeneousSequential` is similar to `nn.Sequential` but can be compiled with
        # `scan` described in https://pytorch.org/xla/release/r2.6/features/scan.html.
        self.layers = HomogeneousSequential(
            *[
                CoDADecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = CoDARMSNorm(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )

        rope_scaling = getattr(config, "rope_scaling", None)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        if rope_scaling is not None:
            rope_scaling = RopeScaling(**rope_scaling)
        self.rotary_emb = CoDARotaryEmbedding(
            head_dim=head_dim, rope_theta=self.rope_theta, scaling=rope_scaling
        )
        self.post_init()

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        # convert input ids to embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        seq_length = inputs_embeds.size(1)

        position_ids = (
            torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0).float()
        )

        # Create a causal attention mask
        causal_mask = torch.triu(
            torch.full(
                (seq_length, seq_length), float("-inf"), device=inputs_embeds.device
            ),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(
            0
        )  # Add batch and head dimension

        if attention_mask is not None:
            causal_mask = causal_mask * attention_mask[:, None, None, :]

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        hidden_states = self.layers(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class CoDALanguageModel(DLMGenerationMixin, PreTrainedModel):
    config_class = CoDAConfig
    base_model_prefix = "model"
    is_parallelizable = True
    supports_gradient_checkpointing = False
    _no_split_modules = ["FlexDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config: CoDAConfig):
        super().__init__(config)
        self.config = config
        self.model = CoDAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mask_token_id = config.mask_token_id
        self.generation_config = DLMGenerationConfig(mask_token_id=config.mask_token_id)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_embeds(self, input_ids):
        """
        Get input embeddings from the model.
        This method is used by the diffusion trainer to access embeddings.
        """
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        src_mask: torch.BoolTensor | None = None,
        training_mode: str = "pretrain",
        **kwargs,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
        if not self.training:
            model_output = self.model(
                input_ids=input_ids, attention_mask=None
            )
            hidden_states = model_output
            logits = self.lm_head(hidden_states)  # NOTE: we shift logits at inference time
            return logits, None

        if training_mode == "sft" and src_mask is None:
            raise ValueError("SFT mode requires a non-null src_mask")

        epoch = kwargs.get("epoch", None)
        sampling_eps = getattr(
            self.config, "sampling_eps", 1e-3
        )  # NOTE: use sampling_eps to control the noise level
        # If sampling_eps is a list, choose based on epoch
        if isinstance(sampling_eps, list):
            if epoch is None:
                # If epoch is not provided, use the first value
                sampling_eps = sampling_eps[0]
            else:
                # Use modulo to cycle through the list if epoch exceeds list length
                sampling_eps = sampling_eps[epoch % len(sampling_eps)]

        mask_token_id = self.mask_token_id
        loss_func = nn.CrossEntropyLoss(reduction="none")
        batch_size, seq_len = input_ids.shape  # input_ids: [batch_size, seq_len]
        masking_schedule = kwargs.get("masking_schedule", None)

        # Create maskable_mask based on training mode and src_mask
        # For SFT: src_mask is provided, maskable_mask = ~src_mask
        # For pretrain: src_mask is None, maskable_mask = all True

        if src_mask is not None:
            maskable_mask = ~src_mask
        else:  # pretrain or midtrain
            maskable_mask = torch.ones_like(
                input_ids, dtype=torch.bool, device=input_ids.device
            )
            if masking_schedule is not None:
                prefix_probability = masking_schedule.get("prefix_probability", 0)
                truncate_probability = masking_schedule.get("truncate_probability", 0)
            else:
                prefix_probability = getattr(self.config, "prefix_probability", 0)
                truncate_probability = getattr(self.config, "truncate_probability", 0)
            if training_mode == "sft":
                prefix_probability = 0
                truncate_probability = 0
            # Generate random decisions for all batch items
            apply_prefix = (
                torch.rand(batch_size, device=input_ids.device) < prefix_probability
            )
            # Only apply truncation to rows that are NOT prefixed
            apply_truncate = (
                torch.rand(batch_size, device=input_ids.device) < truncate_probability
            )
            apply_truncate = apply_truncate & ~apply_prefix

            if prefix_probability > 0:
                maskable_mask = prefix_input_ids(input_ids, maskable_mask, apply_prefix)
            if truncate_probability > 0:
                input_ids = truncate_input_ids(
                    input_ids, apply_truncate, self.config.pad_token_id
                )
                maskable_mask = maskable_mask & (input_ids != self.config.pad_token_id)

        # add noise to input_ids
        sigma = (1 - sampling_eps) * torch.rand(
            input_ids.shape[0], device=input_ids.device
        ) + sampling_eps
        dsigma = torch.reciprocal(sigma)

        # Sample mask block size
        # Use mask_block_sizes from masking_probs if provided, otherwise fall back to config
        if masking_schedule is not None and "mask_block_sizes" in masking_schedule:
            mask_block_sizes = masking_schedule["mask_block_sizes"]
        else:
            mask_block_sizes = getattr(self.config, "mask_block_sizes", None)
        # Use masking_config if provided, otherwise fall back to config values
        if masking_schedule is not None:
            block_masking_probability = masking_schedule.get(
                "block_masking_probability", 0
            )
        else:
            block_masking_probability = getattr(
                self.config, "block_masking_probability", 0
            )
            if isinstance(block_masking_probability, list):
                if epoch is None:
                    block_masking_probability = block_masking_probability[0]
                else:
                    block_masking_probability = block_masking_probability[
                        epoch % len(block_masking_probability)
                    ]

        if block_masking_probability > 0 and mask_block_sizes is not None and len(mask_block_sizes) > 0:
            mask_block_size = mask_block_sizes[
                torch.randint(0, len(mask_block_sizes), (1,)).item()
            ]
        else:
            mask_block_size = 1

        noisy_input_ids = transition(
            input_ids,
            sigma[:, None],
            maskable_mask=maskable_mask,
            mask_token_id=mask_token_id,
            mask_block_size=mask_block_size,
        )
        loss_mask = noisy_input_ids == mask_token_id

        # Use gradient checkpointing if enabled
        if (
            hasattr(self, "gradient_checkpointing")
            and self.gradient_checkpointing
            and self.training
        ):
            # Define a function for gradient checkpointing
            def custom_forward(*inputs):
                return self.model(*inputs)

            # Apply gradient checkpointing to the model forward pass
            hidden_states = self._gradient_checkpointing_func(
                custom_forward, noisy_input_ids, attention_mask
            )
        else:
            hidden_states = self.model(
                input_ids=noisy_input_ids, attention_mask=attention_mask
            )

        logits = self.lm_head(hidden_states)
        logits = logits.float()
        # logits: [bs, seq_len, vocab_size]
        # Shifted logits and labels
        # logits: [bs, seq_len-1, vocab_size]
        logits = logits[..., :-1, :].contiguous()
        # weiran: if the shifted token is not masked in the original input, the loss is 0
        # loss_mask: [bs, seq_len-1]
        loss_mask = loss_mask[..., 1:].contiguous()
        target_ids = input_ids[..., 1:].contiguous()
        # loss: [bs, seq_len-1]
        loss = loss_func(
            logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1)
        ).reshape(target_ids.shape[0], -1)
        loss = loss.masked_fill(~loss_mask, 0)
        # weiran: divide by the number of tokens in the sequence instead of the number of masked tokens
        # justification is dsigma already accounts for the number of masked tokens
        # this is a hack to get something like per token loss
        # https://github.com/ML-GSAI/SMDM/blob/main/pretrain/train_mdm_rl.py#L281-L283
        loss = (dsigma[:, None] * loss).sum() / (
            input_ids.shape[0] * input_ids.shape[1]
        )
        return logits, loss
