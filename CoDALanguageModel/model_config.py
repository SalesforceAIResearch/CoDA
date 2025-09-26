from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class CoDAConfig(PretrainedConfig):
    model_type = "CoDA"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        head_dim=128,
        hidden_act="silu",
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=16,
        num_hidden_layers=28,
        num_key_value_heads=8,
        max_position_embeddings=40960,
        initializer_range=0.02,
        use_cache=True,
        use_sliding_window=False,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        rope_scaling=None,
        rope_theta=1000000,
        sliding_window=None,
        max_window_layers=28,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=151643,
        eos_token_id=151645,
        pad_token_id=151643,
        mask_token_id=151669,
        attention_kernel="flash_attention",
        prefix_probability=0,
        truncate_probability=0,
        block_masking_probability=[0.25, 0.5, 0.5, 0.75, 0.25],
        mask_block_sizes=[4, 8, 16, 32],
        sampling_eps=[0.001, 0.25, 0.5, 0.25, 0.001],  # minimum noise level
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.head_dim = head_dim
        self.attention_bias = attention_bias
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.attention_kernel = attention_kernel
        self.prefix_probability = prefix_probability
        self.truncate_probability = truncate_probability
        self.block_masking_probability = block_masking_probability
        self.mask_block_sizes = mask_block_sizes
        self.sampling_eps = sampling_eps

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
