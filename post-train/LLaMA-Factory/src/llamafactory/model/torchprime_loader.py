"""
Custom model loader for torchprime models to integrate with LLaMA-Factory.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from omegaconf import OmegaConf
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoProcessor

# Add torchprime to path
torchprime_path = Path(__file__).parent.parent.parent.parent.parent.parent / "torchprime"
if torchprime_path.exists():
    sys.path.insert(0, str(torchprime_path))

from torchprime.torch_xla_models.model_utils import (
    set_default_dtype,
    load_safetensors_to_state_dict,
)
from torchprime.torch_xla_models.coda.modeling_qwen import Qwen3ForCausalLM

logger = logging.getLogger(__name__)


class TorchPrimeModelLoader:
    """Custom model loader for torchprime models."""
    
    def __init__(self, model_config_path: str, checkpoint_dir: str):
        """
        Initialize the torchprime model loader.
        
        Args:
            model_config_path: Path to the model configuration YAML file
            checkpoint_dir: Path to the checkpoint directory
        """
        self.model_config_path = model_config_path
        self.checkpoint_dir = checkpoint_dir
        self.model_config = None
        self.model = None
        self.tokenizer = None
        
    def load_model_config(self):
        """Load the model configuration from YAML file."""
        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(f"Model config not found: {self.model_config_path}")
        
        self.model_config = OmegaConf.load(self.model_config_path)
        logger.info(f"Loaded model config from: {self.model_config_path}")
        return self.model_config
    
    def load_model(self, device: str = "cuda") -> Qwen3ForCausalLM:
        """
        Load the torchprime model with weights.
        
        Args:
            device: Device to load the model on
            
        Returns:
            Loaded model
        """
        if self.model_config is None:
            self.load_model_config()
        
        logger.info("Initializing torchprime model...")
        
        with set_default_dtype(torch.bfloat16):
            # Initialize the model
            self.model = Qwen3ForCausalLM(self.model_config)
            
            # Load weights if checkpoint exists
            if os.path.exists(self.checkpoint_dir):
                logger.info(f"Loading weights from: {self.checkpoint_dir}")
                state_dict = load_safetensors_to_state_dict(self.checkpoint_dir)
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("Weights loaded successfully")
            else:
                logger.warning(f"Checkpoint directory not found: {self.checkpoint_dir}")
            
            # Create a proper HuggingFace config for compatibility
            from transformers import PretrainedConfig
            config = PretrainedConfig()
            setattr(config, 'model_type', 'qwen3')
            setattr(config, 'hidden_size', self.model_config.hidden_size if hasattr(self.model_config, 'hidden_size') else 8192)
            setattr(config, 'vocab_size', self.model_config.vocab_size if hasattr(self.model_config, 'vocab_size') else 151937)
            setattr(config, 'num_attention_heads', self.model_config.num_attention_heads if hasattr(self.model_config, 'num_attention_heads') else 32)
            setattr(config, 'num_hidden_layers', self.model_config.num_hidden_layers if hasattr(self.model_config, 'num_hidden_layers') else 32)
            setattr(config, 'intermediate_size', self.model_config.intermediate_size if hasattr(self.model_config, 'intermediate_size') else 28672)
            setattr(config, 'max_position_embeddings', self.model_config.max_position_embeddings if hasattr(self.model_config, 'max_position_embeddings') else 32768)
            setattr(config, 'to_dict', lambda: {
                'model_type': 'qwen3',
                'hidden_size': getattr(config, 'hidden_size'),
                'vocab_size': getattr(config, 'vocab_size'),
                'num_attention_heads': getattr(config, 'num_attention_heads'),
                'num_hidden_layers': getattr(config, 'num_hidden_layers'),
                'intermediate_size': getattr(config, 'intermediate_size'),
                'max_position_embeddings': getattr(config, 'max_position_embeddings'),
            })
            
            # Set the config on the model
            self.model.config = config
            
            # Add gradient checkpointing support for LLaMA-Factory compatibility
            self._add_gradient_checkpointing_support(self.model)
            
            # Move to device
            self.model = self.model.to(device)
            
        logger.info(f"Model loaded successfully on device: {device}")
        return self.model
    
    def _add_gradient_checkpointing_support(self, model):
        """
        Add gradient checkpointing support to the TorchPrime model.
        This makes it compatible with LLaMA-Factory's gradient checkpointing system.
        """
        from torch.utils.checkpoint import checkpoint
        from types import MethodType
        
        # Add the supports_gradient_checkpointing attribute
        setattr(model, 'supports_gradient_checkpointing', True)
        
        # Add gradient checkpointing state
        setattr(model, 'gradient_checkpointing', False)
        
        # Add _set_gradient_checkpointing method
        def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
            self.gradient_checkpointing = enable
            if enable and gradient_checkpointing_func:
                # Store the gradient checkpointing function for use in forward pass
                self._gradient_checkpointing_func = gradient_checkpointing_func
        
        setattr(model, '_set_gradient_checkpointing', MethodType(_set_gradient_checkpointing, model))
        
        # Add gradient_checkpointing_enable method
        def _gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            if gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {"use_reentrant": True}
            
            def custom_gradient_checkpointing_func(func, *args, **kwargs):
                # For TorchPrime models, func is a function, not a method
                # We need to handle this differently than the original implementation
                # Just ensure inputs require gradients if they're tensors
                for arg in args:
                    if torch.is_tensor(arg) and torch.is_floating_point(arg):
                        arg.requires_grad_(True)
                return checkpoint(func, *args, **kwargs, **gradient_checkpointing_kwargs)
            
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=custom_gradient_checkpointing_func)
            setattr(self.config, "use_cache", False)  # turn off when gradient checkpointing is enabled
            logger.info("Gradient checkpointing enabled for TorchPrime model.")
        
        setattr(model, 'gradient_checkpointing_enable', MethodType(_gradient_checkpointing_enable, model))
        
        # Add enable_input_require_grads method
        def enable_input_require_grads(self):
            """Enable input require grads for gradient checkpointing."""
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            
            # Register forward hook on the model to make inputs require grad
            self.register_forward_hook(make_inputs_require_grad)
        
        setattr(model, 'enable_input_require_grads', MethodType(enable_input_require_grads, model))
        
        # Add get_output_embeddings method for compatibility
        def get_output_embeddings(self):
            return self.lm_head
        
        setattr(model, 'get_output_embeddings', MethodType(get_output_embeddings, model))
    
    def load_tokenizer(self, model_args=None, checkpoint_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Load the tokenizer following the style of loader.py's load_tokenizer function.
        
        Args:
            model_args: Model arguments (optional, for compatibility with LLaMA-Factory)
            checkpoint_dir: Optional checkpoint directory (uses self.checkpoint_dir if None)
            
        Returns:
            Dictionary containing tokenizer and processor (following LLaMA-Factory style)
        """
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
            
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Initialize kwargs similar to loader.py
        init_kwargs = {}
        if model_args:
            # Extract relevant arguments from model_args if available
            if hasattr(model_args, 'use_fast_tokenizer'):
                init_kwargs['use_fast'] = model_args.use_fast_tokenizer
            if hasattr(model_args, 'split_special_tokens'):
                init_kwargs['split_special_tokens'] = model_args.split_special_tokens
        
        # Set default values for TorchPrime models
        init_kwargs.setdefault('use_fast', True)
        init_kwargs.setdefault('split_special_tokens', False)
        init_kwargs['padding_side'] = "right"
        
        # Load tokenizer with fallback mechanism (following loader.py style)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-1.7B",  # Default for TorchPrime models
                **init_kwargs
            )
        except ValueError:  # try the fast one
            logger.warning("Failed to load tokenizer, trying with use_fast=True")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-1.7B",
                use_fast=True,
                padding_side="right",
                **{k: v for k, v in init_kwargs.items() if k != 'use_fast'}
            )
        
        # Add new special tokens if specified (following loader.py style)
        if model_args and hasattr(model_args, 'new_special_tokens') and model_args.new_special_tokens is not None:
            num_added_tokens = self.tokenizer.add_special_tokens(
                dict(additional_special_tokens=model_args.new_special_tokens),
                replace_additional_special_tokens=False,
            )
            logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
            if num_added_tokens > 0 and hasattr(model_args, 'resize_vocab') and not model_args.resize_vocab:
                model_args.resize_vocab = True
                logger.warning("New tokens have been added, changed `resize_vocab` to True.")
        
        # Add mask token for diffusion models (TorchPrime specific)
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_tokens("<|mask|>", special_tokens=True)
            self.tokenizer.add_special_tokens(
                {"mask_token": "<|mask|>"}, 
                replace_additional_special_tokens=False
            )
            logger.info("Added mask token for diffusion model")
        
        # Set default tokens for diffusion (following loader.py style)
        # if self.tokenizer.mask_token_id is None:
        #     self.tokenizer.mask_token_id = 811
        #     logger.info("Set default mask_token_id to 811")
        
        # if self.tokenizer.pad_token_id is None:
        #     self.tokenizer.pad_token_id = 30399
        #     logger.info("Set default pad_token_id to 30399")
        
        if self.tokenizer.sep_token_id is None:
            self.tokenizer.sep_token_id = 151664
            logger.info(f"Set default sep_token_id to {self.tokenizer.sep_token_id}")
        
        # Set BOS token for Qwen models
        if self.tokenizer.bos_token_id is None:
            # For Qwen models, typically use the same token as EOS or a specific BOS token
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.bos_token_id = self.tokenizer.eos_token_id
                logger.info(f"Set bos_token_id to eos_token_id: {self.tokenizer.bos_token_id}")
            else:
                # Fallback: use a common BOS token ID for Qwen
                self.tokenizer.bos_token_id = 151644  # Common BOS token for Qwen models
                logger.info(f"Set default bos_token_id to 151644")
        
        # Also set the BOS token string if needed
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = "<|endoftext|>"  # Common BOS token for Qwen
            logger.info("Set bos_token to '<|endoftext|>'")
        # Load processor (following loader.py style)
        processor = None
        # try:
        #     processor = AutoProcessor.from_pretrained("Qwen/Qwen3-1.7B", **init_kwargs)
        #     setattr(processor, "tokenizer", self.tokenizer)
            
        #     # Set image-related attributes if available
        #     if hasattr(model_args, 'image_resolution'):
        #         setattr(processor, "image_resolution", model_args.image_resolution)
            
        #     # Get image sequence length from config if available
        #     if self.model_config:
        #         try:
        #             from ..extras.misc import get_image_seqlen
        #             image_seqlen = get_image_seqlen(self.model_config)
        #             setattr(processor, "image_seqlen", image_seqlen)
        #         except ImportError:
        #             logger.warning("Could not import get_image_seqlen function")
                    
        # except Exception as e:
        #     logger.warning(f"Failed to load processor: {e}")
        #     processor = None
        
        # Validate processor (following loader.py style)
        if processor is not None and "Processor" not in processor.__class__.__name__:
            processor = None
            logger.warning("Processor is not a valid Processor class, setting to None")
        
        logger.info(f"Tokenizer loaded successfully from: {checkpoint_dir}")
        logger.info(f"Mask token ID: {self.tokenizer.mask_token_id}")
        logger.info(f"Pad token ID: {self.tokenizer.pad_token_id}")
        logger.info(f"Vocab size: {len(self.tokenizer)}")
        logger.info(f"final tokenizer: {self.tokenizer}")
        
        # Return in LLaMA-Factory format
        return {"tokenizer": self.tokenizer, "processor": processor}


def load_torchprime_model(
    model_config_path: str,
    checkpoint_dir: str,
    device: str = "cuda"
) -> tuple[Qwen3ForCausalLM, PreTrainedTokenizer]:
    """
    Convenience function to load both model and tokenizer.
    
    Args:
        model_config_path: Path to the model configuration YAML file
        checkpoint_dir: Path to the checkpoint directory
        device: Device to load the model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    loader = TorchPrimeModelLoader(model_config_path, checkpoint_dir)
    model = loader.load_model(device)
    tokenizer_module = loader.load_tokenizer()
    return model, tokenizer_module["tokenizer"] 