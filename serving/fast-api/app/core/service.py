# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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

import torch
import re
import types
import uuid
import time
import logging
from typing import List, Optional, AsyncGenerator
from transformers import AutoTokenizer, AutoModel
from dream.model.generation_utils import DreamGenerationMixin, DreamGenerationConfig
from ..models.chat import ChatMessage, ChatCompletionResponse, ChatCompletionChoice, Usage
from ..models.chat import ChatCompletionStreamResponse, ChatCompletionStreamChoice
from .config import settings

logger = logging.getLogger(__name__)


class Service:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._model_name = None
        
    async def initialize(self, model_name: str = None):
        """Initialize the model and tokenizer"""
        if model_name is None:
            model_name = settings.model_name
            
        if self.model is not None and self._model_name == model_name:
            logger.info(f"Model {model_name} already loaded")
            return
            
        logger.info(f"Loading model: {model_name}")
        
        # Check device availability
        device = settings.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        # Load model and tokenizer
        torch_dtype = getattr(torch, settings.torch_dtype)
        if device == "cpu" and torch_dtype == torch.bfloat16:
            logger.info("Using float32 for CPU inference")
            torch_dtype = torch.float32
            
        token = settings.hf_token or None
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=settings.trust_remote_code,
            token=token,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=settings.trust_remote_code,
            token=token,
        )
        
        # Move model to device
        self.model = self.model.to(device).eval()
        
        # Ensure model.forward returns an object with logits/past_key_values
        try:
            original_forward = self.model.forward

            def _forward_adapter(_self, *args, **kwargs):
                output = original_forward(*args, **kwargs)
                if hasattr(output, 'logits'):
                    return output
                if isinstance(output, (tuple, list)):
                    logits = output[0]
                    return types.SimpleNamespace(logits=logits, past_key_values=None)
                return types.SimpleNamespace(logits=output, past_key_values=None)

            self.model.forward = types.MethodType(_forward_adapter, self.model)
            if hasattr(self.model, 'config'):
                try:
                    self.model.config.return_dict = True
                    self.model.config.use_cache = True
                except Exception:
                    pass
        except Exception:
            pass

        # Attach DreamGenerationMixin methods to the model instance
        # Instance methods
        self.model._validate_generated_length = types.MethodType(
            DreamGenerationMixin._validate_generated_length, self.model
        )
        self.model._prepare_generated_length = types.MethodType(
            DreamGenerationMixin._prepare_generated_length, self.model
        )
        self.model._prepare_generation_config = types.MethodType(
            DreamGenerationMixin._prepare_generation_config, self.model
        )
        self.model._prepare_special_tokens = types.MethodType(
            DreamGenerationMixin._prepare_special_tokens, self.model
        )
        self.model.diffusion_generate = types.MethodType(
            DreamGenerationMixin.diffusion_generate, self.model
        )
        self.model._sample = types.MethodType(
            DreamGenerationMixin._sample, self.model
        )
        # Static method needs a wrapper to avoid self being passed implicitly
        self.model._expand_inputs_for_generation = (
            lambda expand_size=1, input_ids=None, attention_mask=None: 
                DreamGenerationMixin._expand_inputs_for_generation(
                    expand_size=expand_size,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        )
        
        self._model_name = model_name
        logger.info(f"Model loaded successfully on {device}")
        
    def _strip_reasoning(self, text: str) -> str:
        """Remove internal reasoning tags like <think>...</think> from output."""
        if not isinstance(text, str) or not text:
            return text
        # Remove blocks between <think> and </think> (case-insensitive, multiline)
        cleaned = re.sub(r"(?is)<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>", "", text)
        # Remove any stray <think> or </think> tags
        cleaned = re.sub(r"(?is)<\s*/?\s*think\s*/?\s*>", "", cleaned)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer is None:
            return len(text.split())  # Fallback to word count
        return len(self.tokenizer.encode(text))
        
    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """Format messages using the tokenizer's chat template"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
            
        # Convert to dict format expected by tokenizer
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            message_dicts,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True
        )
        
        return inputs
        
    async def generate_chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        steps: Optional[int] = None,
        alg: Optional[str] = None,
        alg_temp: Optional[float] = None,
        block_length: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatCompletionResponse:
        """Generate chat completion"""
        
        if self.model is None or self.tokenizer is None:
            await self.initialize()
            
        # Use provided parameters or fall back to settings
        max_tokens = max_tokens or settings.max_tokens
        temperature = temperature if temperature is not None else settings.temperature
        top_p = top_p if top_p is not None else settings.top_p
        top_k = top_k if top_k is not None else settings.top_k
        steps = steps or settings.steps
        alg = alg or settings.alg
        alg_temp = alg_temp if alg_temp is not None else settings.alg_temp
        block_length = block_length or settings.block_length
        
        # Format input
        inputs = self._format_messages(messages)
        device = self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device
        input_ids = inputs.input_ids.to(device=device)
        attention_mask = inputs.attention_mask.to(device=device) if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None else None
        
        # Count input tokens
        prompt_tokens = input_ids.shape[1]
        
        try:
            # Use DreamGenerationMixin diffusion-based generation
            prompt_len = input_ids.shape[1]
            mask_id = getattr(self.tokenizer, 'mask_token_id', None)
            gen_config = DreamGenerationConfig(
                temperature=temperature if temperature is not None else settings.temperature,
                top_p=top_p if top_p is not None else settings.top_p,
                top_k=top_k if top_k is not None else settings.top_k,
                max_new_tokens=max_tokens,
                steps=steps,
                alg=alg,
                alg_temp=alg_temp,
                mask_token_id=mask_id,
                eos_token_id=getattr(self.tokenizer, 'eos_token_id', None),
                pad_token_id=getattr(self.tokenizer, 'pad_token_id', None),
                bos_token_id=getattr(self.tokenizer, 'bos_token_id', None),
                return_dict_in_generate=False,
            )

            sequences = self.model.diffusion_generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )

            # Handle return types
            if hasattr(sequences, 'sequences'):
                sequences = sequences.sequences

            # Take tokens after the prompt and filter mask/pad
            out_ids = sequences[0, prompt_len:]
            if mask_id is not None:
                out_ids = out_ids[out_ids != mask_id]
            pad_id = getattr(self.tokenizer, 'pad_token_id', None)
            if pad_id is not None:
                out_ids = out_ids[out_ids != pad_id]

            generation = self.tokenizer.decode(out_ids.tolist(), skip_special_tokens=True).strip()
            generation = self._strip_reasoning(generation)

            # Count completion tokens
            completion_tokens = int(out_ids.shape[0])
            total_tokens = prompt_tokens + completion_tokens
            
            # Create response
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            
            choice = ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generation),
                logprobs=None,
                finish_reason="stop"
            )
            
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
            
            response = ChatCompletionResponse(
                id=completion_id,
                model=model,
                choices=[choice],
                usage=usage
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise RuntimeError(f"Generation failed: {str(e)}")
            
    async def generate_chat_completion_stream(
        self,
        messages: List[ChatMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        steps: Optional[int] = None,
        alg: Optional[str] = None,
        alg_temp: Optional[float] = None,
        block_length: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
        """Generate streaming chat completion (simplified implementation)"""
        
        # For simplicity, we'll generate the full response and then stream it word by word
        # A more sophisticated implementation would integrate with the model's generation loop
        
        response = await self.generate_chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            steps=steps,
            alg=alg,
            alg_temp=alg_temp,
            block_length=block_length,
            stream=False,
            **kwargs
        )
        
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        content = response.choices[0].message.content
        content = self._strip_reasoning(content)
        words = content.split()
        
        # Stream words
        for i, word in enumerate(words):
            chunk_content = word + (" " if i < len(words) - 1 else "")
            
            choice = ChatCompletionStreamChoice(
                index=0,
                delta=ChatMessage(role="assistant", content=chunk_content),
                logprobs=None,
                finish_reason=None
            )
            
            chunk = ChatCompletionStreamResponse(
                id=completion_id,
                model=model,
                choices=[choice]
            )
            
            yield chunk
            
        # Final chunk with finish_reason
        final_choice = ChatCompletionStreamChoice(
            index=0,
            delta=ChatMessage(role="assistant", content=""),
            logprobs=None,
            finish_reason="stop"
        )
        
        final_chunk = ChatCompletionStreamResponse(
            id=completion_id,
            model=model,
            choices=[final_choice],
            usage=response.usage
        )
        
        yield final_chunk


# Global service instance
service = Service()
