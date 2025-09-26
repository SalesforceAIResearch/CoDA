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

import json
import logging
from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from typing import Optional
from ..models.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse
)
from ..core.service import service
from ..core.config import settings
from ..core.errors import AuthenticationError, ModelError, InvalidRequestError

logger = logging.getLogger(__name__)

router = APIRouter()


def verify_api_key(authorization: Optional[str] = Header(None)):
    """Optional API key verification"""
    if settings.api_key:
        if not authorization:
            raise AuthenticationError("Missing authorization header")
        
        # Handle Bearer token format
        if authorization.startswith("Bearer "):
            api_key = authorization[7:]
        else:
            api_key = authorization
            
        if api_key != settings.api_key:
            raise AuthenticationError("Invalid API key")
    return True


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    _: bool = Depends(verify_api_key)
):
    """Create a chat completion (OpenAI compatible)"""
    try:
        # Initialize model if not already loaded
        if service.model is None:
            await service.initialize()
            
        # Handle streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(request),
                media_type="text/plain",
                headers={"X-Accel-Buffering": "no"}
            )
            
        # Generate completion
        response = await service.generate_chat_completion(
            messages=request.messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            steps=request.steps,
            alg=request.alg,
            alg_temp=request.alg_temp,
            block_length=request.block_length,
            stream=False
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise ModelError(f"Generation failed: {str(e)}")


async def stream_chat_completion(request: ChatCompletionRequest):
    """Stream chat completion responses"""
    try:
        # Initialize model if not already loaded
        if service.model is None:
            await service.initialize()
            
        async for chunk in service.generate_chat_completion_stream(
            messages=request.messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            steps=request.steps,
            alg=request.alg,
            alg_temp=request.alg_temp,
            block_length=request.block_length
        ):
            # Format as Server-Sent Events
            chunk_json = chunk.model_dump_json()
            yield f"data: {chunk_json}\n\n"
            
        # End stream
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming chat completion: {str(e)}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "internal_server_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion_alt(
    request: ChatCompletionRequest,
    _: bool = Depends(verify_api_key)
):
    """Alternative endpoint for chat completions (without /v1 prefix)"""
    return await create_chat_completion(request, _)
