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

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from ..core.service import service
from ..core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1234567890  # Placeholder timestamp
    owned_by: str = "dream-api"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model_loaded = service.model is not None
        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            model_name=settings.model_name,
            device=settings.device
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI compatible)"""
    models = [
        ModelInfo(
            id=settings.model_name,
            object="model",
            created=1234567890,
            owned_by="dream-api"
        )
    ]
    return ModelsResponse(data=models)


@router.get("/models", response_model=ModelsResponse)
async def list_models_alt():
    """Alternative endpoint for listing models"""
    return await list_models()
