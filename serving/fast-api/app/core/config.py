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

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # API Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Model Configuration
    model_name: str = Field(default="Salesforce/CoDA-v0-Instruct", env="MODEL_NAME")
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")
    device: str = Field(default="cuda", env="DEVICE")
    torch_dtype: str = Field(default="bfloat16", env="TORCH_DTYPE")
    trust_remote_code: bool = Field(default=True, env="TRUST_REMOTE_CODE")
    
    # Generation Configuration
    max_tokens: int = Field(default=256, env="MAX_TOKENS")
    temperature: float = Field(default=0.0, env="TEMPERATURE")
    top_p: Optional[float] = Field(default=None, env="TOP_P")
    top_k: Optional[int] = Field(default=None, env="TOP_K")
    steps: int = Field(default=128, env="STEPS")
    alg: str = Field(default="entropy", env="ALG")
    alg_temp: float = Field(default=0.1, env="ALG_TEMP")
    block_length: int = Field(default=32, env="BLOCK_LENGTH")
    
    # API Keys (optional for compatibility)
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
