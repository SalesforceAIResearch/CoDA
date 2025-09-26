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

from typing import Optional, Dict, Any
from pydantic import BaseModel


class APIError(Exception):
    """Base API error class"""
    def __init__(self, message: str, status_code: int = 500, error_type: str = "internal_server_error"):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(message)


class ModelNotLoadedError(APIError):
    """Error when model is not loaded"""
    def __init__(self, message: str = "Model not loaded"):
        super().__init__(message, status_code=503, error_type="model_not_loaded")


class InvalidRequestError(APIError):
    """Error for invalid requests"""
    def __init__(self, message: str):
        super().__init__(message, status_code=400, error_type="invalid_request")


class AuthenticationError(APIError):
    """Error for authentication failures"""
    def __init__(self, message: str = "Invalid API key"):
        super().__init__(message, status_code=401, error_type="authentication_error")


class RateLimitError(APIError):
    """Error for rate limiting"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429, error_type="rate_limit_exceeded")


class ModelError(APIError):
    """Error during model inference"""
    def __init__(self, message: str):
        super().__init__(message, status_code=500, error_type="model_error")


class ErrorResponse(BaseModel):
    """Standard error response format"""
    error: Dict[str, Any]
    
    @classmethod
    def from_exception(cls, exc: APIError) -> "ErrorResponse":
        return cls(
            error={
                "message": exc.message,
                "type": exc.error_type,
                "code": None
            }
        )
    
    @classmethod
    def from_generic_exception(cls, exc: Exception) -> "ErrorResponse":
        return cls(
            error={
                "message": str(exc),
                "type": "internal_server_error",
                "code": None
            }
        )

