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

from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field
import time


class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    prompt: Union[str, List[str]] = Field(..., description="The prompt(s) to generate completions for")
    max_tokens: Optional[int] = Field(
        16, description="The maximum number of tokens to generate"
    )
    temperature: Optional[float] = Field(
        1.0, description="What sampling temperature to use, between 0 and 2"
    )
    top_p: Optional[float] = Field(
        1.0, description="Nucleus sampling parameter"
    )
    n: Optional[int] = Field(
        1, description="How many completions to generate for each prompt"
    )
    stream: Optional[bool] = Field(
        False, description="Whether to stream back partial progress"
    )
    logprobs: Optional[int] = Field(
        None, description="Include the log probabilities on the logprobs most likely tokens"
    )
    echo: Optional[bool] = Field(
        False, description="Echo back the prompt in addition to the completion"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Up to 4 sequences where the API will stop generating further tokens"
    )
    presence_penalty: Optional[float] = Field(
        0.0, description="Number between -2.0 and 2.0"
    )
    frequency_penalty: Optional[float] = Field(
        0.0, description="Number between -2.0 and 2.0"
    )
    best_of: Optional[int] = Field(
        1, description="Generates best_of completions server-side and returns the best"
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        None, description="Modify the likelihood of specified tokens appearing in the completion"
    )
    user: Optional[str] = Field(
        None, description="A unique identifier representing your end-user"
    )


class CompletionChoice(BaseModel):
    text: str = Field(..., description="The generated text")
    index: int = Field(..., description="The index of the choice in the list of choices")
    logprobs: Optional[Dict[str, Any]] = Field(
        None, description="The log probabilities of the tokens"
    )
    finish_reason: Literal["stop", "length"] = Field(
        ..., description="The reason the model stopped generating tokens"
    )


class CompletionResponse(BaseModel):
    id: str = Field(..., description="A unique identifier for the completion")
    object: Literal["text_completion"] = Field(
        "text_completion", description="The object type, which is always text_completion"
    )
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="The Unix timestamp (in seconds) of when the completion was created"
    )
    model: str = Field(..., description="The model used for completion")
    choices: List[CompletionChoice] = Field(
        ..., description="The list of completion choices"
    )
    usage: Optional[Dict[str, int]] = Field(
        None, description="Usage statistics for the completion request"
    )

