"""Type stubs for Transformers library."""

from typing import Any, Dict, List, Optional, Union

class AutoTokenizer:
    eos_token_id: int

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: Any) -> "AutoTokenizer": ...
    def __call__(self, text: str, return_tensors: str = "pt", **kwargs: Any) -> Any: ...
    def decode(self, tokens: Any, skip_special_tokens: bool = True) -> str: ...

class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(
        cls, model_name: str, **kwargs: Any
    ) -> "AutoModelForCausalLM": ...
    def generate(self, input_ids: Any, **kwargs: Any) -> Any: ...
    def to(self, device: str) -> "AutoModelForCausalLM": ...

class AutoModelForSequenceClassification:
    @classmethod
    def from_pretrained(
        cls, model_name: str, **kwargs: Any
    ) -> "AutoModelForSequenceClassification": ...
    def __call__(self, input_ids: Any, **kwargs: Any) -> Any: ...
    def to(self, device: str) -> "AutoModelForSequenceClassification": ...

class BitsAndBytesConfig:
    def __init__(
        self,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bnb_4bit_compute_dtype: Any = None,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = False,
        **kwargs: Any
    ): ...

class pipeline:
    def __init__(self, task: str, model: str, **kwargs: Any): ...
    def __call__(self, inputs: Any, **kwargs: Any) -> Any: ...
