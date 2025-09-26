import sys
import os
import importlib.util
import types
import argparse
import json

import safetensors
from transformers import AutoTokenizer, AutoModel



# Create a custom import hook for relative imports
class ModelImportHook:
    def __init__(self, base_path):
        self.base_path = base_path
        self.loaded_modules = {}

    def load_module_with_fixes(self, module_name, file_path):
        # Read the source code
        with open(file_path, "r") as f:
            source_code = f.read()

        # Replace relative imports with absolute imports
        replacements = [
            ("from .model_config", "from model_config"),
            ("from .attention", "from attention"),
            ("from .modeling_utils", "from modeling_utils"),
            ("from .generation_utils", "from generation_utils"),
        ]

        for old, new in replacements:
            source_code = source_code.replace(old, new)

        # Create a module
        module = types.ModuleType(module_name)
        module.__file__ = file_path

        # Store it in sys.modules
        sys.modules[module_name] = module
        self.loaded_modules[module_name] = module

        # Execute the modified source code
        exec(source_code, module.__dict__)

        return module




def load_safetensors_to_state_dict(model_dir: str) -> dict:
    """Load a model state dict from safetensors, supporting both sharded and single-file formats.

    This function loads model weights from the specified directory. It supports both
    sharded (`model.safetensors.index.json`) and single-file (`model.safetensors`) formats.

    Args:
        model_dir: Path to the directory containing the model files.

    Returns:
        dict: A state dictionary containing the model's parameters.

    Raises:
        FileNotFoundError: If neither the sharded nor single-file safetensors are found.
    """

    state_dict = {}
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    single_file = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(index_file):
        # Load sharded safetensors
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        for filename in set(weight_map.values()):
            path = os.path.join(model_dir, filename)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    state_dict[key] = f.get_tensor(key)
    elif os.path.exists(single_file):
        # Load single safetensor file
        state_dict = safetensors.torch.load_file(single_file)
    else:
        raise FileNotFoundError(
            f"No safetensors found in {model_dir}. Expected 'model.safetensors' or 'model.safetensors.index.json'."
        )

    return state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="CoDALanguageModel")
    parser.add_argument("--checkpoint_dir", "-c", type=str)
    parser.add_argument("--output_dir", "-o", type=str)
    args = parser.parse_args()


    # Add the Flex directory to sys.path
    coda_path = os.path.join(os.getcwd(), args.model_name)
    if coda_path not in sys.path:
        sys.path.insert(0, coda_path)

    # Create the import hook
    loader = ModelImportHook(coda_path)

    # Load all modules in the correct order
    model_config = loader.load_module_with_fixes(
        "model_config", os.path.join(coda_path, "model_config.py")
    )
    modeling_utils = loader.load_module_with_fixes(
        "modeling_utils", os.path.join(coda_path, "modeling_utils.py")
    )
    generation_utils = loader.load_module_with_fixes(
        "generation_utils", os.path.join(coda_path, "generation_utils.py")
    )
    attention = loader.load_module_with_fixes(
        "attention", os.path.join(coda_path, "attention.py")
    )
    modeling_coda = loader.load_module_with_fixes(
        "modeling_coda", os.path.join(coda_path, "modeling_coda.py")
    )

    # Extract the classes we need
    lm_class = modeling_coda.CoDALanguageModel
    model_class = modeling_coda.CoDAModel
    model_config = model_config.CoDAConfig
    model_config.register_for_auto_class()
    lm_class.register_for_auto_class("AutoModel")

    state_dict = load_safetensors_to_state_dict(args.checkpoint_dir)

    config_path = f"{args.model_name}/config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Create FlexConfig instance from dictionary
    config = model_config(**config_dict)

    # Now load the model with the proper config
    coda_model = lm_class(config)
    coda_model.load_state_dict(state_dict)

    output_dir = args.output_dir
    if "cqin" in args.checkpoint_dir:
        output_dir = output_dir.replace("cqin", "haolin.chen")
    coda_model.save_pretrained(output_dir)
    coda_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    coda_tokenizer.save_pretrained(output_dir)
