import torch

from fim import LoRAConfig, ModelConfig, ModelLoader


class TestModelLoader:
    def test_dtype_mapping(self, model_config: ModelConfig, lora_config: LoRAConfig):
        loader = ModelLoader(model_config, lora_config)

        model_config.dtype = "float16"
        assert loader._get_dtype() == torch.float16

        model_config.dtype = "bfloat16"
        assert loader._get_dtype() == torch.bfloat16

        model_config.dtype = "float32"
        assert loader._get_dtype() == torch.float32

        model_config.dtype = None
        assert loader._get_dtype() is None

    def test_loader_initialization(self, model_config: ModelConfig, lora_config: LoRAConfig):
        loader = ModelLoader(model_config, lora_config)
        assert loader.model_config == model_config
        assert loader.lora_config == lora_config
