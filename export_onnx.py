from pathlib import Path
from collections import OrderedDict
from typing import Dict

from transformers import AutoModelForCausalLM

from optimum.exporters import TasksManager
from optimum.exporters.onnx import export
from optimum.exporters.onnx.model_configs import GPT2OnnxConfig
from optimum.utils import (
    NormalizedTextConfig,
    DummyTextInputGenerator,
    DummyPastKeyValuesGenerator,
)


class DummyTextInputGeneratorforTTTSGPT2(DummyTextInputGenerator):
    def __init__(self, task: str, normalized_config: NormalizedTextConfig, **kwargs):
        super().__init__(task, normalized_config, **kwargs)
        self.CUSTOM_SUPPORTED_INPUT_NAMES = (
            "emb",
            "speech_conditioning_latent",
            "input_length",
        )
        self.__class__.SUPPORTED_INPUT_NAMES = (
            super().SUPPORTED_INPUT_NAMES + self.CUSTOM_SUPPORTED_INPUT_NAMES
        )
        self.hidden_size = normalized_config.hidden_size

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        if input_name not in self.CUSTOM_SUPPORTED_INPUT_NAMES:
            return super().generate(
                input_name=input_name,
                framework=framework,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
            )
        elif input_name == "emb":
            shape = [self.batch_size, self.sequence_length, self.hidden_size]
            return self.random_float_tensor(
                shape=shape,
                min_value=0,
                max_value=1,
                framework=framework,
                dtype=float_dtype,
            )
        elif input_name == "speech_conditioning_latent":
            shape = [1, self.hidden_size]
            return self.random_float_tensor(
                shape=shape,
                min_value=0,
                max_value=1,
                framework=framework,
                dtype=float_dtype,
            )
        elif input_name == "input_length":
            shape = [
                1,
            ]
            return self.random_int_tensor(
                shape=shape,
                min_value=0,
                max_value=10,
                framework=framework,
                dtype=int_dtype,
            )
        else:
            raise Exception(f"Invalid input name {input_name}")


register_for_onnx = TasksManager.create_register("onnx")


@register_for_onnx("tttsgpt2", "text-to-audio")
class TTTSGPT2OnnxConfig(GPT2OnnxConfig):
    DEFAULT_ONNX_OPSET = 15
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers="layers",
        num_attention_heads="heads",
        hidden_size="model_dim",
        vocab_size="number_text_tokens",
    )
    _TASK_TO_COMMON_OUTPUTS = {
        "text-to-audio": OrderedDict(
            {
                "logits": {0: "batch_size", 1: "sequence_length"},
                "hidden_states": {0: "batch_size", 1: "sequence_length"},
            }
        )
    }
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGeneratorforTTTSGPT2,
        DummyPastKeyValuesGenerator,
    )
    # Sets the absolute tolerance to when validating the exported ONNX model against the
    # reference model.
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs
        # Decoders based on GPT2 require a position_ids input to avoid
        # generating wrong position_ids in the model itself:
        # https://github.com/huggingface/transformers/blob/v4.33.1/src/transformers/models/gpt2/modeling_gpt2.py#L802
        common_inputs["position_ids"] = {0: "batch_size", 1: "sequence_length"}
        # add emb to the common inputs
        # common_inputs["emb"] = {0: "batch_size", 1: "sequence_length"}
        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        # in the merged case, we need to allow the `sequence_length` to be variable, as it is not 1
        # during the first pass without past key values
        common_outputs = OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}})
        self.add_past_key_values(common_outputs, direction="outputs")
        common_outputs["hidden_states"] = {0: "batch_size", 1: "sequence_length"}
        return common_outputs


@register_for_onnx("tttsgpt2", "feature-extraction")
class TTTSGPT2PrefillOnnxConfig(TTTSGPT2OnnxConfig):
    _TASK_TO_COMMON_OUTPUTS = {
        "feature-extraction": OrderedDict(
            {
                "logits": {0: "batch_size", 1: "sequence_length"},
                "hidden_states": {0: "batch_size", 1: "sequence_length"},
            }
        )
    }

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs
        # Decoders based on GPT2 require a position_ids input to avoid
        # generating wrong position_ids in the model itself:
        # https://github.com/huggingface/transformers/blob/v4.33.1/src/transformers/models/gpt2/modeling_gpt2.py#L802
        common_inputs["position_ids"] = {0: "batch_size", 1: "sequence_length"}
        common_inputs["speech_conditioning_latent"] = {0: "batch_size"}
        return common_inputs

    def __init__(
        self,
        config: "PretrainedConfig",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
        **kwargs,
    ):
        use_past = True
        use_past_in_inputs = False
        self.is_decoder_with_past = False
        super().__init__(
            config=config,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            **kwargs,
        )


@register_for_onnx("tttsgpt2", "text-generation")
class TTTSGPT2DecodeOnnxConfig(TTTSGPT2OnnxConfig):
    _TASK_TO_COMMON_OUTPUTS = {
        "text-generation": OrderedDict(
            {
                "logits": {0: "batch_size", 1: "sequence_length"},
                "hidden_states": {0: "batch_size", 1: "sequence_length"},
            }
        )
    }

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs
        # Decoders based on GPT2 require a position_ids input to avoid
        # generating wrong position_ids in the model itself:
        # https://github.com/huggingface/transformers/blob/v4.33.1/src/transformers/models/gpt2/modeling_gpt2.py#L802
        common_inputs["position_ids"] = {0: "batch_size", 1: "sequence_length"}
        common_inputs["input_length"] = {}
        return common_inputs

    def __init__(
        self,
        config: "PretrainedConfig",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
        **kwargs,
    ):
        use_past = True
        use_past_in_inputs = True
        self.is_decoder_with_past = True
        super().__init__(
            config=config,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            **kwargs,
        )


def export_onnx(model, save_name, task):
    onnx_path = Path(save_name)
    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
        "onnx", model, task
    )
    onnx_config = onnx_config_constructor(model.config)
    print("--------------- for debug only ---------------")
    dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt")
    dummy_inputs = onnx_config.rename_ambiguous_inputs(dummy_inputs)
    print("--- dummy inputs:")
    for k, v in dummy_inputs.items():
        if isinstance(v, list):
            print(k)
        else:
            print(k, v.shape, v.dtype)
    inputs = onnx_config.ordered_inputs(model)
    for k, v in inputs.items():
        print("--- input placeholder", k, v)
    output_names = list(onnx_config.outputs.keys())
    print("--- output names", output_names)
    dummy_outputs = model(**dummy_inputs)
    print("--- real output names", dummy_outputs.keys())
    print(" ---------------- debug end ------------------")

    onnx_inputs, onnx_outputs = export(
        model, onnx_config, onnx_path, onnx_config.DEFAULT_ONNX_OPSET
    )
    print("Exported ONNX model to", onnx_path)


def main():
    model_save_dir = "tttsgpt2"

    model = AutoModelForCausalLM.from_pretrained(
        model_save_dir, trust_remote_code=True
    ).eval()
    print("--- model config", model.config)

    # Step 1: export gpt model to onnx
    # model.forward = model.gpt_forward
    # export_onnx(model, save_name="gpt_model.onnx", task="text-to-audio")

    # Step 2: export prefiller (the prefill stage in GPT) model to onnx
    model.forward = model.prefill_forward
    export_onnx(model, save_name="prefill_model.onnx", task="feature-extraction")

    # Step 3: export decoder (the decode state in GPT) model to onnx
    model.forward = model.decode_forward
    export_onnx(model, save_name="decoder_model.onnx", task="text-generation")


if __name__ == "__main__":
    main()
