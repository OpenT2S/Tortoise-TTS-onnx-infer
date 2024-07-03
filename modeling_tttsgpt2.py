import time
from transformers import AutoConfig
from optimum.utils import NormalizedTextConfig
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import onnxruntime

from optimum.exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS
from optimum.utils import check_if_transformers_greater

from optimum.onnxruntime.modeling_ort import ORTModel


if check_if_transformers_greater("4.25.0"):
    from transformers.generation import GenerationMixin
else:
    from transformers.generation_utils import GenerationMixin


logger = logging.getLogger(__name__)


class ORTModelForTTTSGPT2CausalLM(ORTModel, GenerationMixin):
    """
    ONNX model with a causal language modeling head for ONNX Runtime inference. This class officially supports bloom, codegen, falcon, gpt2, gpt_bigcode, gpt_neo, gpt_neox, gptj, llama.
    """

    main_input_name = "input_ids"

    def __init__(
        self,
        model: onnxruntime.InferenceSession,
        config: NormalizedTextConfig,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        generation_config: Optional[GenerationConfig] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        if use_io_binding is None:
            use_io_binding = model.get_providers()[0] in [
                "CPUExecutionProvider",
                "CUDAExecutionProvider",
            ]

        super().__init__(
            model, config, use_io_binding, model_save_dir, preprocessors, **kwargs
        )

        self.num_pkv = 2
        self.normalized_config = config
        self.key_value_input_names = [
            key for key in self.inputs_names if (".key" in key) or (".value" in key)
        ]
        self.key_value_output_names = [
            key for key in self.output_names if (".key" in key) or (".value" in key)
        ]
        self.use_cache = len(self.key_value_input_names) > 0

        self.generation_config = generation_config
        self.onnx_paths = [self.model_path]
        self.use_merged = "use_cache_branch" in self.inputs_names
        self.model_type = self.config.model_type

        self.use_fp16 = False
        for inp in model.get_inputs():
            if (
                inp.name == "past_key_values" or inp.name in self.key_value_input_names
            ) and inp.type == "tensor(float16)":
                self.use_fp16 = True
                break

        # Reference: https://github.com/huggingface/optimum/pull/1381
        model_type = config.model_type.replace("_", "-")
        if (
            model_type in MODEL_TYPES_REQUIRING_POSITION_IDS
            and "position_ids" not in self.inputs_names
        ):
            logger.warning(
                f"ORTModelForCausalLM loaded a legacy ONNX model with no position_ids input, although this input is required for batched generation for the architecture {model_type}. "
                "We strongly encourage to re-export the model with optimum>=1.14 for position_ids and batched inference support."
            )

        if use_cache ^ self.use_cache:
            raise ValueError(
                f"`use_cache` was set to `{use_cache}` but the loaded model only supports `use_cache={self.use_cache}`. "
                f"Please load your current model with `use_cache={self.use_cache}` or export the original model "
                f"once again with `use_cache={use_cache}` when calling the `from_pretrained` method. "
                "To export your model, simply set `export=True`."
            )

        if use_io_binding and not use_cache:
            raise ValueError(
                "The parameters combination use_cache=False, use_io_binding=True is not supported. "
                "Please either pass use_cache=True, use_io_binding=True (default), or use_cache=False, use_io_binding=False."
            )

    def forward(
        self,
        input_ids: torch.LongTensor,
        speech_conditioning_latent: Optional[torch.FloatTensor] = None,
        input_length: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache_branch: bool = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # adding use_cache_branch in the signature here is just a hack for IO Binding
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        inputs = {}
        known_output_shapes = {}
        use_cache_branch = None
        loss = None
        if self.use_cache:
            if past_key_values is not None:
                # Flatten the past_key_values (gpt_bigcode has fused key/value cache, so no need to flatten it)
                if self.model_type != "gpt_bigcode":
                    past_key_values = tuple(
                        past_key_value
                        for pkv_per_layer in past_key_values
                        for past_key_value in pkv_per_layer
                    )

            # Create dummy past_key_values for decoder first generation step if none given
            use_cache_branch, past_key_values, known_output_shapes = (
                self.prepare_past_key_values(input_ids, past_key_values, use_torch)
            )

        if self.use_io_binding:
            # TODO: fix transformers generate to have contiguous input_ids here already
            # For an unknown reason, calling `contiguous()` here is necessary to not have errors
            # on CPU EP with batch size > 1, despite it being also called in _prepare_io_binding.
            # I suspect the reason is the contiguous python list that messes something up?
            model_inputs = [input_ids.contiguous()]

            if "input_length" in self.inputs_names:
                model_inputs.append(input_length)

            if "attention_mask" in self.inputs_names:
                model_inputs.append(attention_mask)

            if "position_ids" in self.inputs_names:
                if position_ids is None:
                    raise ValueError(
                        "position_ids was not passed but is a required input for this ONNX model."
                    )
                model_inputs.append(position_ids.contiguous())

            if past_key_values is not None:
                model_inputs += past_key_values

            if use_cache_branch is not None:
                model_inputs.append(use_cache_branch)

            if "labels" in self.inputs_names:
                model_inputs.append(labels)
                known_output_shapes.update({"loss": []})

            if "speech_conditioning_latent" in self.inputs_names:
                model_inputs.append(speech_conditioning_latent)

            io_binding, output_shapes, output_buffers = self._prepare_io_binding(
                self.model,
                *model_inputs,
                known_output_shapes=known_output_shapes,
                ordered_input_names=self._ordered_input_names,
            )

            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            if self.use_cache:
                # Tuple of length equal to : number of layer * number of past_key_value per decoder layer(2)
                past_key_values = ()
                for name in self.key_value_output_names:
                    past_key_values += (output_buffers[name].view(output_shapes[name]),)

            logits = output_buffers["logits"].view(output_shapes["logits"])
            hidden_states = output_buffers["hidden_states"].view(
                output_shapes["hidden_states"]
            )

            if "loss" in self.output_names:
                loss = output_buffers["loss"].view(output_shapes["loss"])
        else:
            inputs["input_ids"] = (
                input_ids.cpu().detach().numpy() if use_torch else input_ids
            )

            if "attention_mask" in self.inputs_names:
                inputs["attention_mask"] = (
                    attention_mask.cpu().detach().numpy()
                    if use_torch
                    else attention_mask
                )

            if "labels" in self.inputs_names:
                inputs["labels"] = (
                    labels.cpu().detach().numpy() if use_torch else labels
                )

            if "position_ids" in self.inputs_names:
                if position_ids is None:
                    raise ValueError(
                        "position_ids was not passed but is a required input for this ONNX model."
                    )
                inputs["position_ids"] = (
                    position_ids.cpu().detach().numpy() if use_torch else position_ids
                )

            if "speech_conditioning_latent" in self.inputs_names:
                inputs["speech_conditioning_latent"] = (
                    speech_conditioning_latent.cpu().detach().numpy()
                    if use_torch
                    else speech_conditioning_latent
                )

            if "input_length" in self.inputs_names:
                inputs["input_length"] = (
                    input_length.cpu().detach().numpy() if use_torch else input_length
                )

            # Add the past_key_values to the decoder inputs
            if past_key_values is not None:
                for input_name, past_key_value in zip(
                    self.key_value_input_names, past_key_values
                ):
                    inputs[input_name] = (
                        past_key_value.cpu().detach().numpy()
                        if use_torch
                        else past_key_value
                    )

            if use_cache_branch is not None:
                inputs["use_cache_branch"] = (
                    use_cache_branch.cpu().detach().numpy()
                    if use_torch
                    else use_cache_branch
                )

            outputs = self.model.run(None, inputs)

            if self.use_cache:
                # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 for the self-attention)
                past_key_values = tuple(
                    torch.from_numpy(outputs[self.output_names[key]]).to(self.device)
                    for key in self.key_value_output_names
                )

            logits = torch.from_numpy(outputs[self.output_names["logits"]]).to(
                self.device
            )
            hidden_states = torch.from_numpy(
                outputs[self.output_names["hidden_states"]]
            ).to(self.device)
            if "loss" in self.output_names:
                loss = torch.from_numpy(outputs[self.output_names["loss"]]).to(
                    self.device
                )

        if self.use_cache and self.model_type != "gpt_bigcode":
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and
            # per decoder layer
            past_key_values = tuple(
                past_key_values[i : i + self.num_pkv]
                for i in range(0, len(past_key_values), self.num_pkv)
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
        )

    def prepare_past_key_values(
        self,
        input_ids: Union[None, torch.LongTensor, np.ndarray],
        past_key_values: Union[None, Tuple[torch.FloatTensor], Tuple[np.ndarray]],
        use_torch: bool,
    ):
        sequence_length = input_ids.shape[1]

        constructor = torch if use_torch else np
        if self.use_merged:
            # Uses without/with branch of a merged decoder depending on whether real past key values are passed
            use_cache_branch = constructor.full((1,), past_key_values is not None)
        else:
            # Uses separate decoders
            use_cache_branch = None

        if use_torch and use_cache_branch is not None:
            use_cache_branch = use_cache_branch.to(self.device)

        pkv_output_shape = {}
        # Generate dummy past for the first forward if uses a merged decoder
        if past_key_values is None:
            batch_size = input_ids.shape[0]
            embed_size_per_head = (
                self.normalized_config.hidden_size
                // self.normalized_config.num_attention_heads
            )
            if self.model_type == "gemma":
                num_attention_heads = self.normalized_config.num_key_value_heads
                embed_size_per_head = self.normalized_config.head_dim
            elif self.model_type in {"mistral", "llama", "qwen2"}:
                num_attention_heads = self.normalized_config.num_key_value_heads
            else:
                num_attention_heads = self.normalized_config.num_attention_heads

            dtype = constructor.float16 if self.use_fp16 else constructor.float32

            # TODO: find a way to better handle this controlflow, this is EXTREMELY UGLY.
            # "1" is the dummy sequence length
            if self.model_type == "bloom":
                shape_value = (batch_size * num_attention_heads, 0, embed_size_per_head)
                shape_key = (batch_size * num_attention_heads, embed_size_per_head, 0)
                key = constructor.zeros(shape_key, dtype=dtype)
                value = constructor.zeros(shape_value, dtype=dtype)

                if use_torch:
                    key = key.to(self.device)
                    value = value.to(self.device)

                past_key_values = tuple(
                    key_or_value
                    for _ in range(len(self.key_value_input_names) // 2)
                    for key_or_value in [key, value]
                )

                for name, value in zip(self.key_value_output_names, past_key_values):
                    shape = [*value.shape]
                    index = 1 if "value" in name else 2

                    shape[index] += sequence_length
                    pkv_output_shape[name] = shape
            elif self.model_type == "gpt_bigcode":
                # GPT BigCode uses muti-query attention, and has the specificity of putting both key and value in the same cache tensor.
                shape_key_and_value = (batch_size, 0, embed_size_per_head * 2)
                key_and_value = constructor.zeros(shape_key_and_value, dtype=dtype)

                if use_torch:
                    key_and_value = key_and_value.to(self.device)

                past_key_values = tuple(
                    key_and_value for _ in range(len(self.key_value_input_names))
                )

                for name, value in zip(self.key_value_output_names, past_key_values):
                    shape = [*value.shape]
                    shape[1] += sequence_length
                    pkv_output_shape[name] = shape
            else:
                num_key_value_heads = (
                    self.num_key_value_heads
                    if self.model_type == "falcon"
                    else num_attention_heads
                )

                shape = (batch_size, num_key_value_heads, 0, embed_size_per_head)
                key_or_value = constructor.zeros(shape, dtype=dtype)

                if use_torch:
                    key_or_value = key_or_value.to(self.device)

                past_key_values = tuple(
                    key_or_value for _ in range(len(self.key_value_input_names))
                )

                for name, value in zip(self.key_value_output_names, past_key_values):
                    shape = [*value.shape]
                    shape[2] += sequence_length
                    pkv_output_shape[name] = shape

        return use_cache_branch, past_key_values, pkv_output_shape

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            # only last token for inputs_ids if past is defined in kwargs
            input_ids = input_ids[:, -1].unsqueeze(-1)
            input_length = torch.tensor(
                [kwargs.get("input_length", None)], dtype=torch.long
            ).to(input_ids.device)
        else:
            # At prefill state, we need to load speech_conditioning_latent
            speech_conditioning_latent = kwargs.get("speech_conditioning_latent", None)
            if speech_conditioning_latent is not None:
                speech_conditioning_latent = np.fromfile(
                    speech_conditioning_latent, dtype=np.float32
                ).reshape(1, -1, 1024)
                speech_conditioning_latent = torch.from_numpy(
                    speech_conditioning_latent[:, 0, :]  # shape (1, 1024)
                ).to(input_ids.device)

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        generation_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        if past_key_values is None:
            generation_inputs["speech_conditioning_latent"] = speech_conditioning_latent
        else:
            generation_inputs["input_length"] = input_length
        return generation_inputs

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )


def main():
    # ------------------------------ load config ------------------------------
    config_save_path = "tttsgpt2"
    config = AutoConfig.from_pretrained(config_save_path, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(
        config_save_path, trust_remote_code=True
    )
    # disable sampling for test only
    generation_config.do_sample = False

    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers="layers",
        num_attention_heads="heads",
        hidden_size="model_dim",
        vocab_size="number_text_tokens",
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        provider = "CUDAExecutionProvider"
        use_io_binding = True
    else:
        device = torch.device("cpu")
        provider = "CPUExecutionProvider"
        use_io_binding = False

    session_options = None
    provider_options = None

    # ------------------------------ load model ------------------------------

    prefill_onnx_model = ORTModel.load_model(
        "prefill_model.onnx",
        provider=provider,
        session_options=session_options,
        provider_options=provider_options,
    )

    decode_onnx_model = ORTModel.load_model(
        "decoder_model.onnx",
        provider=provider,
        session_options=session_options,
        provider_options=provider_options,
    )

    preprocessors = []

    prefill_model = ORTModelForTTTSGPT2CausalLM(
        prefill_onnx_model,
        config=NORMALIZED_CONFIG_CLASS(config),
        use_io_binding=False,
        preprocessors=preprocessors,
        use_cache=False,
        generation_config=generation_config,
    )
    prefill_model.use_cache = True

    decode_model = ORTModelForTTTSGPT2CausalLM(
        decode_onnx_model,
        config=NORMALIZED_CONFIG_CLASS(config),
        use_io_binding=use_io_binding,
        preprocessors=preprocessors,
        use_cache=True,
        generation_config=generation_config,
    )

    # ------------------------------ generate prefill ------------------------------
    inputs_ids = torch.tensor(
        [
            [0, 0, 0, 0, 22, 4, 26, 2, 119, 52, 2, 51, 2, 32, 124, 213, 2, 147, 0]
        ],  # the first 0s, will be removed
        dtype=torch.long,
    ).to(device)
    print("--- inputs_ids", inputs_ids.shape)

    prefill_out = prefill_model.generate(
        inputs_ids,
        speech_conditioning_latent="emb.1x18x1024.float32.bin",
        max_length=inputs_ids.shape[1] + 1,
    )

    input_ids = prefill_out.sequences[:, -1:]
    print("--- input_ids", input_ids.shape, input_ids)
    hidden_states = prefill_out.hidden_states
    print("--- hidden_states", hidden_states[0].shape)  # shape: (1, T, 1024)
    past_key_values = prefill_out.past_key_values

    # ----------------------------- generate decode -----------------------------
    attention_mask = torch.ones(
        (1, inputs_ids.shape[1] + 1), dtype=torch.long, device=inputs_ids.device
    )  # init attention mask with shape (1, T+1)
    decode_out = decode_model.generate(
        input_ids,
        input_length=inputs_ids.shape[1],
        past_key_values=past_key_values,
        attention_mask=attention_mask,
    )
    seq = decode_out.sequences
    print("--- output", seq.shape, seq)
    hidden_states = torch.cat(
        (hidden_states[0][:, -1:, :],) + decode_out.hidden_states, dim=1
    )
    print("hidden_states", hidden_states.shape)

    # save to file
    save_path = (
        f"hidden_states.1x{hidden_states.shape[1]}x{hidden_states.shape[2]}.float32.bin"
    )
    hidden_states.cpu().data.numpy().astype(np.float32).tofile(save_path)
    print("saved to file: ", save_path)

    # ------------------------------ speed test ------------------------------
    num_infer_iter = 100  # for test
    total_time_ns = 0
    squared_time_diffs = 0  # for calculating standard deviation
    for _ in range(num_infer_iter):
        try:
            time_start = time.time_ns()
            prefill_out = prefill_model.generate(
                inputs_ids,
                speech_conditioning_latent="emb.1x18x1024.float32.bin",
                max_length=inputs_ids.shape[1] + 1,
            )
            decode_out = decode_model.generate(
                input_ids,
                input_length=inputs_ids.shape[1],
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )

            time_end = time.time_ns()
            time_diff_ns = time_end - time_start
        except Exception as e:
            print(f"Error during inference: {e}")
            # Handle the error appropriately (e.g., log, raise)
            continue  # Skip this iteration on error

        total_time_ns += time_diff_ns
        squared_time_diffs += time_diff_ns**2

    average_time_ms = total_time_ns / (1e6 * num_infer_iter)
    std_dev_ms = (
        (squared_time_diffs / num_infer_iter) - (total_time_ns / num_infer_iter) ** 2
    ) ** 0.5 / 1e6

    print(f"--- Average inference time (onnxruntime): {average_time_ms:.3f} ms")
    print(f"--- Standard deviation: {std_dev_ms:.3f} ms")

    seq = decode_out.sequences
    print("--- output", seq.shape, seq)


if __name__ == "__main__":
    main()
