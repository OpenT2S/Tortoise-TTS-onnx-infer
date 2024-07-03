import os
import numpy as np
import torch
from tortoise.models.autoregressive import UnifiedVoice
from autoregressive import TTTSGPT2Config, TTTSGPT2Model
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import GenerationConfig

DEFAULT_MODELS_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "tortoise", "models"
)
MODELS_DIR = os.environ.get("TORTOISE_MODELS_DIR", DEFAULT_MODELS_DIR)
MODELS = {
    "autoregressive.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/autoregressive.pth",
    "classifier.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/classifier.pth",
    "clvp2.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/clvp2.pth",
    "cvvp.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/cvvp.pth",
    "diffusion_decoder.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/diffusion_decoder.pth",
    "vocoder.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/vocoder.pth",
    "rlg_auto.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_auto.pth",
    "rlg_diffuser.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_diffuser.pth",
}


def get_model_path(model_name, models_dir=MODELS_DIR):
    """
    Get path to given model, download it if it doesn't exist.
    """
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")
    model_path = hf_hub_download(
        repo_id="Manmay/tortoise-tts", filename=model_name, cache_dir=models_dir
    )
    return model_path


def load_cond(path):
    emb = np.fromfile(path, dtype=np.float32).reshape(1, -1, 1024)
    emb = torch.from_numpy(emb)
    condition = emb[:, 0, :]  # shape (1, 1024)
    return condition


def convert_weight(model):
    converted_state_dict = {}
    key_maps = {
        "mel_embedding.weight": [
            "embeddings.weight",
            "transformer.wte.weight",
        ],  # set wte. Here list is supported also.
        "final_norm.weight": "lm_head.0.weight",
        "final_norm.bias": "lm_head.0.bias",
        "mel_head.weight": "lm_head.1.weight",
        "mel_head.bias": "lm_head.1.bias",
        "mel_pos_embedding.emb.weight": "text_pos_embedding.emb.weight",  # This's weird, but Yes.
        "text_pos_embedding.emb.weight": "real_text_pos_embedding.emb.weight",
    }
    for k, v in model.named_parameters():
        if k.startswith("gpt"):
            k = k.replace("gpt.", "transformer.")
            converted_state_dict[k] = v
            continue
        # don't change
        if k in [
            "text_embedding.weight"
            # "mel_pos_embedding.emb.weight",
            # "text_pos_embedding.emb.weight",
        ]:
            converted_state_dict[k] = v
            continue

        for old, new in key_maps.items():
            if k == old:
                if not isinstance(new, list):
                    new = [new]
                for n in new:
                    converted_state_dict[n] = v
                continue
    converted_state_dict["speech_conditioning_latent"] = load_cond(
        "emb.1x18x1024.float32.bin"
    )
    return converted_state_dict


def create_soft_link(target, link_name):
    """
    Create a soft link pointing to the target file or directory.

    Parameters:
    - target: The path of the target file or directory.
    - link_name: The path of the soft link to be created.
    """
    # Check if the soft link already exists
    if os.path.islink(link_name):
        print(f"The soft link {link_name} already exists.")
    else:
        try:
            # Create the soft link
            os.symlink(target, link_name)
            print(f"Soft link {link_name} created successfully, pointing to {target}.")
        except OSError as e:
            print(f"Error creating soft link {link_name}: {e}")


def main():
    model_save_dir = "tttsgpt2"
    print("Loading model...")
    autoregressive = (
        UnifiedVoice(
            max_mel_tokens=604,
            max_text_tokens=402,
            max_conditioning_inputs=2,
            layers=30,
            model_dim=1024,
            heads=16,
            number_text_tokens=255,
            start_text_token=255,
            checkpointing=False,
            train_solo_embeddings=False,
        )
        .cpu()
        .eval()
    )
    autoregressive.load_state_dict(
        torch.load(get_model_path("autoregressive.pth", MODELS_DIR)), strict=False
    )
    autoregressive.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)
    print("Model loaded.")
    print("--- original model state dict ---")
    for k, v in autoregressive.state_dict().items():
        print(k, v.shape)
    config = TTTSGPT2Config(
        max_mel_tokens=604,
        max_text_tokens=402,
        max_conditioning_inputs=2,
        layers=30,
        model_dim=1024,
        heads=16,
        checkpointing=False,
    )
    # register the model
    AutoConfig.register("tttsgpt2", TTTSGPT2Config)
    AutoModelForCausalLM.register(TTTSGPT2Config, TTTSGPT2Model)
    config.auto_map = {
        "AutoConfig": "autoregressive.TTTSGPT2Config",
        "AutoModelForCausalLM": "autoregressive.TTTSGPT2Model",
    }
    config.save_pretrained(model_save_dir)
    create_soft_link(
        os.path.abspath("autoregressive.py"),
        os.path.join(model_save_dir, "autoregressive.py"),
    )
    model = TTTSGPT2Model(config)
    print("--- converted model state dict ---")
    for k, v in model.state_dict().items():
        print(k, v.shape)
    print("--- converting weights ---")
    state_dict = convert_weight(autoregressive)
    print("--- converted weights ---")
    for k, v in state_dict.items():
        print(k, v.shape)
    model.load_state_dict(state_dict)
    model.save_pretrained(model_save_dir)
    print(f"--- converted model saved to {model_save_dir} ---")
    # save generation config
    start_mel_token = 8192
    stop_mel_token = 8193
    num_return_sequences = 1  # default 16
    max_length = 512
    hf_generate_kwargs = {
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.8,
        "length_penalty": 1.0,
        "repetition_penalty": 2.0,
        "bos_token_id": start_mel_token,
        "pad_token_id": stop_mel_token,
        "eos_token_id": stop_mel_token,
        "max_length": max_length,
        "num_return_sequences": num_return_sequences,
        "output_hidden_states": True,
        "return_dict_in_generate": True,
    }
    generation_config = GenerationConfig.from_pretrained(
        model_save_dir, **hf_generate_kwargs
    )
    generation_config.save_pretrained(model_save_dir)
    print(f"--- saved model and generation config to {model_save_dir}---")


if __name__ == "__main__":
    main()
