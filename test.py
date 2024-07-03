import time
import numpy as np
import torch

from autoregressive import TTTSGPT2Model
from transformers import AutoConfig, AutoModelForCausalLM

model_save_dir = "tttsgpt2"
start_mel_token = 8192
stop_mel_token = 8193
max_mel_tokens = 604
max_generate_length = 500
num_return_sequences = 16

num_infer_iter = 100

# model = TTTSGPT2Model.from_pretrained(model_save_dir).cuda().eval()
model = (
    AutoModelForCausalLM.from_pretrained(model_save_dir, trust_remote_code=True)
    .cuda()
    .eval()
)

# saved from pdb using:
# emb.cpu().numpy().tofile("emb.1x18x1024.float32.bin")
emb = np.fromfile("emb.1x18x1024.float32.bin", dtype=np.float32).reshape(1, 18, 1024)
emb = torch.from_numpy(emb).cuda()
condition = emb[:, 0, :]  # shape (1, 1024)

model.store_mel_emb(None)

fake_inputs = torch.full(
    (
        emb.shape[0],
        1 + emb.shape[1],
    ),
    fill_value=1,
    dtype=torch.long,
)
fake_inputs[:, -1] = start_mel_token
trunc_index = fake_inputs.shape[1]
inputs = fake_inputs.cuda()
max_length = (
    trunc_index + max_mel_tokens - 1
    if max_generate_length is None
    else trunc_index + max_generate_length
)

inputs_ids = torch.tensor(
    [
        [0, 0, 0, 0, 22, 4, 26, 2, 119, 52, 2, 51, 2, 32, 124, 213, 2, 147, 0]
    ],  # the first 0s, will be removed
    dtype=torch.long,
).cuda()
assert inputs_ids.shape[1] == inputs.shape[1]

hf_generate_kwargs = {
    "do_sample": True,
    "top_p": 0.8,
    "temperature": 0.8,
    "length_penalty": 1.0,
    "repetition_penalty": 2.0,
}

# ------------------ Test-ONLY Config ------------------
hf_generate_kwargs["do_sample"] = False  # Only for testing
num_return_sequences = 1
model.generation_config.do_sample = hf_generate_kwargs["do_sample"]
model.generation_config.num_return_sequences = num_return_sequences
# ------------------ Test-ONLY Config ------------------

total_time_ns = 0
squared_time_diffs = 0  # for calculating standard deviation
# Warmup iteration (not included in measurements)
model.generate(inputs_ids)

for _ in range(num_infer_iter):
    try:
        time_start = time.time_ns()
        gen = model.generate(inputs_ids)
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

print(f"--- Average inference time (AutoModel): {average_time_ms:.3f} ms")
print(f"--- Standard deviation: {std_dev_ms:.3f} ms")

print(gen.keys())  # ['sequences', 'hidden_states', 'past_key_values']

sequences = gen["sequences"]
hidden_states = gen["hidden_states"]
hidden_states = torch.cat([hs[:, -1:] for hs in hidden_states], dim=1)
gen = sequences
print("--- sequences", sequences.shape)
print("--- hidden_states", hidden_states.shape)
hidden_states.cpu().data.numpy().tofile("hidden_states.float32.bin")

out = gen[:, trunc_index:]
print(out.shape)
print(out)
del model

# ------------------- original model infer result -------------------
from tortoise.models.autoregressive import UnifiedVoice
from convert import get_model_path, MODELS_DIR

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
    .cuda()
    .eval()
)
autoregressive.load_state_dict(
    torch.load(get_model_path("autoregressive.pth", MODELS_DIR)), strict=False
)
autoregressive.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)
print("Model loaded.")

autoregressive.inference_model.store_mel_emb(emb)

total_time_ns = 0
squared_time_diffs = 0  # for calculating standard deviation
# Warmup iteration (not included in measurements)
gen2 = autoregressive.inference_model.generate(
    inputs,
    bos_token_id=start_mel_token,
    pad_token_id=stop_mel_token,
    eos_token_id=stop_mel_token,
    max_length=max_length,
    num_return_sequences=num_return_sequences,
    **hf_generate_kwargs,
)

for _ in range(num_infer_iter):
    try:
        time_start = time.time_ns()
        gen2 = autoregressive.inference_model.generate(
            inputs,
            bos_token_id=start_mel_token,
            pad_token_id=stop_mel_token,
            eos_token_id=stop_mel_token,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            **hf_generate_kwargs,
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
print(f"--- Average inference time (Tortoise-TTS): {average_time_ms:.3f} ms")
print(f"--- Standard deviation: {std_dev_ms:.3f} ms")

out2 = gen2[:, trunc_index:]
print(out2.shape)
print(out2)
print("--- is equal:", out == out2)
