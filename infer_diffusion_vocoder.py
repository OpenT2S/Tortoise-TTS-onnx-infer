import numpy as np
import torch
from diffusion_decoder import DiffusionTts
from vocoder import UnivNetGenerator
from convert import get_model_path, MODELS_DIR
from diffusion import (
    SpacedDiffusion,
    space_timesteps,
    get_named_beta_schedule,
)
import torchaudio

# -------------------------- helper functions --------------------------


def load_discrete_vocoder_diffuser(
    trained_diffusion_steps=4000,
    desired_diffusion_steps=200,
    cond_free=True,
    cond_free_k=1,
):
    """
    Helper function to load a GaussianDiffusion instance configured for use as a vocoder.
    """
    return SpacedDiffusion(
        use_timesteps=space_timesteps(
            trained_diffusion_steps, [desired_diffusion_steps]
        ),
        model_mean_type="epsilon",
        model_var_type="learned_range",
        loss_type="mse",
        betas=get_named_beta_schedule("linear", trained_diffusion_steps),
        conditioning_free=cond_free,
        conditioning_free_k=cond_free_k,
    )


def denormalize_tacotron_mel(norm_mel):
    TACOTRON_MEL_MAX = 2.3143386840820312
    TACOTRON_MEL_MIN = -11.512925148010254
    return ((norm_mel + 1) / 2) * (
        TACOTRON_MEL_MAX - TACOTRON_MEL_MIN
    ) + TACOTRON_MEL_MIN


def do_spectrogram_diffusion(
    diffusion_model,
    diffuser,
    latents,
    conditioning_latents,
    temperature=1,
    verbose=True,
):
    """
    Uses the specified diffusion model to convert discrete codes into a spectrogram.
    """
    with torch.no_grad():
        output_seq_len = (
            latents.shape[1] * 4 * 24000 // 22050
        )  # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        output_shape = (latents.shape[0], 100, output_seq_len)
        precomputed_embeddings = diffusion_model.timestep_independent(
            latents, conditioning_latents, output_seq_len, False
        )

        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.p_sample_loop(
            diffusion_model,
            output_shape,
            noise=noise,
            model_kwargs={"precomputed_aligned_embeddings": precomputed_embeddings},
            progress=verbose,
        )
        return denormalize_tacotron_mel(mel)[:, :, :output_seq_len]


# ---------------------------- init models ----------------------------
diffusion = (
    DiffusionTts(
        model_channels=1024,
        num_layers=10,
        in_channels=100,
        out_channels=200,
        in_latent_channels=1024,
        in_tokens=8193,
        dropout=0,
        use_fp16=False,
        num_heads=16,
        layer_drop=0,
        unconditioned_percentage=0,
    )
    .cuda()
    .eval()
)
diffusion.load_state_dict(
    torch.load(get_model_path("diffusion_decoder.pth", MODELS_DIR))
)

vocoder = UnivNetGenerator().cuda()
vocoder.load_state_dict(
    torch.load(
        get_model_path("vocoder.pth", MODELS_DIR), map_location=torch.device("cpu")
    )["model_g"]
)
vocoder.eval(inference=True)

# ----------------------------- inference -----------------------------
diffusion_iterations = 80
cond_free = True
cond_free_k = 2.0
diffusion_temperature = 1.0
verbose = True

diffuser = load_discrete_vocoder_diffuser(
    desired_diffusion_steps=diffusion_iterations,
    cond_free=cond_free,
    cond_free_k=cond_free_k,
)

"""
saved to file via pdb: 
!latents.cpu().data.numpy().tofile("latents.1xTx1024.float32.bin")
!diffusion_conditioning.cpu().data.numpy().tofile("diffusion_conditioning.1x2048.float32.bin")
"""
latents = np.fromfile("latents.1xTx1024.float32.bin", dtype=np.float32).reshape(
    1, -1, 1024
)  # shape: [1, T, 1024], dtype: torch.float32
diffusion_conditioning = np.fromfile(
    "diffusion_conditioning.1x2048.float32.bin", dtype=np.float32
).reshape(
    1, 2048
)  # shape: [1, 2048], dtype: torch.float32
latents = torch.from_numpy(latents).cuda()
diffusion_conditioning = torch.from_numpy(diffusion_conditioning).cuda()
print(
    f"--- latents shape: {latents.shape}, diffusion_conditioning shape: {diffusion_conditioning.shape}"
)

mel = do_spectrogram_diffusion(
    diffusion,
    diffuser,
    latents,
    diffusion_conditioning,
    temperature=diffusion_temperature,
    verbose=verbose,
)
wav = vocoder.inference(mel)  # shape: [1, 1, T], dtype: torch.float32, range: -1 ~ 1
torchaudio.save(
    "diffusion_vocoder.wav",
    wav.squeeze(0).cpu(),
    24000,
)
