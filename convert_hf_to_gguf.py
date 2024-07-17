import fire
import numpy as np
import struct
from transformers import AutoConfig, AutoModelForCausalLM


def rename(name):
    if name.startswith("transformer.") or name.startswith("lm_head."):
        name = "inference_model." + name
    elif name == "embeddings.weight":
        name = "mel_embedding.weight"
    elif name == "text_pos_embedding.emb.weight":
        name = "mel_pos_embedding.emb.weight"
    elif name == "real_text_pos_embedding.emb.weight":
        name = "text_pos_embedding.emb.weight"
    return name


def should_skip(name):
    if name in [
        "lm_head.weight",
        "transformer.wte.weight",
        "speech_conditioning_latent",
    ]:
        return True
    return False


def main(
    model_save_dir: str = "./", fname_out: str = "ggml-model.bin", use_f16: bool = False
):
    model = AutoModelForCausalLM.from_pretrained(
        model_save_dir, trust_remote_code=True
    ).eval()

    list_vars = model.state_dict()
    fout = open(fname_out, "wb")
    fout.write(struct.pack("i", 0x67676D6C))  # magic: ggml in hex
    for name in list_vars.keys():
        # skip some variables
        if should_skip(name):
            continue
        data = list_vars[name].squeeze().numpy()
        print("Processing variable: " + name + " with shape: ", data.shape)
        n_dims = len(data.shape)

        ftype = 0  # default float32
        if use_f16:
            if name[-7:] == ".weight" and n_dims == 2:
                print("  Converting to float16")
                data = data.astype(np.float16)
                ftype = 1  # use float16
            else:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype = 0

        # rename headers to keep compatibility
        name = rename(name)

        str = name.encode("utf-8")
        fout.write(struct.pack("iii", n_dims, len(str), ftype))
        if not n_dims in [1, 2]:
            raise Exception(
                "Error: unsupported shape: %s, name: %s" % (data.shape, name)
            )
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)
        # data
        data.tofile(fout)
    fout.close()
    print("Done. Output file: " + fname_out)


if __name__ == "__main__":
    fire.Fire(main)

