set -e

# how to install requirements see tortoise-tts

# we need the original source code
git clone git@github.com:neonbjb/tortoise-tts.git

# set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/tortoise-tts/

# extract GPT2 and save as AutoModel. default save path: tttsgpt2
python convert.py

# [GGUF] Convert AutoModel to GGUF for tortoise.cpp inference (optional)
python convert_hf_to_gguf.py --model_save_dir tttsgpt2 --fname_out ggml-model.bin

# make sure AutoModel output is the same to the original
python test.py 

# export pytorch model to onnx models: prefill and decode
python export_onnx.py

# inference onnx models, the generated latents will be saved in hidden_states.1xTx1024.float32.bin
python modeling_tttsgpt2.py

# rename the generated latents
mv hidden_states.1x*.float32.bin latents.1xTx1024.float32.bin

# generate wav file
python infer_diffusion_vocoder.py

echo "generated wav file diffusion_vocoder.wav"
