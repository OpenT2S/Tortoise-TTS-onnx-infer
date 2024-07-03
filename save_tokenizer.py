import torch
import torch.nn.functional as F
from tortoise.utils.tokenizer import VoiceBpeTokenizer as TortoiseVoiceBpeTokenizer
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, normalizers, Regex, pre_tokenizers

save_path = "tttsgpt2"

base = TortoiseVoiceBpeTokenizer(
    vocab_file=None,
    use_basic_cleaners=None,
)

text = "I'm going to speak this"

text_tokens = torch.IntTensor(base.encode(text)).unsqueeze(0)
text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
print(text_tokens)

'''
tokenizer = Tokenizer.from_file("../tortoise/data/tokenizer.json")
tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.NFC(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(" {2,}"), " "),
        normalizers.Replace('"', ""),
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.Replace(" ", "[SPACE]"),
    ]
)

tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    stop_token="[STOP]",
    space_token="[SPACE]",
)

wrapped_tokenizer.save_pretrained(save_path)


out = wrapped_tokenizer(text)
print(out)
'''