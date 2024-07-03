from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer
from typing import Optional, List, Union, Tuple


class TTTSTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        # 简单的tokenizer没有真正的vocab，所以返回一个虚拟的词汇表大小
        return 0

    def get_vocab(self):
        # 返回一个空的词汇表，因为我们不需要它
        return {}

    def _tokenize(self, text, **kwargs):
        # 按空格分割文本
        return text.split()

    def _convert_token_to_id(self, token):
        try:
            return int(token)
        except ValueError:
            return 0

    def _convert_id_to_token(self, index):
        # 将整数ID转换回字符串
        return str(index)

    def convert_tokens_to_string(self, tokens):
        # 将token列表转换为字符串
        return " ".join(tokens)

    def _decode(self, token_ids: Union[int, List[int]], **kwargs):
        # 将整数序列解码回字符串
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]
        return self.convert_tokens_to_string(tokens)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """
        保存词表到指定目录。由于TTTSTokenizer没有实际的词表，因此这是一个空实现。

        Args:
            save_directory (`str`): 词表文件将被保存到的目录。
            filename_prefix (`str`, *optional*): 如果提供，将被添加到词表文件名的开头。

        Returns:
            `Tuple(str)`: 保存的词表文件的路径。
        """
        return ()


def test_save():
    # 测试TTTSTokenizer
    tokenizer = TTTSTokenizer()
    tokenizer.save_pretrained("TTTSTokenizer")


def test_load():
    tokenizer = AutoTokenizer.from_pretrained("TTTSTokenizer", trust_remote_code=True)

    encoded_input = tokenizer("12 34 554")
    print(encoded_input)  # 输出: [12, 34, 554]

    decoded_output = tokenizer.decode(encoded_input["input_ids"])
    print("---", decoded_output)  # 输出: "12 34 554"


if __name__ == "__main__":
    test_save()
    test_load()
