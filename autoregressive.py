import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map


class TTTSGPT2Config(PretrainedConfig):
    model_type = "tttsgpt2"

    def __init__(
        self,
        layers=30,
        model_dim=1024,
        heads=16,
        max_mel_tokens=604,
        max_conditioning_inputs=2,
        max_text_tokens=402,
        checkpointing=False,
        number_mel_codes=8194,
        kv_cache=True,
        start_text_token=255,
        stop_text_token=0,
        number_text_tokens=255,
        start_mel_token=8192,
        **kwargs,
    ):
        self.kv_cache = kv_cache
        self.layers = layers
        self.model_dim = model_dim
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_conditioning_inputs = max_conditioning_inputs
        self.max_text_tokens = max_text_tokens
        self.checkpointing = checkpointing
        self.number_mel_codes = number_mel_codes
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_text_tokens = number_text_tokens
        self.start_mel_token = start_mel_token
        super().__init__(**kwargs)


def null_position_embeddings(range, dim):
    return torch.zeros(
        (range.shape[0], range.shape[1], dim), device=range.device, dtype=range.dtype
    )


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        # NOTE: for TGI, sl may be longer than seq_len.
        max_seq_len = self.emb.weight.shape[0]
        if sl <= max_seq_len:
            return self.emb(torch.arange(0, sl, device=x.device))
        else:
            out = self.emb(torch.arange(0, max_seq_len, device=x.device))
            # keep the length same
            return F.pad(out, (0, 0, 0, sl - max_seq_len))

    def get_fixed_embedding(self, ind, dev):
        # Ensure ind is a tensor, regardless of the input type
        ind_tensor = ind if isinstance(ind, torch.Tensor) else torch.tensor([ind])
        # Move the tensor to the specified device
        ind_tensor = ind_tensor.to(dev)
        # Get the embedding and add a dimension
        return self.emb(ind_tensor).unsqueeze(0)


def build_hf_gpt_transformer(
    layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing
):
    """
    GPT-2 implemented by the HuggingFace library.
    """
    gpt_config = GPT2Config(
        vocab_size=256,  # Unused.
        n_positions=max_mel_seq_len + max_text_seq_len,
        n_ctx=max_mel_seq_len + max_text_seq_len,
        n_embd=model_dim,
        n_layer=layers,
        n_head=heads,
        gradient_checkpointing=checkpointing,
        use_cache=not checkpointing,
    )
    print("gpt_config", gpt_config)
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte
    return (
        gpt,
        LearnedPositionEmbeddings(max_mel_seq_len, model_dim),
        LearnedPositionEmbeddings(max_text_seq_len, model_dim),
        None,
        None,
    )


class TTTSGPT2Model(GPT2PreTrainedModel):
    config_class = TTTSGPT2Config

    def __init__(self, config: TTTSGPT2Config):
        # udpate config
        config.vocab_size = config.max_mel_tokens
        config.n_positions = config.max_mel_tokens + config.max_text_tokens + 2
        config.n_ctx = config.n_positions
        super().__init__(config)
        (
            self.transformer,
            self.text_pos_embedding,  # means mel_pos_embedding
            self.real_text_pos_embedding,  # means text_pos_embedding
            self.mel_layer_pos_embedding,  # None
            self.text_layer_pos_embedding,  # None
        ) = build_hf_gpt_transformer(
            config.layers,
            config.model_dim,
            config.heads,
            config.max_mel_tokens + 2 + config.max_conditioning_inputs,
            config.max_text_tokens + 2,
            config.checkpointing,
        )

        self.embeddings = nn.Embedding(config.number_mel_codes, config.model_dim)
        self.register_buffer(
            "speech_conditioning_latent",
            torch.rand(1, config.model_dim),
        )
        self.transformer.wte = self.embeddings
        self.text_embedding = nn.Embedding(
            config.number_text_tokens + 1, config.model_dim
        )

        final_norm = nn.LayerNorm(config.model_dim)
        mel_head = nn.Linear(config.model_dim, config.number_mel_codes)
        self.lm_head = nn.Sequential(final_norm, mel_head)
        self.kv_cache = config.kv_cache
        self.start_text_token = config.start_text_token
        self.stop_text_token = config.stop_text_token
        self.start_mel_token = config.start_mel_token

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.cached_mel_emb = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(
                len(self.transformer.h), range(max(1, torch.cuda.device_count()))
            )
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def store_mel_emb(self, mel_emb):
        self.cached_mel_emb = mel_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def prefill_forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        speech_conditioning_latent=None,
    ):
        input_ids = input_ids[
            :, 4:
        ]  # remove the first 4 tokens, which are used for position_ids
        input_ids = F.pad(input_ids, (0, 1), value=self.stop_text_token)
        input_ids, _ = self.build_aligned_inputs_and_targets(
            input_ids, self.start_text_token, self.stop_text_token
        )
        text_emb = self.text_embedding(input_ids) + self.real_text_pos_embedding(
            input_ids
        )
        conds = speech_conditioning_latent.unsqueeze(1)
        if conds.shape[0] != text_emb.shape[0]:
            conds = conds.repeat_interleave(text_emb.shape[0] // conds.shape[0], 0)
        mel_emb = torch.cat([conds, text_emb], dim=1)  # cached_mel_emb
        # ---------------------------------------------------------
        start_mel_token = torch.zeros_like(input_ids[:, -1:]) + self.start_mel_token
        text_emb = self.embeddings(start_mel_token)
        text_emb = text_emb + self.text_pos_embedding(text_emb)
        emb = torch.cat([mel_emb, text_emb], dim=1)
        return self.gpt_forward(
            emb,
            past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

    def decode_forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        input_length=None
    ):
        emb = self.embeddings(input_ids)
        position = attention_mask.shape[1] - (input_length - 1)  # shape: [1,]
        emb = emb + self.text_pos_embedding.get_fixed_embedding(
            position, position.device  # position is a tensor
        )
        return self.gpt_forward(
            emb,
            past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

    def gpt_forward(
        self,
        emb=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            if torch.backends.mps.is_available():
                self.to(self.transformer.first_device)
            else:
                torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        out_logits = self.lm_head[0](hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=out_logits,  # transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # assert self.cached_mel_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # Create embedding
        if input_ids.shape[1] != 1:
            input_ids = input_ids[
                :, 4:
            ]  # remove the first 4 tokens, which are used for position_ids
            input_ids = F.pad(input_ids, (0, 1), value=self.stop_text_token)
            input_ids, _ = self.build_aligned_inputs_and_targets(
                input_ids, self.start_text_token, self.stop_text_token
            )
            text_emb = self.text_embedding(input_ids) + self.real_text_pos_embedding(
                input_ids
            )
            conds = self.speech_conditioning_latent.unsqueeze(1)
            if conds.shape[0] != text_emb.shape[0]:
                conds = conds.repeat_interleave(text_emb.shape[0] // conds.shape[0], 0)
            mel_emb = torch.cat([conds, text_emb], dim=1)  # cached_mel_emb
            self.mel_len = mel_emb.shape[1]
            # ---------------------------------------------------------
            start_mel_token = torch.zeros_like(input_ids[:, -1:]) + self.start_mel_token
            text_emb = self.embeddings(start_mel_token)
            text_emb = text_emb + self.text_pos_embedding(text_emb)
            emb = torch.cat([mel_emb, text_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.text_pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - self.mel_len, attention_mask.device
            )

        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            if torch.backends.mps.is_available():
                self.to(self.transformer.first_device)
            else:
                torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        out_logits = self.lm_head[0](hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=out_logits,  # transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )
