"""
TODO list

ContextWhisperModel
- [x] config
- [x] abstract class
- [x] the model

ContextWhisperEncoder (BERT wrapper)
- [x] layer
- [x] the module

ContextWhisperSpectrogramEncoder
- [x] layer
- [x] the module

ContextWhisperEncoder
- [x] the module # no layer needed since this is a composition of Text and Spectrogram parts

ContextWhisperDecoder
- [x] layer
- [x] the module # only change minor things, such as the config class

"""

import math
from typing import Any, Literal, Optional, Tuple, Union

import torch
from torch import nn
from transformers import GenerationMixin
from transformers.configuration_utils import PretrainedConfig
from transformers.models.bert.modeling_bert import BertConfig, BertModel
from transformers.models.whisper.modeling_whisper import (
    ACT2FN, WHISPER_ATTENTION_CLASSES, BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions, EncoderDecoderCache, Seq2SeqModelOutput,
    WhisperConfig, WhisperDecoder, WhisperDecoderLayer, WhisperEncoder,
    WhisperEncoderLayer, WhisperForCausalLM, WhisperModel,
    WhisperPreTrainedModel, _compute_mask_indices, sinusoids)


class ContextWhisperConfig(PretrainedConfig):
    model_type = "context_whisper"

    def __init__(
        self,
        vocab_size: int = 51865,
        num_mel_bins: int = 80,
        text_encoder_layers: int = 4,
        text_encoder_attention_heads: int = 6,
        text_encoder_ffn_dim: int = 1536,
        decoder_layers: int = 4,
        decoder_attention_heads: int = 6,
        decoder_ffn_dim: int = 1536,
        spectrogram_encoder_layers: int = 4,
        spectrogram_encoder_attention_heads: int = 6,
        spectrogram_encoder_ffn_dim: int = 1536,
        text_encoder_layerdrop: float = 0.0,
        decoder_layerdrop: float = 0.0,
        spectrogram_encoder_layerdrop: float = 0.0,
        decoder_start_token_id: int = 50257,
        use_cache: bool = True,
        is_encoder_decoder: bool = True,  # TODO: I think so? It's just that the encoder is a bit complicated
        activation_function: str = "gelu",
        d_model: int = 384,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        init_std: float = 0.02,
        scale_embedding: bool = False,
        max_source_positions: int = 1500,
        max_target_positions: int = 448,
        pad_token_id: int = 50256,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        suppress_tokens: Optional[list[int]] = None,
        begin_suppress_tokens: Optional[list[int]] = [220, 50256],
        use_weighted_layer_sum: bool = False,
        classifier_proj_size: int = 256,
        apply_spec_augment: bool = False,
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        mask_time_min_masks: int = 2,
        mask_feature_prob: float = 0.0,
        mask_feature_length: int = 10,
        mask_feature_min_masks: int = 0,
        median_filter_width: int = 7,
        text_encoder_pretrained_str: Optional[str] = None,
        decoder_pretrained_str: Optional[str] = None,
        spectrogram_encoder_pretrained_str: Optional[str] = None,
        whisper_pretrained_str: Optional[
            str
        ] = None,  # if specified, takes prescendence over {decoder,spectrogram_encoder}pretrained_str
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.text_encoder_layers = text_encoder_layers
        self.text_encoder_attention_heads = text_encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.spectrogram_encoder_layers = spectrogram_encoder_layers
        self.spectrogram_encoder_attention_heads = spectrogram_encoder_attention_heads
        self.text_encoder_ffn_dim = text_encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.spectrogram_encoder_ffn_dim = spectrogram_encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.text_encoder_layerdrop = text_encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.spectrogram_encoder_layerdrop = spectrogram_encoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = text_encoder_layers
        self.scale_embedding = (
            scale_embedding  # scale factor will be sqrt(d_model) if True
        )
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions

        # Audio Classification-specific parameters. Feel free to ignore for other classes.
        self.classifier_proj_size = classifier_proj_size
        self.use_weighted_layer_sum = use_weighted_layer_sum

        # fine-tuning config parameters for SpecAugment: https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks
        self.median_filter_width = median_filter_width

        # pretrained args:
        self.text_encoder_pretrained_str = text_encoder_pretrained_str
        self.decoder_pretrained_str = (
            decoder_pretrained_str
            if whisper_pretrained_str is None
            else whisper_pretrained_str
        )
        self.spectrogram_encoder_pretrained_str = (
            spectrogram_encoder_pretrained_str
            if whisper_pretrained_str is None
            else whisper_pretrained_str
        )
        self.whisper_pretrained_str = whisper_pretrained_str

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            **kwargs,
        )


class ContextWhisperPreTrainedModel(WhisperPreTrainedModel):
    config_class = ContextWhisperConfig
    _no_split_modules = [
        "ContextWhisperEncoderLayer",
        "ContextWhisperSpectrogramEncoderLayer",
        "ContextWhisperDecoderLayer",
    ]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, ContextWhisperSpectrogramEncoder):
            with torch.no_grad():
                embed_positions = module.embed_positions.weight
                embed_positions.copy_(sinusoids(*embed_positions.shape))

    def _freeze_parameters(
        self,
    ):  # this is defined in ContextWhisperPreTrainedModel. Not sure why it is not properly inherited. TODO
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False


class ContextWhisperEncoderLayer(WhisperEncoderLayer):
    pass  # this is the same and we don't need to change anything (because the layer already works with embeddings and not spectrograms)


class ContextWhisperDecoderLayer(WhisperDecoderLayer):
    pass  # This is the same and we do not need to change anything


class ContextWhisperSpectrogramEncoderLayer(nn.Module):
    """
    Very similar to WhisperDecoderLayer except that it is not causal so we can drop cache functionality
    """

    def __init__(self, config: ContextWhisperConfig, layer_idx: int = None):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=False,  # NOTE: important difference to WhisperDecoderLayer
            layer_idx=layer_idx,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            layer_idx=layer_idx,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    @classmethod
    def from_whisper_layer(
        cls,
        from_layer: WhisperEncoderLayer,
        layer_idx: int,
        config: ContextWhisperConfig,
    ) -> "ContextWhisperSpectrogramEncoderLayer":
        """Take all attributes possible from layer, and add other attributes where required"""
        layer = cls(config, layer_idx=layer_idx)
        layer.embed_dim = from_layer.embed_dim
        layer.self_attn = from_layer.self_attn
        layer.dropout = from_layer.dropout
        layer.activation_fn = from_layer.activation_fn
        layer.activation_dropout = from_layer.activation_dropout
        layer.self_attn_layer_norm = from_layer.self_attn_layer_norm
        layer.fc1 = from_layer.fc1
        layer.fc2 = from_layer.fc2
        layer.final_layer_norm = from_layer.final_layer_norm
        return layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            # past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            # cache_position=cache_position,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            (
                hidden_states,
                cross_attn_weights,
                cross_attn_present_key_value,
            ) = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                # past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states

            # add cross-attn to positions 1 of present_key_value tuple
            present_key_value = (present_key_value, cross_attn_present_key_value)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # if use_cache:
        #     outputs += (present_key_value,)

        return outputs


class ContextWhisperTextEncoder(BertModel, ContextWhisperPreTrainedModel):  # TODO
    """This is a wrapper for the BERT encoder"""

    def __init__(self, config: ContextWhisperConfig) -> None:
        if config.text_encoder_pretrained_str is None:
            bert_kwargs = ContextWhisperTextEncoder.strip_config_for_bert(config)
            bert_cfg = BertConfig(
                # vocab_size=, # let BERT  decide
                hidden_size=config.d_model,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.text_encoder_attention_heads,
                intermediate_size=config.text_encoder_ffn_dim,
                hidden_act=config.activation_function,
                hidden_dropout_prob=config.dropout,
                attention_probs_dropout_prob=config.attention_dropout,
                max_position_embeddings=config.max_source_positions,
                # type_vocab_size=, # TODO: ???
                # initializer_range=, # TODO: ???
                # layer_norm_eps=,
                # pad_token_id=,
                # position_embedding_type=,
                use_cache=config.use_cache,
                # classifier_dropout=,
                **bert_kwargs,
            )
        else:
            bert_cfg = BertConfig.from_pretrained(config.text_encoder_pretrained_str)
        super().__init__(config=bert_cfg)

    @classmethod
    def strip_config_for_bert(cls, config: ContextWhisperConfig) -> dict[str, Any]:
        drop_attrs = set(
            [
                "vocab_size",
                "num_mel_bins",
                "d_model",
                "text_encoder_layers",
                "text_encoder_attention_heads",
                "decoder_layers",
                "decoder_attention_heads",
                "spectrogram_encoder_layers",
                "spectrogram_encoder_attention_heads",
                "text_encoder_ffn_dim",
                "decoder_ffn_dim",
                "spectrogram_encoder_ffn_dim",
                "dropout",
                "attention_dropout",
                "activation_dropout",
                "activation_function",
                "init_std",
                "text_encoder_layerdrop",
                "decoder_layerdrop",
                "spectrogram_encoder_layerdrop",
                "use_cache",
                "num_hidden_layers",
                "scale_embedding",
                "max_source_positions",
                "max_target_positions",
                "classifier_proj_size",
                "use_weighted_layer_sum",
                "apply_spec_augment",
                "mask_time_prob",
                "mask_time_length",
                "mask_time_min_masks",
                "mask_feature_prob",
                "mask_feature_length",
                "mask_feature_min_masks",
                "median_filter_width",
                "pad_token_id",
                "bos_token_id",
                "eos_token_id",
                "is_encoder_decoder",
                "decoder_start_token_id",
                "suppress_tokens",
                "begin_suppress_tokens",
            ]
        )
        keep_attrs = set(
            [
                # 'pad_token_id',
                # 'bos_token_id',
                # 'eos_token_id',
                # 'is_encoder_decoder',
                # 'decoder_start_token_id',
                # 'suppress_tokens',
                # 'begin_suppress_tokens',
            ]
        )  # passed to super in ContextWhisperConfig
        return {
            k: v
            for k, v in config.to_dict().items()
            if k not in drop_attrs - keep_attrs
        }


class ContextWhisperSpectrogramEncoder(ContextWhisperPreTrainedModel):
    """
    This takes a spectrogram as input and uses cross attention to get information from the user context.
    The context embeddings are provided by a ContextWhisperTextEncoder.

    It is very similar to the implementation of WhisperEconder.
    """

    def __init__(
        self,
        config: ContextWhisperConfig,
    ) -> None:
        if config.spectrogram_encoder_pretrained_str is None:
            super().__init__(config)
            self.dropout = config.dropout
            self.layerdrop = config.spectrogram_encoder_layerdrop

            embed_dim = config.d_model
            self.num_mel_bins = config.num_mel_bins
            self.padding_idx = config.pad_token_id
            self.max_source_positions = config.max_source_positions
            self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

            self.conv1 = nn.Conv1d(
                self.num_mel_bins, embed_dim, kernel_size=3, padding=1
            )
            self.conv2 = nn.Conv1d(
                embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
            )

            self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
            self.embed_positions.requires_grad_(False)

            self.layers = nn.ModuleList(
                [
                    ContextWhisperSpectrogramEncoderLayer(
                        config, layer_idx
                    )  # Note: in the original encoder, this is a WhisperEncoderLayer
                    for layer_idx in range(config.spectrogram_encoder_layers)
                ]
            )
            self.layer_norm = nn.LayerNorm(config.d_model)

            self.gradient_checkpointing = False
            # Initialize weights and apply final processing
            self.post_init()
        else:
            pt_config = WhisperConfig.from_pretrained(
                config.spectrogram_encoder_pretrained_str
            )
            pt_config = ContextWhisperConfig(**pt_config.to_dict())
            self.__init__(pt_config)
            # new_layers = nn.ModuleList(
            #     [
            #         ContextWhisperSpectrogramEncoderLayer.from_whisper_layer(
            #             from_layer=from_layer,
            #             layer_idx=layer_idx,
            #             config=config
            #         ) # Note: in the original encoder, this is a WhisperEncoderLayer
            #         for layer_idx, from_layer in enumerate(self.layers)
            #     ]
            # )
            # self.layers = new_layers
            self.config = config

    @classmethod
    def from_whisper(
        cls, whisper_encoder: WhisperEncoder, config: ContextWhisperConfig
    ) -> "ContextWhisperSpectrogramEncoder":
        whisper_spec_enc: ContextWhisperSpectrogramEncoder = cls(config)
        whisper_spec_enc.layers = nn.ModuleList(
            [
                ContextWhisperSpectrogramEncoderLayer.from_whisper_layer(
                    layer, idx, config
                )
                for idx, layer in enumerate(whisper_encoder.layers)
            ]
        )
        whisper_spec_enc.conv1 = whisper_encoder.conv1
        whisper_spec_enc.conv2 = whisper_encoder.conv2
        whisper_spec_enc.embed_positions = whisper_encoder.embed_positions
        whisper_spec_enc.layer_norm = whisper_encoder.layer_norm
        return whisper_spec_enc

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(  # This function is more similar to the WhisperDecoder, but the preprocessing is different
        self,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict=None,
    ):
        """
        Args:
        input_features: Optional[torch.Tensor] = None
            Input features from ConextWhisperTextEncoder
        attention_mask: Optional[torch.Tensor] = None
            Attention mask for ContextWhisperTextEncoder
        encoder_hidden_states: Optional[torch.Tensor] = None
            Output of ContextWhisperTextEncoder
        head_mask: Optional[torch.Tensor] = None
            head mask for ContextWhisperTextEncoder
        cross_attn_head_mask: Optional[torch.Tensor] = None
            cross attention head mask between ContextWhisperSpectrogramEncoder
            and ContextWhisperTextEncoder
        output_attentions: Optional[bool] = None
            Whether to output attentions
        output_hidden_states: Optional[bool] = None
            Whether to output hidden states
        return_dict=None


        """

        expected_seq_length = (
            self.config.max_source_positions
            * self.conv1.stride[0]
            * self.conv2.stride[0]
        )

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        # input_shape = inputs_embeds.size()[:-1]

        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = (
            self.embed_positions.weight
        )  # TODO: is this the same as positions in WhisperDecoder.forward?

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    None,  # encoder attention mask
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx]
                    if cross_attn_head_mask is not None
                    else None,
                    None,  # past_key_value
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class ContextWhisperDecoder(WhisperDecoder, ContextWhisperPreTrainedModel):
    def __init__(self, config: ContextWhisperConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]
        ] = None,
        inputs_embeds: Optional[Tuple[torch.Tensor]] = None,
        position_ids=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Tuple | BaseModelOutputWithPastAndCrossAttentions:
        # we implement this function just for type annotations
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )


class ContextWhisperEncoder(ContextWhisperPreTrainedModel):
    def __init__(
        self,
        config: ContextWhisperConfig,  # TODO: are you sure?
        text_encoder: ContextWhisperTextEncoder,
        spectrogram_encoder: ContextWhisperSpectrogramEncoder,
    ) -> None:
        super().__init__(config)
        self.text_encoder = text_encoder
        self.spectrogram_encoder = spectrogram_encoder

    def forward(
        self,
        spectrogram_input_features: Optional[torch.FloatTensor] = None,
        spectrogram_attention_mask: Optional[
            torch.LongTensor
        ] = None,  # "spectrogram self attention"
        encoder_cross_attn_head_mask: Optional[torch.Tensor] = None,
        text_encoder_input_ids: Optional[torch.LongTensor] = None,
        text_encoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        text_encoder_attention_mask: Optional[
            torch.LongTensor
        ] = None,  # "spectrogram self attention"
        encoder_head_mask: Optional[torch.Tensor] = None,
        # handle outputting these:
        # encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        text_out = self.text_encoder.forward(
            input_ids=text_encoder_input_ids,
            inputs_embeds=text_encoder_inputs_embeds,
            attention_mask=text_encoder_attention_mask,
            position_ids=None,  # TODO: understand this better
            # head_mask=,
            # encoder_hidden_states=,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        spec_out = self.spectrogram_encoder.forward(
            input_features=spectrogram_input_features,
            attention_mask=spectrogram_attention_mask,
            encoder_hidden_states=text_out.last_hidden_state,  # hidden states in text_out
            head_mask=encoder_head_mask,
            cross_attn_head_mask=encoder_cross_attn_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return spec_out
        # # TODO: format outputs based on output_attentions etc
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # if not return_dict:
        #     pass
        # return BaseModelOutputWithPastAndCrossAttentions(
        #     last_hidden_state=spec_out.last_hidden_state,
        #     hidden_states=spec_out.hidden_states,
        #     past_key_values=spec_out.past_key_values
        # )


class ContextWhisperModel(ContextWhisperPreTrainedModel):
    def __init__(self, config: ContextWhisperConfig):
        self.config = config
        super().__init__(config)

        if config.text_encoder_pretrained_str is None:
            self.text_encoder = ContextWhisperTextEncoder(config)
        else:
            assert (
                "google-bert/" in config.text_encoder_pretrained_str
            ), "Required in this implementation"
            self.text_encoder = BertModel.from_pretrained(
                config.text_encoder_pretrained_str
            )
        if config.spectrogram_encoder_pretrained_str is None:
            self.spectrogram_encoder = ContextWhisperSpectrogramEncoder(config)
        else:
            spectrogram_encoder = WhisperModel.from_pretrained(
                config.spectrogram_encoder_pretrained_str
            ).get_encoder()
            self.spectrogram_encoder = ContextWhisperSpectrogramEncoder.from_whisper(
                spectrogram_encoder, config
            )
            config = self.spectrogram_encoder.config
        self.encoder = ContextWhisperEncoder(
            config=config,
            text_encoder=self.text_encoder,
            spectrogram_encoder=self.spectrogram_encoder,
        )
        # in the case where we load from pretrained, there will be a discrepancy, and we have to reload:
        # self.text_encoder = self.encoder.text_encoder
        # self.spectrogram_encoder = self.encoder.spectrogram_encoder
        if config.decoder_pretrained_str is None:
            self.decoder = ContextWhisperDecoder(config)
        else:
            self.decoder = WhisperModel.from_pretrained(
                config.decoder_pretrained_str
            ).get_decoder()
        # Initialize weights and apply final processing
        self.post_init()  # from transformers - no worries

    def get_input_embeddings(self, which: str = "decoder"):
        if which == "decoder":
            return self.decoder.embed_tokens
        elif which == "text_encoder":
            return self.text_encoder.embed_tokens  # TODO
        else:
            raise ValueError(f"{which=} is not a valid input")

    def set_input_embeddings(self, value: torch.Tensor, which: str = "decoder"):
        if which == "decoder":
            self.decoder.embed_tokens = value
        elif which == "text_encoder":
            self.text_encoder.embed_tokens = value
        else:
            raise ValueError(f"{which=} is not a valid input")

    def get_text_encoder(self):
        return self.text_encoder

    def get_spectrogram_encoder(self):
        return self.spectrogram_encoder

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_module(
        self,
        which: Literal["text_encoder", "decoder", "spectrogram_encoder", "encoder"],
    ) -> None:
        """
        Calling this function will disable the gradient computation for
        the relevant module `which`, so that its parameters will
        not be updated during training.

        In the regular WhisperModel, this function is called `freeze_encoder`,
        and the same effect can be obtained with this function by first
        calling self.freeze_module('text_encoder') and then
        self.freeze_module('spectrogram_encoder')
        """
        getattr(self, which)._freeze_parameters()

    def _mask_input_features(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return input_features

        # generate indices & apply SpecAugment along time axis
        batch_size, hidden_size, sequence_length = input_features.size()

        if self.config.mask_time_prob > 0 and self.training:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(
                mask_time_indices, device=input_features.device, dtype=torch.bool
            )
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features[mask_time_indices] = 0

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(
                mask_feature_indices, device=input_features.device, dtype=torch.bool
            )
            input_features[mask_feature_indices] = 0

        return input_features

    def forward(
        self,
        spectrogram_input_features: Optional[torch.FloatTensor] = None,
        spectrogram_attention_mask: Optional[
            torch.LongTensor
        ] = None,  # "spectrogram self attention"
        encoder_cross_attn_head_mask: Optional[torch.Tensor] = None,
        text_encoder_input_ids: Optional[torch.LongTensor] = None,
        text_encoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        text_encoder_attention_mask: Optional[
            torch.LongTensor
        ] = None,  # "spectrogram self attention"
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[
            Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]
        ] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        r"""
        Args:
        input_features: Optional[torch.FloatTensor] = None
            to the regular WhisperEncoder - it is taken from a WhisperPreprocessor
            TODO: we have to add additional information here for the user prompt
        attention_mask: Optional[torch.LongTensor] = None
            Attention mask for the Encoder. Decoder is named with prefix 'decoder_'
            TODO: might have to distinguish between self attention and cross attention
            in this setup
        decoder_input_ids: Optional[torch.LongTensor] = None
            The ids that have been decoded this far
        decoder_attention_mask: Optional[torch.LongTensor] = None
            Attention mask for the decoder ; PROBABLY has "number of layers" as a dimension
        head_mask: Optional[torch.Tensor] = None
            Attention mask for heads in encoder (decoder is named with 'decoder_'-prefix)
            TODO: may have to distinguish between self and cross attention for the encoder here
        decoder_head_mask: Optional[torch.Tensor] = None
            Attention mask for decoder self attention
        cross_attn_head_mask: Optional[torch.Tensor] = None
            Cross attention mask for the decoder
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
            Output (embeddings) of the encoder
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None
            For caching
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None
            Input embeddings to pass directly to the decoder (i.e. embedding of
            tokenized input sequence)
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None
            ???
        use_cache: Optional[bool] = None
            For caching KQV
        output_attentions: Optional[bool] = None
            Whether to output attentions
        output_hidden_states: Optional[bool] = None
            Whether to output hidden states
        return_dict: Optional[bool] = None
            Controlling whether to return tuple or dict
        cache_position: Optional[torch.LongTensor] = None
            ???

        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import AutoFeatureExtractor, WhisperModel
         >>> from datasets import load_dataset

         >>> model = WhisperModel.from_pretrained("openai/whisper-base")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
         >>> input_features = inputs.input_features
         >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
         >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
         >>> list(last_hidden_state.shape)
         [1, 2, 512]
         ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            spectrogram_input_features = self._mask_input_features(
                spectrogram_input_features, attention_mask=spectrogram_attention_mask
            )

            encoder_outputs = (
                self.encoder(  # TODO: this needs additional inputs due to user prompt
                    spectrogram_input_features=spectrogram_input_features,
                    spectrogram_attention_mask=spectrogram_attention_mask,
                    encoder_cross_attn_head_mask=encoder_cross_attn_head_mask,
                    encoder_head_mask=encoder_head_mask,
                    text_encoder_input_ids=text_encoder_input_ids,
                    text_encoder_inputs_embeds=text_encoder_inputs_embeds,
                    text_encoder_attention_mask=text_encoder_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder.forward(  # This can be left untouched since the decoder does not change
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=decoder_cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class ContextWhisperForCausalLM(ContextWhisperPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["proj_out.weight"]
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        config.is_encoder_decoder = False
        self.model = ContextWhisperModel(config)

        if config.whisper_pretrained_str is None:
            self.proj_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.proj_out = WhisperForCausalLM.from_pretrained(
                config.whisper_pretrained_str
            ).get_output_embeddings()
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.get_decoder()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_text_encoder(self):
        return self.model.get_text_encoder()

    def get_spectrogram_encoder(self):
        return self.model.get_spectrogram_encoder()

    def forward(
        self,
        # the commented features are not used for performance reasons
        # spectrogram_input_features: Optional[torch.FloatTensor] = None,
        # spectrogram_attention_mask: Optional[torch.LongTensor] = None, # "spectrogram self attention"
        # encoder_cross_attn_head_mask: Optional[torch.Tensor] = None,
        # text_encoder_input_ids: Optional[torch.LongTensor] = None,
        # text_encoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        # text_encoder_attention_mask: Optional[torch.LongTensor] = None, # "spectrogram self attention"
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_cross_attn_head_mask: Optional[torch.Tensor] = None,
        # encoder_head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[
            Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]
        ] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # If the user passed a tuple or `BaseModelOutput` for encoder_outputs, we extract only the hidden states
        if isinstance(encoder_outputs, (BaseModelOutput, tuple, list, BaseModelOutputWithPastAndCrossAttentions)):
            encoder_outputs = encoder_outputs[0]

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.get_decoder().forward(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=decoder_cross_attn_head_mask,
            encoder_hidden_states=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            position_ids=decoder_position_ids,
        )

        logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


if __name__ == "__main__":

    def debug_msg(s: str, sep="\n" * 2 + "#" * 20 + "\n" * 2) -> None:
        print(f"{sep}{s}{sep}", end="")

    # debug some stuff
    d = 768
    config = ContextWhisperConfig(
        d_model=d
    )  # TODO: somewhere here, the value of d_model gets overwritten!
    model = ContextWhisperModel(config)
    test_input_decoder = torch.randint(0, 10, size=(10, 100))
    model.decoder(test_input_decoder)
    debug_msg("model.decoder ok")

    test_spectrogram_encoder = torch.rand(10, 80, 3000)
    model.spectrogram_encoder(test_spectrogram_encoder)
    debug_msg(
        "model.spectrogram_encoder ok"
    )  # TODO: this is weird. How does it get cross attention?

    test_text_encoder = torch.randint(0, 10, size=(10, 100))
    model.text_encoder(test_text_encoder)
    debug_msg("model.text_encoder ok")

    # this should fail because there is no prompt:
    test_encoder_spec = torch.rand(10, 80, 3000)
    test_encoder_tok = torch.randint(0, 10, size=(10, 100))
    model.encoder.forward(
        text_encoder_input_ids=test_encoder_tok,
        spectrogram_input_features=test_encoder_spec,
    )
    debug_msg("model.encoder ok")

    test_decoder_tok = torch.randint(0, 10, size=(10, 100))
    model.forward(
        decoder_input_ids=test_decoder_tok,
        spectrogram_input_features=test_encoder_spec,
        text_encoder_input_ids=test_encoder_tok,
    )
    debug_msg("model.forward ok")
