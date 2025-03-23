import pytest
import torch.optim.adam 
from src.context_whisper.modules import (
    ContextWhisperConfig,
    ContextWhisperPreTrainedModel,
    ContextWhisperEncoderLayer,
    ContextWhisperDecoderLayer,
    ContextWhisperSpectrogramEncoderLayer,
    ContextWhisperEncoder,
    ContextWhisperDecoder,
    ContextWhisperModel,
    ContextWhisperSpectrogramEncoder,
    ContextWhisperTextEncoder
)
from src.context_whisper.processing import ContextWhisperProcessor
import torch
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    # BaseModelOutput,
    Seq2SeqModelOutput,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperModel,
    WhisperEncoderLayer
)
from transformers.models.bert.modeling_bert import BertModel

from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer

# define shorthand aliases
# (we could also import the classes like this but by defining it here, we can use two names)
Config = ContextWhisperConfig
PreTrainedModel = ContextWhisperPreTrainedModel
EncoderLayer = ContextWhisperEncoderLayer
TextEncoder = ContextWhisperTextEncoder
DecoderLayer = ContextWhisperDecoderLayer
SpectrogramEncoderLayer = ContextWhisperSpectrogramEncoderLayer
Encoder = ContextWhisperEncoder
Decoder = ContextWhisperDecoder
Model = ContextWhisperModel
SpectrogramEncoder = ContextWhisperSpectrogramEncoder

@pytest.fixture
def default_config() -> ContextWhisperConfig:
    return ContextWhisperConfig(d_model=768)

@pytest.fixture
def small_config() -> ContextWhisperConfig:
    return ContextWhisperConfig(d_model=64, decoder_attention_heads=4, text_encoder_attention_heads=4)

@pytest.fixture
def pretrained_config() -> Config:
    return Config(
        d_model=768,
        whisper_pretrained_str='openai/whisper-small',
        text_encoder_pretrained_str='google-bert/bert-base-uncased'
    )

@pytest.fixture
def semipretrained_config():
    return Config(
        d_model=768,
        decoder_pretrained_str='openai/whisper-small',
        text_encoder_pretrained_str='google-bert/bert-base-uncased'
    )

@pytest.fixture
def default_model(default_config) -> Model:
    return Model(default_config)

@pytest.fixture
def pretrained_model(pretrained_config: Config) -> Model:
    # pretrained text_encoder, decoder and spectrogram_encoder
    return Model(pretrained_config)

@pytest.fixture
def semipretrained_model(semipretrained_config: Config) -> Model:
    return Model(semipretrained_config)    

@pytest.fixture
def small_model(small_config) -> Model:
    return Model(small_config)

@pytest.fixture
def spectrogram_input() -> torch.Tensor:
    return torch.rand(10, 80, 3000)

@pytest.fixture
def token_input() -> torch.LongTensor:
    return torch.randint(0, 100, size=(10, 100))

@pytest.fixture
def token_emb_input(default_config: Config) -> torch.Tensor:
    return torch.rand(10, 100, default_config.d_model)

@pytest.fixture
def small_token_emb_input(small_config: Config) -> torch.Tensor:
    return torch.rand(10, 100, small_config.d_model)

@pytest.fixture
def default_encoder_output(default_config: Config) -> torch.Tensor:
    return (torch.rand(10, default_config.d_model), None, None)

@pytest.fixture
def pretrained_encoder_output(pretrained_config: Config) -> torch.Tensor:
    return (torch.rand(10, pretrained_config.d_model), None, None)

@pytest.fixture
def small_encoder_output(small_config: Config) -> torch.Tensor:
    return (torch.rand(10, small_config.d_model), None, None)

@pytest.fixture
def config_args() -> dict[str, str]:
    """Map config args to their type"""
    return {
        'vocab_size': 'int',
        'num_mel_bins': 'int',
        'text_encoder_layers': 'int',
        'text_encoder_attention_heads': 'int',
        'text_encoder_ffn_dim': 'int',
        'decoder_layers': 'int',
        'decoder_attention_heads': 'int',
        'decoder_ffn_dim': 'int',
        'spectrogram_encoder_layers': 'int',
        'spectrogram_encoder_attention_heads': 'int',
        'spectrogram_encoder_ffn_dim': 'int',
        'text_encoder_layerdrop': 'float',
        'decoder_layerdrop': 'float',
        'spectrogram_encoder_layerdrop': 'float',
        'decoder_start_token_id': 'int',
        'use_cache': 'bool',
        'is_encoder_decoder': 'bool',
        'activation_function': 'str',
        'd_model': 'int',
        'dropout': 'float',
        'attention_dropout': 'float',
        'activation_dropout': 'float',
        'init_std': 'float',
        'scale_embedding': 'bool',
        'max_source_positions': 'int',
        'max_target_positions': 'int',
        'pad_token_id': 'int',
        'bos_token_id': 'int',
        'eos_token_id': 'int',
        'suppress_tokens': 'Optional',
        'begin_suppress_tokens': 'Optional',
        'use_weighted_layer_sum': 'bool',
        'classifier_proj_size': 'int',
        'apply_spec_augment': 'bool',
        'mask_time_prob': 'float',
        'mask_time_length': 'int',
        'mask_time_min_masks': 'int',
        'mask_feature_prob': 'float',
        'mask_feature_length': 'int',
        'mask_feature_min_masks': 'int',
        'median_filter_width': 'int',
    }

config_args2config_attr = {
    'vocab_size': 'vocab_size',
    'num_mel_bins': 'num_mel_bins',
    'd_model': 'd_model',
    'text_encoder_layers': 'text_encoder_layers',
    'text_encoder_attention_heads': 'text_encoder_attention_heads',
    'decoder_layers': 'decoder_layers',
    'decoder_attention_heads': 'decoder_attention_heads',
    'spectrogram_encoder_layers': 'spectrogram_encoder_layers',
    'spectrogram_encoder_attention_heads': 'spectrogram_encoder_attention_heads',
    'text_encoder_ffn_dim': 'text_encoder_ffn_dim',
    'decoder_ffn_dim': 'decoder_ffn_dim',
    'spectrogram_encoder_ffn_dim': 'spectrogram_encoder_ffn_dim',
    'dropout': 'dropout',
    'attention_dropout': 'attention_dropout',
    'activation_dropout': 'activation_dropout',
    'activation_function': 'activation_function',
    'init_std': 'init_std',
    'text_encoder_layerdrop': 'text_encoder_layerdrop',
    'decoder_layerdrop': 'decoder_layerdrop',
    'spectrogram_encoder_layerdrop': 'spectrogram_encoder_layerdrop',
    'use_cache': 'use_cache',
    'num_hidden_layers': 'text_encoder_layers',
    'scale_embedding': 'scale_embedding',
    'max_source_positions': 'max_source_positions',
    'max_target_positions': 'max_target_positions',
    'classifier_proj_size': 'classifier_proj_size',
    'use_weighted_layer_sum': 'use_weighted_layer_sum',
    'apply_spec_augment': 'apply_spec_augment',
    'mask_time_prob': 'mask_time_prob',
    'mask_time_length': 'mask_time_length',
    'mask_time_min_masks': 'mask_time_min_masks',
    'mask_feature_prob': 'mask_feature_prob',
    'mask_feature_length': 'mask_feature_length',
    'mask_feature_min_masks': 'mask_feature_min_masks',
    'median_filter_width': 'median_filter_width',
    'pad_token_id': 'pad_token_id',
    'bos_token_id': 'bos_token_id',
    'eos_token_id': 'eos_token_id',
    'is_encoder_decoder': 'is_encoder_decoder',
    'decoder_start_token_id': 'decoder_start_token_id',
    'suppress_tokens': 'suppress_tokens',
    'begin_suppress_tokens': 'begin_suppress_tokens',
}

class TestConfig:
    def test_random_args(self, config_args: dict):
        def get_value(v: str):
            if v == 'int':
                return 10 
            elif v == 'bool':
                return False
            elif v == 'float':
                return .5
            elif v == 'Optional':
                return None
            elif v == 'str':
                return 'gelu'
            else:
                raise ValueError(f'{v=} not expected as a type')
        kwargs = {
            k: get_value(v)
            for k, v in config_args.items()
        }
        config = Config(**kwargs)
        for k, v in kwargs.items():
            assert getattr(config, config_args2config_attr[k]) == v 
        
class TestPretrained:
    def test_hasgrad(self, small_config: Config):
        model = Model(small_config)
        # check params of different modules
        module = model.text_encoder
        assert module.requires_grad_()
        module = model.spectrogram_encoder
        assert module.requires_grad_()
        module = model.decoder
        assert module.requires_grad_()

class TestTextEncoder:
    """Tests for textencoder and its layer"""
    # TODO: test layer
    def test_small_init(self, small_config: Config):
        enc = TextEncoder(small_config)
        assert enc.embeddings.word_embeddings.embedding_dim == small_config.d_model
        assert enc.embeddings.position_embeddings.embedding_dim == small_config.d_model
    
    def test_basic_tok(self, small_model: Model, token_input: torch.LongTensor):
        enc = small_model.text_encoder
        output = enc.forward(input_ids=token_input)
        assert output.last_hidden_state.shape == (*token_input.shape, small_model.config.d_model)
        assert output.pooler_output.shape == (*token_input.shape[:-1], small_model.config.d_model)

    def test_basic_emb(self, small_model: Model, small_token_emb_input: torch.Tensor):
        enc = small_model.text_encoder
        output = enc.forward(inputs_embeds=small_token_emb_input)
        assert output.last_hidden_state.shape == (*small_token_emb_input.shape[:-1], small_model.config.d_model)

class TestSpectrogramEncoder:
    """Tests for spectrogram encoder and its layer"""
    # TODO: test layer
    @pytest.mark.parametrize(
        'size_id',
        ['small', 'default', 'pretrained', 'semipretrained']
    )
    def test_basic_emb(
        self, 
        size_id: str,
        request: pytest.FixtureRequest,
        spectrogram_input: torch.Tensor
    ):
        model = request.getfixturevalue(f'{size_id}_model')
        assert isinstance(model, Model)
        enc = model.spectrogram_encoder
        config = model.config
        outputs = enc.forward(input_features=spectrogram_input)
        assert outputs.last_hidden_state.shape == (len(spectrogram_input), config.max_source_positions, config.d_model)

class TestEncoder:
    """Test the whole encoder"""
    # TODO: test layer
    @pytest.mark.parametrize(
        'size_id',
        ['small', 'default', 'pretrained', 'semipretrained']
    )
    def test_basic_inputs(
        self, 
        size_id: str, 
        request:pytest.FixtureRequest, 
        spectrogram_input: torch.Tensor, 
        token_input: torch.Tensor
    ):
        with torch.no_grad():
            model = request.getfixturevalue(f'{size_id}_model')
            assert isinstance(model, Model)
            enc = model.encoder 
            config = model.config
            outputs = enc.forward(spectrogram_input_features=spectrogram_input, text_encoder_input_ids=token_input)
            assert outputs.last_hidden_state.shape == (len(spectrogram_input), config.max_source_positions, config.d_model)

class TestDecoder:
    """Test the whole decoder"""
    # TODO: test layer
    @pytest.mark.parametrize(
        'size_id',
        ['small', 'default', 'pretrained']
    )
    def test_basic_inputs(
        self,
        size_id: str,
        request: pytest.FixtureRequest,
        token_input: torch.LongTensor, 
    ):
        with torch.no_grad():
            model = request.getfixturevalue(f'{size_id}_model')
            assert isinstance(model, Model)
            decoder = model.decoder
            output = decoder.forward(
                input_ids=token_input, 
                encoder_hidden_states=request.getfixturevalue(
                    f'{size_id}_encoder_output'
                )[0]
            )
            assert isinstance(output, BaseModelOutputWithPastAndCrossAttentions)
            assert output.last_hidden_state.shape == (*token_input.shape, decoder.config.d_model)

class TestModel:
    """Test the whole model"""
    def test_basic_inputs(
        self,
        small_model: Model,
        token_input: torch.LongTensor, # note: used as input to both decoder and text_encoder
        spectrogram_input: torch.Tensor
    ):
        with torch.no_grad():
            outputs = small_model.forward(
                decoder_input_ids=token_input,
                spectrogram_input_features=spectrogram_input,
                text_encoder_input_ids=token_input
            )
            assert isinstance(outputs, Seq2SeqModelOutput)
            assert outputs.last_hidden_state.shape == (*token_input.shape, small_model.config.d_model)

    @pytest.mark.parametrize(
        'module', ['decoder', 'spectrogram_encoder', 'text_encoder']
    )
    def test_freeze_module(
        self, 
        small_model: Model, 
        module: str,
        token_input: torch.LongTensor,
        small_encoder_output: torch.Tensor,
        small_token_emb_input: torch.Tensor,
        spectrogram_input: torch.Tensor,
    ):
        """
        Test the freeze_module functionality by training for 
        a couple of iterations
        """
        def get_ref_ids(model, key, *dims, require_grad: bool):
            module = getattr(model, key)
            assert isinstance(module, torch.nn.Module)
            params = module.parameters()
            for p in params:
                if p.dim() == 2 and all([s >= max(d) for s, d in zip(p.shape, dims)]) and p.requires_grad == require_grad:
                    retval = p[dims[0], dims[1]].detach()
                    retval.requires_grad = False 
                    return retval
            raise ValueError(f'did not find an appropriate parameter for {key=}. It has {sum([p.requires_grad for p in module.parameters()])} parameters with grad.')

        model = small_model
        # freeze all but one module:
        modules_str = ['decoder', 'spectrogram_encoder', 'text_encoder']
        frozen_modules_str = [m for m in modules_str if m != module]
        for _module in frozen_modules_str:
            if _module == module: 
                continue
            model.freeze_module(_module)
        # sanity check that the number of parameters requiring a gradient is consistent:
        # the number of parameters requiring a gradient in the model must be the same
        # as in the module chosen
        n_requires_grad_model = sum([p.requires_grad for p in model.parameters()])
        n_requires_grad_module = sum([p.requires_grad for p in getattr(model, module).parameters()])
        assert n_requires_grad_model == n_requires_grad_module
        assert n_requires_grad_model > 0
        assert all([all([not p.requires_grad for p in getattr(model, m).parameters()]) for m in frozen_modules_str])
        # get reference parameters for comparison
        ids = [ # ids to check that there is (no) change after training
            (
                torch.randint(model.config.d_model, size=(1, )).item(),
                torch.randint(model.config.d_model, size=(1, )).item(),
            ) for _ in range(100)
        ]
        row_ids, col_ids = zip(*ids)
        reference_params = {
            k: get_ref_ids(model, k, row_ids, col_ids, require_grad=k==module)
            for k in modules_str
        }

        # train for one step:
        # note that the choices for optimizer etc are not really relevant,
        # we just want to check that some parameters change and others don't
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.zero_grad()
        loss_fn = torch.nn.CrossEntropyLoss() 
        outputs = model.forward(
            decoder_input_ids=token_input, 
            encoder_outputs=small_encoder_output if module == 'decoder' else None,
            text_encoder_input_ids=token_input if module == 'text_encoder' else None,
            text_encoder_inputs_embeds=small_token_emb_input if module == 'spectrogram_encoder' else None,
            spectrogram_input_features=spectrogram_input if module != 'decoder' else None,
        )
        loss = loss_fn(outputs.last_hidden_state, torch.rand(*outputs.last_hidden_state.shape)) # take a random target
        loss.backward()
        optimizer.step()

        new_params = {
            k: get_ref_ids(model, k, row_ids, col_ids, require_grad=k==module)
            for k in modules_str
        }

        # assert that the frozen modules do not change:
        for k in frozen_modules_str:
            assert torch.allclose(new_params[k], reference_params[k])

        # assert that the unfrozen module changes:
        assert not torch.allclose(new_params[module], reference_params[module])
    
    @pytest.mark.parametrize(
        ['pretrained_bert', 'pretrained_whisper', 'pretrained_whisper_decoder_only'],
        [
            (True, True, False),
            (True, False, False),
            (False, True, False),
            (True, False, True)
        ]
    )
    def test_pretrained(
        self,
        pretrained_bert: bool,
        pretrained_whisper: bool,
        pretrained_whisper_decoder_only: bool, # use only a pretrained decoder, not a pretrained spectrogram encoder
    ):
        """
        Test that we can load pretrained models
        Try out
        - pretrained BERT and pretrained Whisper
        - pretrained BERT but not pretrained Whisper
        - not pretrained BERT but pretrained Whisper
        """
        def model2params(m: torch.nn.Module):
            return torch.concat([
                p.flatten()
                for p in m.parameters()
            ])
        def model_eq(model1: torch.nn.Module, model2: torch.nn.Module):
            p1 = model2params(model1)
            p2 = model2params(model2)
            assert p1.shape == p2.shape 
            assert torch.allclose(p1, p2)
        if pretrained_bert:
            text_encoder_pretrained_str = 'google-bert/bert-base-uncased'
        else:
            text_encoder_pretrained_str = None
        if pretrained_whisper:
            whisper_pretrained_str = 'openai/whisper-small'
        else: 
            whisper_pretrained_str = None
        if pretrained_whisper_decoder_only:
            decoder_pretrained_str = 'openai/whisper-small'
        else:
            decoder_pretrained_str = None
        config = Config(
            d_model=768,
            whisper_pretrained_str=whisper_pretrained_str,
            decoder_pretrained_str=decoder_pretrained_str,
            text_encoder_pretrained_str=text_encoder_pretrained_str
        )
        model = Model(config)
        if pretrained_bert:
            bert = BertModel.from_pretrained('google-bert/bert-base-uncased')
            model_eq(bert, model.text_encoder)
        if pretrained_whisper or pretrained_whisper_decoder_only:
            # check decoder
            whisper = WhisperModel.from_pretrained('openai/whisper-small')
            assert isinstance(whisper, WhisperModel)
            decoder = model.decoder 
            model_eq(decoder, whisper.get_decoder())
        if pretrained_whisper and not pretrained_whisper_decoder_only:
            # check spectrogram_encoder
            enc = model.spectrogram_encoder
            wenc = whisper.get_encoder()
            model_eq(enc.conv1, wenc.conv1)
            model_eq(enc.conv1, wenc.conv1)
            model_eq(enc.embed_positions, wenc.embed_positions)
            model_eq(enc.layer_norm, wenc.layer_norm)
            for i in range(len(enc.layers)):
                l1 = enc.layers[i]
                l2 = wenc.layers[i]
                assert isinstance(l1, ContextWhisperSpectrogramEncoderLayer)
                assert isinstance(l2, WhisperEncoderLayer)
                model_eq(l1.self_attn, l2.self_attn)
                model_eq(l1.fc1, l2.fc1)
                model_eq(l1.fc2, l2.fc2)
                model_eq(l1.final_layer_norm, l2.final_layer_norm)
            # TODO: check that this is a working ContextWhisperDecoder!

class TestProcessor:
    def test_basic_inputs(self, pretrained_model: Model):
        whisper_str = 'openai/whisper-small'
        bert_str = 'google-bert/bert-base-uncased'
        with torch.no_grad():
            processor = ContextWhisperProcessor(
                feature_extractor=WhisperFeatureExtractor.from_pretrained(whisper_str),
                tokenizer=WhisperTokenizer.from_pretrained(whisper_str),
                prompt_tokenizer=BertTokenizer.from_pretrained(bert_str)
            )
            tokens = processor(text='An example sentence')
            text_encoder_tokens = processor(prompt='some user prompt')
            audio_in = torch.rand(3000)
            features = processor(audio=audio_in.numpy(), sampling_rate=int(16e3))
            model_out = pretrained_model.forward(
                spectrogram_input_features=torch.tensor(features['input_features']),
                text_encoder_input_ids=torch.tensor(text_encoder_tokens['input_ids']).unsqueeze(0),
                decoder_input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0),
            )
            assert model_out.last_hidden_state.shape == (1, len(tokens['input_ids']), pretrained_model.decoder.layer_norm.weight.shape[0])

