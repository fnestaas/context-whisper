"""
Classes and functions related to processing inputs to
and outputs from ContextWhisper
"""

from transformers.processing_utils import ProcessorMixin
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
import torch


class ContextWhisperProcessor(ProcessorMixin):
    attributes = ['feature_extractor', 'tokenizer', 'prompt_tokenizer']
    feature_extractor_class = 'WhisperFeatureExtractor'
    tokenizer_class = 'WhisperTokenizer'
    prompt_tokenizer_class = 'BertTokenizer'

    def __init__(
        self, 
        feature_extractor: WhisperFeatureExtractor, 
        tokenizer: WhisperTokenizer,
        prompt_tokenizer: BertTokenizer
    ) -> None:
        super().__init__(feature_extractor, tokenizer, prompt_tokenizer)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False
        # TODO: not sure about the below. Are we overwriting something?
        # for thing in ['feature_extractor', 'tokenizer', 'prompt_tokenizer']:
        #     if hasattr(self, thing):
        #         print(f'Warning! {thing} is an attribute with {getattr(self, thing)=}')
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer 
        self.prompt_tokenizer = prompt_tokenizer
    
    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` argument to WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] and the `text`
        argument to [`~WhisperTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        prompt = kwargs.pop('prompt', None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is None and text is None and prompt is None:
            raise ValueError("You need to specify either an `audio`, `text` or `prompt` input to process.")

        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)
        if prompt is not None:
            prompt_encodings = self.prompt_tokenizer(prompt, **kwargs)

        if audio is not None and text is None and prompt is None:
            return inputs
        elif audio is None and text is not None and prompt is None:
            return encodings
        elif audio is None and text is None and prompt is not None:
            return prompt_encodings
        elif audio is not None and text is not None:
            # text and audio are specified. Set labels of features to be 
            # encodings of the text.
            inputs["labels"] = encodings["input_ids"]
            return inputs
        elif audio is not None and text is not None and prompt is not None:
            inputs["labels"] = encodings["input_ids"]
            inputs["prompt"] = prompt_encodings["input_ids"]
            return inputs

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)


    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def get_prompt_ids(self, text: str, return_tensors="np"):
        return self.tokenizer.get_prompt_ids(text, return_tensors=return_tensors)
        
if __name__ == '__main__':
    # do some debugging
    prompt_tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
    tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-small')
    feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-small')
    processor = ContextWhisperProcessor(feature_extractor, tokenizer, prompt_tokenizer)
    a = processor(audio=torch.rand(80, 3000).numpy(), sampling_rate=int(16e3), return_tensors='pt')
    b = processor(text='this is an example text', return_tensors='pt')
    c = processor(prompt='this is an example prompt', return_tensors='pt')
