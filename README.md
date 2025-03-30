# Idea:
use cross attention in the encoder of Whisper instead of just self attention.
The signal to cross-attend to would come from a language model that encodes context.
This way, Whisper should be able to more effectively use context in an intuitive way; the user says "this is an interview about XYZ", which gets encoded, and provided to Whisper's encoder.
## Day 1
- Looked into LLMs to use for encoding.
Decided on an encoder-decoder architecture, since that makes it easiest to extract user prompt embeddings.

## Day 2
- Looked heavily into Whisper architecture and started implementing my own architecture.

## Day 3
- Debugging, started writing tests

## Day 4
- Testing worked. Implemented using pretrained Whisper and Bert models, as well as Preprocessing for ContextWhisper.

## Day 5
- Implemented `ContextWhisperForCausalLM` and made training work

## Day 6
Despite some outstanding TODOs (in the code, and regarding test coverage), I am making the repo public today.

- Made some bug fixes (e.g. `ContextWhisperModel.get_decoder` returned a `WhisperDecoder` and not a `ContextWhisperDecoder`)
- Implemented freezing modules for `ContextWhisperForCausalLM`
- Managed to overfit on a single sample (for debugging)
- Managed to overfit on a single sample changing only the `text_encoder`.
This is a first indication that the signal from the `text_encoder` could be helpful.
