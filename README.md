# Idea:
use cross attention in the encoder of Whisper instead of just self attention.
The signal to cross-attend to would come from a language model that encodes context.
This way, Whisper should be able to more effectively use context in an intuitive way; the user says "this is an interview about XYZ", which gets encoded, and provided to Whisper's encoder.
## Day 1
- Qwen-1.5 1.8B is quite fast. However it's decoder only. That might be undesirable because we have no latent representation of the user's request.

## Day 2
- Looked heavily into Whisper architecture and started implementing my own architecture.

## Day 3
- Debugging, started writing tests

## Day 4
- Testing worked. Implemented using pretrained Whisper and Bert models, as well as Preprocessing for ContextWhisper.