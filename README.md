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

## Days 7 and 8
I looked more into GPU options and datasets for training.
The data proved more challenging than I had thought - it was hard to find long recordings with transcriptions.
- LibriSpeech and CommonVoice have sometimes, but not always, coherent speech accross files.
Even files with adjacient names apparently do not have to be consecutive - e.g. from LibriSpeech train.100 with ChapterID `16042`, we find an excerpt from Les Miserables, but the recordings do not read out the full chapter.
This makes LibriSpeech less reliable than I would like. 
Maybe it can still be used, but ideally, I would like longer and coherent transcriptions, so that we can provide sufficient context to my new models.
- Other options I will consider are [LibriVox](https://archive.org/download/count_monte_cristo_0711_librivox/) (example link only), which is also [on Huggingface](https://huggingface.co/datasets/pykeio/librivox-tracks), [The Spoken Wikipedia Corpus](https://www.fdr.uni-hamburg.de/record/1875), and possibly [The Switchboard Dataset](https://isip.piconepress.com/projects/switchboard/).

## Day 9
Long break, had time again.
I decided on using the Spoken Wikipedia Corpus because it has long, coherent text, as a starting point.
However, I was not able to find a ready-to-use version of the dataset and wrote my own python parser based on the provided `aligned.swc` files and the schema defined on [their website](https://nats.gitlab.io/swc/).
The parser is (at the time of writing) in `src/dev_scripts/parse_wiki.py`.

## Day 10
Short session. 
Started implementing `SWCDataset`, a dataset which is to be used in pytorch to train ContextWhisperModels.
It is based on the Spoken Wikipedia Corpus and provides descriptions, waveforms and text from segments of batches of Wikipedia articles.
