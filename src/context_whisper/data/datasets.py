"""
In this file, we define datasets used for training, validation and testing.
"""
from abc import ABC
from context_whisper.data.utils.parse_wiki import Document
from pathlib import Path
import torch
import torchaudio
import torchaudio.functional as F

def load_waveform(src: Path) -> torch.Tensor:
    """
    Load the waveform from the source file src and return a torch tensor.
    """
    return get_waveform_between(src, 0, None)

def get_waveform_between(src: Path, start: float, end: float | None, sr: int | None = 16_000) -> torch.Tensor:
    """
    Load waveform of src between start (seconds) and end (seconds).

    This is implemented separately in hopes that we can find a better-than-naiive implementation.
    """
    start_frame = int((sr or 16_000) * start)
    if end is not None:
        end_frame = int((sr or 16_000) * end)
    else:
        end_frame = -1
    wf, sr_ = torchaudio.load(src, start_frame, end_frame)
    if sr is not None and sr_ != sr:
        wf = F.resample(wf, sr_, sr)
    return wf

class ContextWhisperDataset(ABC):
    """
    Base dataset class for ContextWhisper use.

    Contains methods to get a description of a certain segment (as input to the text encoder), timestamp for
    a spoken segment, and the tokens and waveforms for the relevant segment.
    """

class SWCArticle:
    """
    Helper class to get timestamps, text and waveforms from a Spoken Wikipedia Corpus article.

    Parameters:
        description: str | None
            Description of the article or title of the article if not provided
        source_file: Path
            Path to the `aligned.swc` file for this article
        sample_rate: int
            Sample rate of the audio file
        lazy: bool
            Whether not to load the waveform immediately but rather on a needs-basis
    """
    def __init__(
        self,
        description: str | None,
        source_file: Path,
        sample_rate: int = 16_000,
        lazy: bool = False
    ) -> None:
        assert source_file.name == "aligned.swc"
        assert source_file.exists()
        self.description = description or source_file.parent.name
        self.source_file = source_file
        self.doc = Document.from_file(source_file).get_flat_token_tuples()
        self.sr = sample_rate
        source_audio_file = source_file.parent / "audio.ogg" # TODO: not always provided...
        assert source_file.exists()
        self.source_audio_file = source_audio_file
        if not lazy:
            self.wf = load_waveform(source_audio_file)
        else:
            self.wf = None
        self.texts = self.init_texts()

    @property
    def available_timestamps(self) -> list[float]:
        """
        Timestamps in seconds where speech utterances are specified.

        Note that the final timestamp is omitted by default.
        """
        return sorted(
            list(
                set(int(s) / 1000 for s, e, t in self.doc if s is not None)
                .union(set(int(e) / 1000 for s, e, t in self.doc if e is not None))
                .union({0})
            )
        )
    
    @property
    def durations(self) -> list[float]:
        """
        Durations between available timestamps.
        """
        return [e - s for s, e in zip(self.available_timestamps[:-1], self.available_timestamps[1:])]
    
    def init_texts(self) -> list[str]:
        """
        Texts between available timestamps of self.

        self.texts[i] starts at self.available_timestamps[i] and ends at self.available_timestamps[i+1], or
        at the end of the recording.
        """
        texts = []
        curr_t = ''
        for (s, e, t), (next_s, next_e, next_t) in zip(self.doc[:-1], self.doc[1:]):
            curr_t = curr_t + ' ' + t
            if next_s is not None: # next start is defined
                texts.append(curr_t)
                curr_t = ''
        # add final t:
        if self.doc[-1][0] is None: # start of final text is not defined; add it here
            texts[-1] = texts[-1] + ' ' + self.doc[-1][-1]
        else:
            texts.append(self.doc[-1][-1])
        return texts


    def __getitem__(self, i: int):
        return self.doc[i]
    
    def get_waveform(self, start: float, end: float | None) -> torch.Tensor:
        """
        Get the waveform between start (seconds) and end (seconds) of the recorded article.
        """
        if self.wf is None:
            return get_waveform_between(
                self.source_audio_file,
                start,
                end
            )
        if end is None:
            end_id = -1
        else:
            end_id = int(end * self.sr)
        return self.wf[..., int(start * self.sr):end_id]
    
    def as_tuples(self, idx: int | None) -> list[tuple[str, torch.Tensor, str]] | tuple[str, torch.Tensor, str]:
        """
        Get tuples of self.description, self.wf and self.texts.
        """
        if idx is None:
            return [
                (
                    self.description,
                    self.get_waveform(s, e),
                    self.texts[i]
                )
                for i, (s, e) in enumerate(zip(self.available_timestamps, self.available_timestamps[1:] + [None]))
            ]
        e = self.available_timestamps[idx + 1] if len(self.available_timestamps) > idx + 1 else None
        return (
                self.description,
                self.get_waveform(
                    self.available_timestamps[idx], e),
                self.texts[idx]
        )


class SWCDataset(ContextWhisperDataset):
    """
    Spoken Wikipedia Corpus dataset.
    """
    articles: dict[str, SWCArticle]
    def __init__(
        self,
        files: list[Path],
        descriptions: None | dict[str, None | str],
        sample_rate: int = 16_000,
        lazy: bool = False
    ) -> None:
        self.articles = {}
        if descriptions is None:
            descriptions = {str(f): None for f in files}
        for f in files:
            self.articles[str(f)] = SWCArticle(descriptions[str(f)], f, lazy=lazy, sample_rate=sample_rate)
        
    def batch_as_str(self, idx: int) -> list[tuple[str, torch.Tensor, str]]:
        return [
            a.as_tuples(idx)
            for a in self.articles.values()
        ] # type: ignore # the return type is not a list when we provide the idx as an argument.
    
if __name__ == '__main__':
    root = Path("~/.cache/usr_datasets").expanduser()
    files = [
        root / f"spoken-wiki/spoken-wikipedia-corpus-2.0/english/{topic}/aligned.swc"
        for topic in (
            "New_York_State_Route_174",
            "Wakefield%2c_Massachusetts",
        )
    ]
    ds = SWCDataset(
        files=files,
        descriptions=None
    )
    for i in range(3):
        print(ds.batch_as_str(i))
