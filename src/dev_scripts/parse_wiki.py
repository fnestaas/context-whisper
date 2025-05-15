from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TypeVar

import xmltodict as xml

T = TypeVar("T", bound="BaseElement")
D = TypeVar("D", bound="Document")
Tk = TypeVar("Tk", bound="Token")


@dataclass
class BaseElement(ABC):
    @classmethod
    @abstractmethod
    def from_dict_(cls, input: dict) -> T:
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, input: dict | list | None | str) -> T | list[T] | None:
        if input is None:
            return None
        elif isinstance(input, str):
            assert hasattr(cls, "from_str")
            return cls.from_str(input)
        if isinstance(input, dict):
            return cls.from_dict_(input)
        assert isinstance(input, list)
        return [cls.from_dict(i) for i in input]

    def get_flat_token_tuples(self) -> list[tuple[int | None, int | None, str]]:
        """
        Return a flat list of token tuples.

        Walk this element and find all children with token children to find `Token`s.
        """
        retval = []
        for f in fields(self):
            obj = getattr(self, f.name)
            if isinstance(obj, BaseElement) or obj is None:
                obj = [obj]
            for oi in obj:
                if oi is None:
                    continue
                retval.extend(oi.get_flat_token_tuples())
        return retval


@dataclass
class Timing(BaseElement):
    start_time: int | None = None
    end_time: int | None = None

    @classmethod
    def from_dict_(cls, input_dict: dict) -> T:
        # only implemented because the parent has an abstract method like this
        return cls(
            start=input_dict.get("@start"),
            end=input_dict.get("@end"),
        )

    def get_flat_token_tuples(self):
        raise NotImplementedError()


@dataclass
class Phoneme(BaseElement):
    type: str | None = None
    timing: Timing | None = None

    @classmethod
    def from_dict_(cls, input_dict: dict) -> T:
        return cls(
            type=input_dict.get("@type"),
            timing=Timing(input_dict.get("@start"), input_dict.get("@end")),
        )

    def get_flat_token_tuples(self):
        raise NotImplementedError()


@dataclass
class Normalization(BaseElement):
    pronunciation: str | None = None
    timing: Timing | None = None
    ph: Phoneme | None = None

    @classmethod
    def from_dict_(cls, input_dict: dict) -> T:
        return cls(
            pronunciation=input_dict.get("@pronunciation"),
            timing=Timing(input_dict.get("@start"), input_dict.get("@end")),
            ph=Phoneme(input_dict.get("ph")),
        )

    def get_flat_token_tuples(self):
        raise NotImplementedError()


@dataclass
class Token(BaseElement):
    text: str
    n: Normalization | list[Normalization] | None = None

    @classmethod
    def from_dict_(cls, input_dict: dict) -> T:
        return cls(
            text=input_dict.get("#text"),
            n=Normalization.from_dict(input_dict.get("n")),
        )

    @classmethod
    def from_str(cls, input: str) -> Tk:
        return cls(text=input)

    def as_tuple(self) -> tuple[int | None, int | None, str]:
        """Return self as (start, end, text)"""
        n = self.n
        if n is None:
            t = None
        elif isinstance(n, list):
            t = None  # let's ignore this case
        else:
            t = n.timing
        if t is None:
            return (None, None, self.text)
        else:
            return (t.start_time, t.end_time, self.text)

    def get_flat_token_tuples(self) -> list[int | None, int | None, str]:
        return [self.as_tuple()]


@dataclass
class Sentence(BaseElement):
    t: Token | list[Token]

    @classmethod
    def from_dict_(cls, input_dict: dict) -> T:
        return cls(t=Token.from_dict(input_dict.get("t")))


@dataclass
class Paragraph(BaseElement):
    s: Sentence | list[Sentence] | None = None

    @classmethod
    def from_dict_(cls, input_dict: dict) -> T:
        return cls(s=Sentence.from_dict(input_dict.get("s")))


@dataclass
class Sectioncontent(BaseElement):
    s: Sentence | list[Sentence] | None = None
    p: Paragraph | list[Paragraph] | None = None
    section: "Section | list[Section] | None" = None

    @classmethod
    def from_dict_(cls, input_dict: dict) -> T:
        return cls(
            s=Sentence.from_dict(input_dict.get("s")),
            p=Paragraph.from_dict(input_dict.get("p")),
            section=Section.from_dict(input_dict.get("section")),
        )


@dataclass
class Sectiontitle(BaseElement):
    t: Token | list[Token] | None = None

    @classmethod
    def from_dict_(cls, input_dict: dict) -> T:
        return cls(
            t=Token.from_dict(input_dict.get("t")),
        )


@dataclass
class Section(BaseElement):
    level: int | None = None
    sectiontitle: Sectiontitle | None = None
    sectioncontent: Sectioncontent | None = None

    @classmethod
    def from_dict_(cls, input_dict: dict) -> T:
        return cls(
            level=input_dict.get("level"),
            sectiontitle=Sectiontitle.from_dict(input_dict.get("sectiontitle")),
            sectioncontent=Sectioncontent.from_dict(input_dict.get("sectioncontent")),
        )


@dataclass
class Document(BaseElement):
    p: Paragraph | list[Paragraph] | None = None
    section: Section | None = None
    s: Sentence | None = None

    @classmethod
    def from_file(cls, source: Path | str) -> D:
        with open(source) as xml_stream:
            input_dict: dict = xml.parse(xml_stream.read()).get("article")
        doc_dict = input_dict["d"]
        assert isinstance(doc_dict, dict)
        return Document.from_dict(doc_dict)

    @classmethod
    def from_dict_(cls, doc_dict: dict) -> T:
        return cls(
            p=Paragraph.from_dict(doc_dict.get("p")),
            section=Section.from_dict(doc_dict.get("section")),
            s=Sentence.from_dict(doc_dict.get("s")),
        )


if __name__ == "__main__":
    root = Path("~/.cache/usr_datasets").expanduser()
    for topic in [
        "Australia",
        "New_York_State_Route_174",
        "Wakefield%2c_Massachusetts",
    ]:
        source = (
            root
            / f"spoken-wiki/spoken-wikipedia-corpus-2.0/english/{topic}/aligned.swc"
        )
        doc = Document.from_file(source)
        # get a list of tuples of (start_time, end_time, text) for each Token:
        tuples = doc.get_flat_token_tuples()
        pass
        # parse_file(source)
