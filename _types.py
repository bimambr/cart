from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

if TYPE_CHECKING:
    import aiohttp


class CSVWriter(Protocol):
    def writerow(self, row: Iterable[Any], /) -> Any: ...  # pyright: ignore[reportExplicitAny, reportAny]

    def writerows(self, rows: Iterable[Iterable[Any]], /) -> None: ...  # pyright: ignore[reportExplicitAny]


class TranslationAttempt(TypedDict, total=False):
    type: Literal["attempt"]
    translation: str
    raw_output: str
    prompt: str
    system_prompt: str
    seed: int
    temp: float


class TranslationEvaluation(TypedDict, total=False):
    type: Literal["evaluation"]
    rubric: "Rubric"
    raw_output: str
    prompt: str
    system_prompt: str
    seed: int
    temp: float


class SourceTextEntry(TypedDict):
    source_lang: str
    target_lang: str
    text: str
    type: str
    id: int

    # RAG
    external_knowledge: list[str]
    idiom_matches: Sequence["IdiomMatchResult"]


class State(TypedDict):
    iteration_id: int
    source_text: SourceTextEntry
    next_state: str
    max_attempt: int
    attempt: int
    history: list[TranslationAttempt | TranslationEvaluation]
    optimiser_seed: int
    evaluator_seed: int
    client: "aiohttp.ClientSession"
    csv_writer: "CSVWriter | None"


class Corpus(TypedDict):
    source_lang: str
    target_lang: str
    type: str
    external_knowledge: list[str]
    texts: list["TextEntry"]


class TextEntry(TypedDict):
    content: str
    external_knowledge: list[str]


class IdiomEntry(TypedDict):
    idiom: str
    master_key: str
    senses: list[str]
    translations: dict[str, str]


class IdiomMatchResult(IdiomEntry):
    matched_chunk: str
    score: float


class Payload(TypedDict, total=False):
    model: str
    stream: bool
    temperature: float
    seed: int
    messages: list[dict[str, str]]
    cache_prompt: bool
    grammar: str
    thinking_budget: Literal["low", "high"]


class StreamingResponse(TypedDict, total=False):
    choices: list["StreamingChoice"]


class StreamingChoice(TypedDict, total=False):
    delta: "Delta"


class Delta(TypedDict, total=False):
    reasoning_content: str
    content: str


@dataclass
class CLIArgs:
    endpoint: str
    model: str
    embedding_model: str
    iterations: int
    input: str
    timeout: int
    refinement_iterations: int
    cache_prompt: bool
    omit_roles: bool
    save_output: bool
    baseline: bool
    vectorise: bool
    match_idioms_only: bool


class ExampleEntry(TypedDict):
    source_lang: str
    target_lang: str
    source_text: str
    translation: str
    rubric: "Rubric"
    revision: str | None
    known_idioms: list[IdiomEntry]


class Rubric(TypedDict):
    accuracy: "RubricEntry"
    acceptability: "RubricEntry"
    readability: "RubricEntry"


class RubricEntry(TypedDict):
    score: int
    feedback: str
