"""
Copyright 2026 Muhammad Bima Ramadhan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    import aiohttp


class TranslationAttempt(TypedDict, total=False):
    type: Literal["attempt"]
    translation: str
    raw_content: str
    raw_reasoning: str
    prompt: str
    system_prompt: str
    seed: int
    temp: float


class TranslationEvaluation(TypedDict, total=False):
    type: Literal["evaluation"]
    rubric: "Rubric"
    raw_content: str
    raw_reasoning: str
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


class Corpus(TypedDict):
    source_lang: str
    target_lang: str
    type: str
    texts: list["TextEntry"]


class TextEntry(TypedDict):
    content: str
    external_knowledge: list[str]


class IdiomEntry(TypedDict):
    idiom: str
    master_key: str
    senses: list[str]
    translations: dict[str, str]


class IdiomMatchCandidate(TypedDict):
    phrase_idx: int
    sense_idx: int
    base_score: float
    rerank_score: float


class IdiomMatchResult(IdiomEntry):
    matched_chunk: str
    base_score: float
    rerank_score: float


class ChatTemplateKwargs(TypedDict, total=False):
    enable_thinking: bool


class Payload(TypedDict, total=False):
    model: str
    stream: bool
    temperature: float
    seed: int
    messages: list[dict[str, str]]
    cache_prompt: bool
    grammar: str
    reasoning_budget: int
    chat_template_kwargs: ChatTemplateKwargs


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
    evaluator_temperature: float
    optimiser_init_temperature: float
    optimiser_retry_temperature: float
    embedding_model: str
    rerank_model: str
    iterations: int
    input: str
    timeout: int
    refinement_iterations: int
    cache_prompt: bool
    save_output: bool
    baseline: bool
    vectorise: bool
    match_idioms_only: bool
    verbose: bool


class Rubric(TypedDict):
    accuracy: "RubricEntry"
    acceptability: "RubricEntry"
    readability: "RubricEntry"


class RubricEntry(TypedDict):
    score: int
    feedback: str
