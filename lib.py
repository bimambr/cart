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

import argparse
import asyncio
import json
import logging
import os
import pickle
import re
from collections.abc import Awaitable
from itertools import product
from pathlib import Path
from typing import Protocol, TypeGuard, TypeVar, cast, final

import aiohttp
import numpy as np
import spacy
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from spacy.language import Language

from _types import (
    CLIArgs,
    IdiomEntry,
    IdiomMatchCandidate,
    IdiomMatchResult,
    Payload,
    StreamingResponse,
)

LOGGER = logging.getLogger("lib")

T = TypeVar("T")


class Bail(Exception): ...


ENDPOINT = "http://localhost:8127/v1/chat/completions"
MODEL_NAME = "gemma-4-E2B-it-GGUF"
EMBED_MODEL_NAME = "./MongoDB/mdbr-leaf-ir"
RERANK_MODEL_NAME = "./mixedbread-ai/mxbai-rerank-base-v2"
DEFAULT_N_ITERATIONS = 5
MAX_N_ITERATIONS = 10
DEFAULT_REFINEMENT_ITERATIONS = 3
MAX_REFINEMENT_ITERATIONS = 5
TIMEOUT = 240
VECTORISED_DICTIONARY_PATH = "vectorised_dict.pkl"
PUNCTUATIONS = ".,!?\"'()[]{}"
MIN_SENTENCE_LENGTH = 5
MIN_RETRIEVAL_SCORE = 0.1
MIN_RERANK_SCORE = 0.6
TOP_K = 50


class LoadedEmbedder(Protocol):
    bi_model: str
    rerank_model: str
    idioms: dict[str, IdiomEntry]
    phrases: list[str]
    phrase_embeddings: torch.Tensor
    embedder: SentenceTransformer
    reranker: CrossEncoder
    nlp: Language


@final
class Embedder:
    def __init__(self, bi_model: str, rerank_model: str) -> None:
        self.bi_model = bi_model
        self.rerank_model = rerank_model

        self.idioms: dict[str, IdiomEntry] | None = None
        self.phrases: list[str] | None = None
        self.phrase_embeddings: torch.Tensor | None = None
        self.embedder: SentenceTransformer | None = None
        self.reranker: CrossEncoder | None = None
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def is_loaded(obj: "Embedder") -> TypeGuard[LoadedEmbedder]:
        return (
            obj.idioms is not None
            and obj.phrases is not None
            and obj.phrase_embeddings is not None
            and obj.embedder is not None
            and obj.reranker is not None
        )

    def load_vectors(self) -> None:
        if not os.path.exists(VECTORISED_DICTIONARY_PATH):
            return

        if Embedder.is_loaded(self):
            return

        with open(VECTORISED_DICTIONARY_PATH, "rb") as f:
            _vector_data = pickle.load(f)  # pyright: ignore[reportAny]
            self.idioms = cast("dict[str, IdiomEntry]", _vector_data["dictionary"])
            self.phrases = cast("list[str]", _vector_data["phrases"])
            self.phrase_embeddings = cast(
                torch.Tensor, _vector_data["phrase_embeddings"]
            )

        self.embedder = SentenceTransformer(self.bi_model)
        self.reranker = CrossEncoder(self.rerank_model)

    async def get_idiom_definitions(self, excerpt: str) -> list[IdiomMatchResult]:
        if not Embedder.is_loaded(self):
            return []

        doc = self.nlp(excerpt)
        sentences = [
            sent.text.strip()
            for sent in doc.sents
            if len(sent.text.strip()) > MIN_SENTENCE_LENGTH
        ]
        candidates: list[tuple[int, int, float]] = []
        rerank_pairs: list[list[str]] = []

        for sent in sentences:
            sentence_embedding = cast(
                torch.Tensor,
                self.embedder.encode(  # pyright: ignore[reportUnknownMemberType, reportCallIssue]
                    sent,
                    convert_to_tensor=True,
                    device=self.phrase_embeddings.device,  # pyright: ignore[reportArgumentType]
                    prompt_name=None,
                ),
            )
            cosine_scores = util.cos_sim(  # pyright: ignore[reportUnknownMemberType]
                sentence_embedding, self.phrase_embeddings
            ).flatten()
            top_scores, top_indices = torch.topk(
                cosine_scores, k=min(TOP_K, len(cosine_scores))
            )

            for score_tensor, phrase_idx_tensor in zip(top_scores, top_indices):
                if (score := float(score_tensor)) < MIN_RETRIEVAL_SCORE:
                    continue

                phrase_idx = int(phrase_idx_tensor)
                idiom_key = self.phrases[phrase_idx]
                entry = self.idioms[idiom_key]
                context = f"Target: {sent} | Context: {excerpt}"

                for sense_idx, sense in enumerate(entry.get("senses", [])):
                    candidates.append((phrase_idx, sense_idx, score))
                    rerank_pairs.append(
                        [
                            f"{idiom_key}: {sense}",
                            context,
                        ]
                    )

        if not candidates:
            return []

        rerank_scores = await asyncio.to_thread(self.reranker.predict, rerank_pairs)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        cross_scores: list[float] = np.asarray(rerank_scores).ravel().tolist()
        valid_candidates: list[IdiomMatchCandidate] = []

        for i, (p_idx, s_idx, base_phrase_score) in enumerate(candidates):
            cross_score = cross_scores[i]
            bi_context_score = max(0.0, base_phrase_score)
            hybrid_score = (bi_context_score * 0.5) + (cross_score * 0.5)

            LOGGER.debug(
                "Candidate: %s | Bi Score: %.3f | Rerank Score: %.3f | Hybrid: %.3f",
                self.phrases[p_idx],
                base_phrase_score,
                cross_score,
                hybrid_score,
            )

            if cross_score < MIN_RERANK_SCORE:
                continue

            valid_candidates.append(
                IdiomMatchCandidate(
                    phrase_idx=p_idx,
                    sense_idx=s_idx,
                    base_score=base_phrase_score,
                    rerank_score=cross_score,
                )
            )

        valid_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        results: dict[str, IdiomMatchResult] = {}

        for candidate in valid_candidates:
            idiom_key = self.phrases[candidate["phrase_idx"]]
            entry = self.idioms[idiom_key]
            master_key = entry["master_key"]
            existing_senses = results.setdefault(
                master_key,
                IdiomMatchResult(
                    idiom=idiom_key,
                    senses=[],
                    translations=entry.get("translations", {}),
                    matched_chunk=idiom_key,
                    base_score=round(candidate["base_score"], 3),
                    rerank_score=round(candidate["rerank_score"], 3),
                    master_key=master_key,
                ),
            )["senses"]

            if (
                sense := entry.get("senses", [])[candidate["sense_idx"]]
            ) not in existing_senses:
                existing_senses.append(sense)

            if len(results) >= 5:
                break

        LOGGER.info("Found idiom matches: \n%s", json.dumps(results, indent=4))
        return [*results.values()]

    @staticmethod
    def _expand_phrase(phrase: str) -> list[str]:
        tokens = phrase.split()
        options = [
            option
            for token in tokens
            if (
                option := [token[1:-1], ""]
                if token.startswith("(") and token.endswith(")")
                else token.split("/")
            )
        ]
        return [" ".join(w for w in c if w) for c in product(*options)]

    def generate_vectors(self) -> None:
        normalised_dict: dict[str, IdiomEntry] = {}

        for path in ("idiom_dict/wiktionary-en.json", "idiom_dict/cherrypicked.json"):
            with open(path, "r", encoding="utf-8") as f:
                json_data = json.load(f)  # pyright: ignore[reportAny]
                for phrase, data in json_data.items():  # pyright: ignore[reportAny]
                    assert isinstance(phrase, str)
                    normalised_dict[phrase] = IdiomEntry(
                        idiom=phrase,
                        senses=data.get("senses", []),  # pyright: ignore[reportAny]
                        translations={},
                        master_key=data.get("master_key", phrase),  # pyright: ignore[reportAny]
                    )

        with open("idiom_dict/idiomKB.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)  # pyright: ignore[reportAny]
            for entry in json_data:  # pyright: ignore[reportAny]
                phrase = entry.get("idiom").lower()  # pyright: ignore[reportAny]
                en_meaning = entry.get("en_meaning")  # pyright: ignore[reportAny]

                assert isinstance(phrase, str)
                assert isinstance(en_meaning, str)

                translations: dict[str, str] = {}
                if "zh_meaning" in entry:
                    translations["zh"] = entry["zh_meaning"]
                if "ja_meaning" in entry:
                    translations["ja"] = entry["ja_meaning"]

                if phrase in normalised_dict:
                    if (
                        en_meaning
                        and en_meaning not in normalised_dict[phrase]["senses"]
                    ):
                        normalised_dict[phrase]["senses"].append(en_meaning)
                    normalised_dict[phrase]["translations"].update(translations)
                    continue

                normalised_dict[phrase] = IdiomEntry(
                    idiom=phrase,
                    senses=[en_meaning] if en_meaning else [],
                    translations=translations,
                    master_key=phrase,
                )

        for k, v in {**normalised_dict}.items():
            for variant in self._expand_phrase(k):
                if variant in normalised_dict:
                    continue

                normalised_dict[variant] = {**v, "idiom": variant}

        phrases = [*normalised_dict]
        LOGGER.info("Loading model and computing %d embeddings", len(phrases))
        model = SentenceTransformer(self.bi_model)
        phrase_embeddings = model.encode(  # pyright: ignore[reportUnknownMemberType]
            phrases, convert_to_tensor=True, prompt_name="query"
        )

        with open(VECTORISED_DICTIONARY_PATH, "wb") as f:
            pickle.dump(
                {
                    "dictionary": normalised_dict,
                    "phrases": phrases,
                    "phrase_embeddings": phrase_embeddings,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        LOGGER.info("Saved to %s", VECTORISED_DICTIONARY_PATH)


async def stream_response(response: aiohttp.ClientResponse) -> str:
    json_data: StreamingResponse = {}
    full_response = ""
    chunk = ""

    LOGGER.info("Streaming response...")
    async for line in response.content:
        decoded_line = line.decode("utf-8").strip()
        if not decoded_line.startswith("data: "):
            continue
        data = decoded_line[len("data: ") :].strip()
        if data == "[DONE]":
            break
        try:
            json_data = cast(StreamingResponse, json.loads(data))
            delta = json_data.get("choices", [{}])[0].get("delta", {})
            reasoning = delta.get("reasoning_content") or ""
            chunk = (delta.get("content")) or ""
            full_response += chunk
            if reasoning:
                print(reasoning, end="", flush=True)
            if chunk:
                print(chunk, end="", flush=True)
        except json.JSONDecodeError:
            LOGGER.error("Failed to decode JSON chunk: %s", data)

    if not chunk.endswith("\n"):
        print()

    LOGGER.info("Completed streaming response. Last chunk: %s", json_data)
    # let llama-server disconnect
    await asyncio.sleep(0.1)
    return full_response.strip()


async def run_inference(
    client: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    temperature: float,
    seed: int,
    timeout: float,
    grammar: str | None = None,
    cache_prompt: bool = False,
    enable_thinking: bool = True,
    messages: list[tuple[str, str, str]] | None = None,
) -> str:
    LOGGER.info("Hitting %s with temp=%f, seed=%d", endpoint, temperature, seed)
    for i in range(3):
        LOGGER.debug("Trying attempt %d...", i + 1)
        try:
            formatted_messages = [
                {"role": role, "content": content, "name": name}
                for role, content, name in messages or []
            ]
            if not formatted_messages:
                raise ValueError("Messages must be provided for inference.")

            payload: Payload = {
                "model": model,
                "stream": True,
                "temperature": temperature,
                "seed": seed,
                "messages": formatted_messages,
                "cache_prompt": cache_prompt,
                "reasoning_budget": -1 if enable_thinking else 0,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }

            if grammar is not None:
                payload["grammar"] = grammar

            async with client.post(
                endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                response.raise_for_status()
                return await stream_response(response)
        except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError) as e:
            LOGGER.error("API request failed: %s", e, exc_info=e)
            LOGGER.info("Retrying in 1 second...")
            await asyncio.sleep(1)
            continue
        except json.JSONDecodeError:
            LOGGER.error("Failed to decode JSON from response.")
            return "Failed to decode JSON from response."

    return "API request failed"


async def wait(awaitable: Awaitable[T], event: asyncio.Event) -> T:
    async def _wrap(awaitable: Awaitable[T]) -> T:
        return await awaitable

    done, pending = await asyncio.wait(
        [
            asyncio.Task(_wrap(awaitable), name="coro"),
            asyncio.Task(event.wait(), name="event"),
        ],
        return_when=asyncio.FIRST_COMPLETED,
    )
    task = done.pop()

    try:
        for future in pending:
            _ = future.cancel()
            await future
    except asyncio.CancelledError:
        pass

    if task.get_name() == "event":
        raise Bail

    return cast("T", task.result())


def signal_handler(event: asyncio.Event) -> None:
    LOGGER.info("Received CTRL+C")
    _ = asyncio.get_running_loop().call_soon_threadsafe(lambda: event.set())


def get_next_available_path(path: Path) -> Path:
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    pattern = f"{stem}_*{suffix}"
    existing = parent.glob(pattern)

    max_index = 0
    for f in existing:
        f_match = re.match(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$", f.name)
        if f_match:
            idx = int(f_match.group(1))
            max_index = max(max_index, idx)

    next_index = max_index + 1
    return parent / f"{stem}_{next_index}{suffix}"


def get_parsed_args() -> type[CLIArgs]:
    parser = argparse.ArgumentParser(description="llama.cpp Translation Experiment")
    _ = parser.add_argument(
        "--endpoint",
        default=ENDPOINT,
        help=f"OpenAI-like API endpoint URL (default: {ENDPOINT})",
    )
    _ = parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Model name to use (default: {MODEL_NAME})",
    )
    _ = parser.add_argument(
        "--embedding-model",
        default=EMBED_MODEL_NAME,
        help=f"Embedding model to use (default: {EMBED_MODEL_NAME})",
    )
    _ = parser.add_argument(
        "--rerank-model",
        default=RERANK_MODEL_NAME,
        help=f"Rerank model to use (default: {RERANK_MODEL_NAME})",
    )
    _ = parser.add_argument(
        "--inject-few-shot",
        default=False,
        action="store_true",
        help="Inject few-shot examples as multi-turn conversations",
    )
    _ = parser.add_argument(
        "--iterations",
        type=lambda x: min(int(x), MAX_N_ITERATIONS),
        default=DEFAULT_N_ITERATIONS,
        help=f"Number of iterations per temperature (default: {DEFAULT_N_ITERATIONS}, cap: {MAX_N_ITERATIONS})",
    )
    _ = parser.add_argument(
        "--refinement-iterations",
        type=lambda x: min(int(x), MAX_REFINEMENT_ITERATIONS),
        default=DEFAULT_REFINEMENT_ITERATIONS,
        help=f"Number of refinement iterations (default: {DEFAULT_REFINEMENT_ITERATIONS}, cap: {MAX_REFINEMENT_ITERATIONS})",
    )
    _ = parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSON file(s) containing the source text to translate. Use `,` to separate multiple files",
    )
    _ = parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT,
        help=f"Timeout for API requests in seconds (default: {TIMEOUT})",
    )
    _ = parser.add_argument(
        "--cache-prompt",
        action="store_true",
        default=False,
        help="Cache the prompt for faster subsequent requests",
    )
    _ = parser.add_argument(
        "--no-save",
        action="store_false",
        default=True,
        help="Do not save the output to a file.",
        dest="save_output",
    )
    _ = parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Run baseline inference",
    )
    _ = parser.add_argument(
        "--vectorise",
        action="store_true",
        default=False,
        help="Generate idiom vectors",
    )
    _ = parser.add_argument(
        "--match-idioms-only",
        action="store_true",
        default=False,
        help="Only match idioms without translating",
    )
    _ = parser.add_argument(
        "--evaluator-temperature",
        type=float,
        default=0.0,
        help="The temperature for the evaluation generation",
    )
    _ = parser.add_argument(
        "--optimiser-init-temperature",
        type=float,
        default=0.0,
        help="The temperature for the initial translation generation",
    )
    _ = parser.add_argument(
        "--optimiser-retry-temperature",
        type=float,
        default=0.0,
        help="The temperature for the translation refinement generation",
    )
    _ = parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Set logging level to DEBUG",
    )

    parsed = parser.parse_args(namespace=CLIArgs)
    LOGGER.info("Verbose logging: %s", parsed.verbose)
    LOGGER.info("Using endpoint: %s", parsed.endpoint)
    LOGGER.info("Model: %s", parsed.model)
    LOGGER.info(
        "Optimiser init temperature: %f",
        parsed.optimiser_init_temperature,
    )
    LOGGER.info(
        "Optimiser refinement temperature: %f",
        parsed.optimiser_retry_temperature,
    )
    LOGGER.info(
        "Evaluator temperature: %f",
        parsed.evaluator_temperature,
    )
    LOGGER.info("Embedding model: %s", parsed.embedding_model)
    LOGGER.info("Rerank model: %s", parsed.rerank_model)
    LOGGER.info("Iterations per seed: %d", parsed.iterations)
    LOGGER.info("Refinement iterations: %d", parsed.refinement_iterations)
    LOGGER.info("Input files: %s", parsed.input)
    LOGGER.info("Timeout: %d seconds", parsed.timeout)
    LOGGER.info("Cache prompt: %s", parsed.cache_prompt)
    LOGGER.info("Save output: %s", parsed.save_output)
    LOGGER.info("Baseline generation: %s", parsed.baseline)
    LOGGER.info("Generating vectors: %s", parsed.vectorise)
    LOGGER.info("Match idioms only: %s", parsed.match_idioms_only)
    return parsed
