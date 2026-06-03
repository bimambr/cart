"""
Copyright 2025 Muhammad Bima Ramadhan

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
from functools import partial
import json
import logging
import os
import pickle
import re
from collections.abc import Awaitable
from pathlib import Path
from typing import Protocol, TypeGuard, TypeVar, cast, final

import aiohttp
import torch
from sentence_transformers import SentenceTransformer, util

from _types import CLIArgs, IdiomEntry, IdiomMatchResult, Payload, StreamingResponse

LOGGER = logging.getLogger("lib")

T = TypeVar("T")


class Bail(Exception): ...


ENDPOINT = "http://localhost:8127/v1/chat/completions"
MODEL_NAME = "gemma-4-E2B-it-GGUF"
EMBED_MODEL_NAME = "./all-MiniLM-L6-v2"
DEFAULT_N_ITERATIONS = 5
MAX_N_ITERATIONS = 10
DEFAULT_REFINEMENT_ITERATIONS = 3
MAX_REFINEMENT_ITERATIONS = 5
TIMEOUT = 240
VECTORISED_DICTIONARY_PATH = "vectorised_dict.pkl"
IDIOM_MATCH_THRESHOLD = 0.55


class LoadedEmbedder(Protocol):
    model: str
    idiom_embeddings: dict[str, IdiomEntry]
    phrases: list[str]
    dict_embeddings: torch.Tensor
    embedding_model: SentenceTransformer

    def compute_similarities(
        self, chunks: list[str]
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]: ...


@final
class Embedder:
    def __init__(self, model: str) -> None:
        self.model = model

        self.idiom_embeddings: dict[str, IdiomEntry] | None = None
        self.phrases: list[str] | None = None
        self.dict_embeddings: torch.Tensor | None = None
        self.embedding_model: SentenceTransformer | None = None

    @staticmethod
    def is_loaded(obj: "Embedder") -> TypeGuard[LoadedEmbedder]:
        return (
            obj.idiom_embeddings is not None
            and obj.phrases is not None
            and obj.dict_embeddings is not None
            and obj.embedding_model is not None
        )

    def load_vectors(self) -> None:
        if not os.path.exists(VECTORISED_DICTIONARY_PATH):
            return

        if Embedder.is_loaded(self):
            return

        with open(VECTORISED_DICTIONARY_PATH, "rb") as f:
            _vector_data = pickle.load(f)  # pyright: ignore[reportAny]
            self.idiom_embeddings = cast(
                "dict[str, IdiomEntry]", _vector_data["dictionary"]
            )
            self.phrases = cast("list[str]", _vector_data["phrases"])
            self.dict_embeddings = cast(torch.Tensor, _vector_data["embeddings"])

        self.embedding_model = SentenceTransformer(self.model)

    @staticmethod
    def _get_sliding_windows(
        text: str, window_size: int = 6, overlap: int = 1
    ) -> list[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        step = max(1, window_size - overlap)
        windows: list[str] = []
        for i in range(0, len(words), step):
            chunk = words[i : i + window_size]
            if len(chunk) > overlap:
                windows.append(" ".join(chunk))
        return list(set(windows))

    def compute_similarities(
        self, chunks: list[str]
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        assert Embedder.is_loaded(self)
        chunk_embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True)  # pyright: ignore[reportUnknownMemberType]
        cosine_scores = util.cos_sim(chunk_embeddings, self.dict_embeddings)  # pyright: ignore[reportUnknownMemberType]
        match_coords = torch.where(cosine_scores >= IDIOM_MATCH_THRESHOLD)  # pyright: ignore[reportPrivateImportUsage]
        return match_coords, cosine_scores

    async def get_idiom_definitions(self, excerpt: str) -> list[IdiomMatchResult]:
        if not Embedder.is_loaded(self):
            return []

        if not (chunks := Embedder._get_sliding_windows(excerpt)):
            return []

        match_coordinates, cosine_scores = await asyncio.to_thread(
            partial(self.compute_similarities, chunks)
        )
        results: list[IdiomMatchResult] = []
        found_idiom_keys: set[str] = set()
        for chunk_idx, idiom_idx in zip(match_coordinates[0], match_coordinates[1]):
            idiom_key = self.phrases[idiom_idx]

            if idiom_key not in found_idiom_keys:
                score = cosine_scores[chunk_idx][idiom_idx].item()
                entry = self.idiom_embeddings[idiom_key]

                results.append(
                    {
                        "idiom": idiom_key,
                        "senses": entry.get("senses", []),
                        "translations": entry.get("translations", {}),
                        "matched_chunk": chunks[chunk_idx],
                        "score": round(score, 3),
                    }
                )
                found_idiom_keys.add(idiom_key)

        results.sort(key=lambda x: x["score"], reverse=True)
        LOGGER.info("Found idiom matches: \n%s", results)
        return results

    def generate_vectors(self) -> None:
        normalised_dict: dict[str, IdiomEntry] = {}

        with open("idiom_dict/cherrypicked.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)  # pyright: ignore[reportAny]
            for phrase, data in json_data.items():  # pyright: ignore[reportAny]
                normalised_dict[phrase] = IdiomEntry(
                    idiom=cast(str, phrase),
                    senses=data.get("senses", []),  # pyright: ignore[reportAny]
                    translations={},
                )

        with open("idiom_dict/idiomKB.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)  # pyright: ignore[reportAny]
            for entry in json_data:  # pyright: ignore[reportAny]
                phrase = entry.get("idiom")  # pyright: ignore[reportAny]
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
                else:
                    normalised_dict[phrase] = IdiomEntry(
                        idiom=phrase,
                        senses=[en_meaning] if en_meaning else [],
                        translations=translations,
                    )

        phrases = list(normalised_dict.keys())
        LOGGER.info("Loading model and computing %d embeddings", len(normalised_dict))
        model = SentenceTransformer(self.model)
        embeddings_tensor = model.encode(phrases, convert_to_tensor=True)  # pyright: ignore[reportUnknownMemberType]

        with open(VECTORISED_DICTIONARY_PATH, "wb") as f:
            pickle.dump(
                {
                    "dictionary": normalised_dict,
                    "phrases": phrases,
                    "embeddings": embeddings_tensor,
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
                "thinking_budget": "high",
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
        "--omit-roles",
        action="store_true",
        default=False,
        help="Omit roles in system prompts",
    )
    _ = parser.add_argument(
        "--preserve-last-n-messages",
        type=int,
        default=0,
        help="Preserve last N messages of interaction history when optimizing translations (-1: all, 0: none).",
        dest="keep_n_messages",
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

    parsed = parser.parse_args(namespace=CLIArgs)
    LOGGER.info("Using endpoint: %s", parsed.endpoint)
    LOGGER.info("Model: %s", parsed.model)
    LOGGER.info("Embedding model: %s", parsed.embedding_model)
    LOGGER.info("Iterations per seed: %d", parsed.iterations)
    LOGGER.info("Refinement iterations: %d", parsed.refinement_iterations)
    LOGGER.info("Input files: %s", parsed.input)
    LOGGER.info("Timeout: %d seconds", parsed.timeout)
    LOGGER.info("Cache prompt: %s", parsed.cache_prompt)
    LOGGER.info("Omit roles: %s", parsed.omit_roles)
    LOGGER.info("Preserve last N messages: %d", parsed.keep_n_messages)
    LOGGER.info("Save output: %s", parsed.save_output)
    LOGGER.info("Baseline generation: %s", parsed.baseline)
    LOGGER.info("Generating vectors: %s", parsed.vectorise)
    LOGGER.info("Match idioms only: %s", parsed.match_idioms_only)
    return parsed
