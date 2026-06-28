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

import asyncio
import json
import logging
import re
import signal
from collections.abc import Sequence
from io import TextIOWrapper
from pathlib import Path
from types import TracebackType
from typing import Final, Literal, cast, overload

import aiohttp

from _types import (
    Corpus,
    IdiomMatchResult,
    Rubric,
    SourceTextEntry,
    State,
    TranslationAttempt,
    TranslationEvaluation,
)
from lib import (
    Bail,
    Embedder,
    get_next_available_path,
    get_parsed_args,
    log_args,
    run_inference,
    signal_handler,
    wait,
)

EVALUATOR_SEED = 727
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]


LOGGER = logging.getLogger(__name__)
ARGS = get_parsed_args()

logging.basicConfig(level=logging.DEBUG if ARGS.verbose else logging.INFO)


def format_idiom_knowledge(idioms: Sequence[IdiomMatchResult]) -> str:
    nl = "\n"
    if not idioms:
        return ""

    def format_senses(senses: list[str]) -> str:
        return "\n".join([f"  {j}. {k} " for j, k in enumerate(senses, start=1)])

    disclaimer = (
        "=== POTENTIAL IDIOM SUGGESTIONS ===\n"
        "The following are possible meanings of matched phrases. Use them only if they are consistent with the surrounding context."
    )
    entries = nl.join(
        [
            f"- Phrase form: {i['idiom']}\n  Senses:\n{format_senses(i['senses'])}"
            for i in idioms
        ]
    )
    return f"\n{disclaimer}\n\n{entries}\n===================================="


def format_context(state: State) -> str:
    source_text = state["source_text"]
    return f"""
Text type: {source_text["type"]}

{format_idiom_knowledge(source_text.get("idiom_matches", []))}
""".strip()


def format_rubric(rubric: Rubric) -> str:
    return f"""
- accuracy: {rubric["accuracy"]["score"]}. {rubric["accuracy"]["feedback"]}
- acceptability: {rubric["acceptability"]["score"]}. {rubric["acceptability"]["feedback"]}
- readability: {rubric["readability"]["score"]}. {rubric["readability"]["feedback"]}
""".strip()


TRANSLATOR_SYSTEM_PROMPT = """
You are an expert literary translator specialising in dynamic equivalence. 

Your goal is to convey the psychological subtext, tone, and idiomatic impact of the source text so that it reads as an original work in the target language. Avoid literal translations, syntactic calques, and word-for-word substitutions of idioms.
""".strip()


TRANSLATOR_GRAMMAR = """
root     ::= <|channel> "thought" thinking <channel|> "Translation: " .*
thinking ::= !<channel|>*
"""


OPTIMISER_INIT_PROMPT = """
Translate the following text into {TARGET_LANG}. Provide the translation text alone, without any introductory phrases, alternative options, or post-translation notes.

Text:
{SOURCE_TEXT}

{CONTEXT}
""".strip()


OPTIMISER_RETRY_PROMPT = """
You are given grades on a 1–3 scale and feedback regarding your translation. Revise accordingly and provide the revised translation text alone, without any introductory phrases, alternative options, or post-translation notes.

Feedback:
{GRADES}
""".strip()


EVALUATOR_SYSTEM_PROMPT = """
You are a strict translation evaluator.

Evaluate based on these strict definitions:
- accuracy: Is the psychological and contextual meaning preserved? (Penalty if a figurative phrase is translated literally, altering its implied meaning).
- acceptability: Does the translation sound like natural prose written native-to-native, or does it sound like "translationese" (English syntax/idioms masquerading as target language words)?
- readability: Flow, rhythm, and coherence.

Each aspect uses an ordinal scale of 1 to 3 (the greater the better).
""".strip()


EVALUATOR_GRAMMAR = r"""
root    ::= <|channel> "thought" thinking <channel|> "- accuracy: " score ". " reason "\n- acceptability: " score ". " reason "\n- readability: " score ". " reason "\n"?
score   ::= [1-3]
reason  ::= [^\n]+
thinking ::= !<channel|>*
"""


EVALUATOR_INIT_PROMPT = """
Evaluate the translation using the rubric format.

Text:
{SOURCE_TEXT}

Translation:
{TRANSLATION_ATTEMPT}

{CONTEXT}
""".strip()


EVALUATOR_RETRY_PROMPT = """
Consider the following revision and regrade it using the rubric format.

Revision: {TRANSLATION_ATTEMPT}
""".strip()


@overload
def get_last_state(
    state: State, type: Literal["attempt"]
) -> TranslationAttempt | None: ...


@overload
def get_last_state(
    state: State, type: Literal["evaluation"]
) -> TranslationEvaluation | None: ...


def get_last_state(
    state: State, type: Literal["attempt"] | Literal["evaluation"]
) -> TranslationEvaluation | TranslationAttempt | None:
    for entry in state["history"][::-1]:
        assert "type" in entry
        if entry["type"] == type:
            return entry
    return None


def build_messages(
    state: State, system_prompt: str, user_prompt: str
) -> list[tuple[str, str, str]]:
    messages: list[tuple[str, str, str]] = []

    if system_prompt:
        messages.append(("system", system_prompt, "system"))

    history = state["history"]
    is_evaluating = history and history[-1].get("type") == "attempt"

    for s in [
        s
        for s in history
        if s.get("type") == ("evaluation" if is_evaluating else "attempt")
    ]:
        assert "prompt" in s
        assert "raw_content" in s
        messages.append(("user", s["prompt"], "user"))
        messages.append(("assistant", s["raw_content"], "assistant"))

    messages.append(("user", user_prompt, "user"))
    return messages


def parse_rubric(text: str) -> Rubric:
    text = re.sub(
        r"^Grades:\s*",
        "",
        text,
    )
    rubric: Rubric = {
        "accuracy": {"score": 0, "feedback": "Missing feedback."},
        "acceptability": {"score": 0, "feedback": "Missing feedback."},
        "readability": {"score": 0, "feedback": "Missing feedback."},
    }
    pattern = re.compile(
        r"\*{0,2}\s*(accuracy|acceptability|readability)\s*:\s*\*{0,2}\s*(\d+)\s*\*{0,2}\s*\.*\s*"
        + r"([\s\S]*?)"
        + r"(?=\*{0,2}\s*(?:accuracy|acceptability|readability)\s*:|\Z)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        key, score, feedback = match.groups()
        normalised_key = key.lower()

        rubric[normalised_key] = {
            "score": int(score),
            "feedback": feedback.strip(),
        }
    return rubric


def parse_translation(text: str) -> str:
    match = re.search(
        r"Translation:\s*(.*)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else text.strip()


async def handle_baseline_state(state: State) -> None:
    state["next_state"] = ""
    state["attempt"] += 1

    LOGGER.info(
        "Generating baseline translation for text %d", state["source_text"]["id"]
    )
    system_prompt = TRANSLATOR_SYSTEM_PROMPT if ARGS.treatment_level > 1 else ""
    # baseline borrows optimiser init prompt with an empty context
    prompt = OPTIMISER_INIT_PROMPT.format(
        TARGET_LANG=state["source_text"]["target_lang"],
        SOURCE_TEXT=state["source_text"]["text"],
        CONTEXT="",
    ).strip()
    temp = ARGS.optimiser_init_temperature
    seed = state["optimiser_seed"]
    reasoning, content = await run_inference(
        state["client"],
        ARGS.endpoint,
        ARGS.model,
        temp,
        seed,
        timeout=ARGS.timeout,
        cache_prompt=ARGS.cache_prompt,
        grammar=TRANSLATOR_GRAMMAR,
        messages=([("system", system_prompt, "system")] if system_prompt else [])
        + [("user", prompt, "user")],
    )
    state["history"].append(
        {
            "type": "attempt",
            "translation": parse_translation(content) or content,
            "raw_content": content,
            "raw_reasoning": reasoning,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "seed": seed,
            "temp": temp,
        }
    )


async def handle_optimisation_state(state: State) -> None:
    state["attempt"] += 1
    state["next_state"] = "evaluation"

    is_draft = state["attempt"] == 1

    LOGGER.info(
        "Starting %s for text %d, iteration %d/%d, attempt %d/%d",
        "draft generation" if is_draft else "refinement",
        state["source_text"]["id"],
        state["iteration_id"],
        ARGS.iterations,
        state["attempt"],
        state["max_attempt"],
    )

    context = format_context(state)
    system_prompt = TRANSLATOR_SYSTEM_PROMPT
    if is_draft:
        prompt = OPTIMISER_INIT_PROMPT.format(
            SOURCE_TEXT=state["source_text"]["text"],
            CONTEXT=context,
            TARGET_LANG=state["source_text"]["target_lang"],
        )
    else:
        last_attempt = get_last_state(state, "attempt") or {}
        last_evaluation = get_last_state(state, "evaluation")

        # THIS SHOULD NEVER HAPPEN
        if not last_evaluation:
            LOGGER.error(
                "No evaluation found from previous attempts for text %d, iteration %d/%d, attempt %d/%d. Cannot proceed with refinement.",
                state["source_text"]["id"],
                state["iteration_id"],
                ARGS.iterations,
                state["attempt"],
                state["max_attempt"],
            )
            state["next_state"] = ""
            return

        assert "translation" in last_attempt
        assert "rubric" in last_evaluation
        prompt = OPTIMISER_RETRY_PROMPT.format(
            GRADES=format_rubric(last_evaluation["rubric"])
        )

    temp = (
        ARGS.optimiser_init_temperature
        if is_draft
        else ARGS.optimiser_retry_temperature
    )
    seed = state["optimiser_seed"] * 10 + state["attempt"]
    messages = build_messages(state, system_prompt, prompt)
    reasoning, content = await run_inference(
        state["client"],
        ARGS.endpoint,
        ARGS.model,
        temp,
        seed,
        timeout=ARGS.timeout,
        grammar=TRANSLATOR_GRAMMAR,
        cache_prompt=ARGS.cache_prompt,
        messages=messages,
    )
    state["history"].append(
        {
            "type": "attempt",
            "translation": parse_translation(content),
            "raw_content": content,
            "raw_reasoning": reasoning,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "seed": seed,
            "temp": temp,
        }
    )


async def handle_evaluation_state(state: State) -> None:
    LOGGER.info(
        "Starting evaluation for text %d, iteration %d/%d, attempt %d/%d",
        state["source_text"]["id"],
        state["iteration_id"],
        ARGS.iterations,
        state["attempt"],
        state["max_attempt"],
    )

    is_retrying = (
        len([s for s in state["history"] if s.get("type") == "evaluation"]) > 0
    )

    # do not mutate the original evaluator seed
    seed = state["evaluator_seed"] + state["iteration_id"] * 100
    last_attempt = state["history"][-1]

    assert last_attempt.get("type") == "attempt"
    assert "raw_content" in last_attempt

    system_prompt = EVALUATOR_SYSTEM_PROMPT
    prompt = (EVALUATOR_RETRY_PROMPT if is_retrying else EVALUATOR_INIT_PROMPT).format(
        SOURCE_TEXT=state["source_text"]["text"],
        TRANSLATION_ATTEMPT=re.sub(
            r"^Translation:\s*", "", last_attempt["raw_content"]
        ),
        CONTEXT=format_context(state),
    )
    temp = ARGS.evaluator_temperature
    messages = build_messages(state, system_prompt, prompt)
    reasoning, content = await run_inference(
        state["client"],
        ARGS.endpoint,
        ARGS.model,
        temp,
        seed,
        timeout=ARGS.timeout,
        cache_prompt=ARGS.cache_prompt,
        grammar=EVALUATOR_GRAMMAR,
        messages=messages,
    )
    rubric = parse_rubric(content)
    evaluation: TranslationEvaluation = {
        "type": "evaluation",
        "prompt": prompt,
        "system_prompt": system_prompt,
        "seed": seed,
        "temp": temp,
        "rubric": rubric,
        "raw_content": content,
        "raw_reasoning": reasoning,
    }
    state["history"].append(evaluation)
    state["next_state"] = "optimisation"
    if (
        sum(rubric[i]["score"] for i in ("accuracy", "acceptability", "readability"))
        == 9
        or state["attempt"] >= state["max_attempt"]
    ):
        state["next_state"] = ""


class FileProcessor:
    STATE_HANDLERS: Final = {
        "baseline": handle_baseline_state,
        "optimisation": handle_optimisation_state,
        "evaluation": handle_evaluation_state,
    }

    def __init__(
        self,
        id: int,
        input_file: Path,
        output_file: Path,
        embedder: Embedder,
        client: aiohttp.ClientSession,
    ) -> None:
        self.id: int = id
        self.input_file: Path = input_file
        self.output_file: Path = output_file
        self.client: aiohttp.ClientSession = client
        self.embedder: Embedder = embedder

        self.log_file: TextIOWrapper | None = None
        self.hints_file: Path = Path("hints") / self.input_file.name
        self.hints: list[list[IdiomMatchResult]] | None = None

    def open(self) -> None:
        if not ARGS.save_output or ARGS.generate_hints:
            return

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.log_file:
            self.log_file = open(self.output_file, "w", encoding="utf-8")

    async def __aenter__(self) -> "FileProcessor":
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    async def process(self) -> None:
        if not self.input_file.exists():
            LOGGER.error("Input file '%s' does not exist.", self.input_file)
            return

        LOGGER.info("Processing input file: %s", self.input_file)

        self.open()

        input_json = cast(
            Corpus, json.loads(self.input_file.read_text("utf-8").strip())
        )

        if ARGS.generate_hints:
            await self._run_hints_generation_pass(input_json)
            return

        await self._run_translation_pass(input_json)

    async def _run_hints_generation_pass(self, input_json: Corpus) -> None:
        LOGGER.info("--- Running hints generation pass (1/2) ---")

        self.embedder.load()

        hints = []
        texts = input_json["texts"]
        for text_idx, text in enumerate(texts):
            LOGGER.info("Generating hints for text %d/%d", text_idx + 1, len(texts))
            LOGGER.info("Source text: %s", text["content"])
            hints.append(await self.embedder.get_idiom_definitions(text["content"]))

        self.hints_file.parent.mkdir(parents=True, exist_ok=True)
        self.hints_file.write_text(json.dumps(hints, indent=4), encoding="utf-8")
        LOGGER.info("Successfully cached hints to %s", self.hints_file)

    async def _run_translation_pass(self, input_json: Corpus) -> None:
        texts = input_json["texts"]

        if ARGS.treatment_level > 2:
            if self.hints_file.exists():
                self.hints = json.loads(self.hints_file.read_text("utf-8"))
                if len(self.hints) != len(texts):
                    LOGGER.warning("Hints file size mismatch!")
                    self.hints = None

            if self.hints is None:
                self.embedder.load()

        LOGGER.info(
            "--- Running translation pass %s ---",
            "using cached hints (2/2) " * (self.hints is not None),
        )

        for text_idx, text in enumerate(input_json["texts"]):
            LOGGER.info(
                "--- Translating text %d out of %d ---",
                text_idx + 1,
                len(input_json["texts"]),
            )
            LOGGER.info("Source text: %s", text["content"])

            if ARGS.treatment_level > 2:
                idiom_matches = (
                    self.hints[text_idx]
                    if self.hints
                    else await self.embedder.get_idiom_definitions(text["content"])
                )
            else:
                idiom_matches = []

            source_text: SourceTextEntry = {
                "source_lang": input_json["source_lang"],
                "target_lang": input_json["target_lang"],
                "text": text["content"],
                "type": input_json.get("type", "general"),
                "id": text_idx + 1,
                "idiom_matches": idiom_matches,
            }

            await self._process_text(source_text)

    async def _process_text(self, source_text: SourceTextEntry) -> None:
        for i in range(ARGS.iterations):
            iteration_num = i + 1
            LOGGER.info(
                "=== Iteration %d out of %d ===",
                iteration_num,
                ARGS.iterations,
            )

            state = State(
                iteration_id=iteration_num,
                source_text=source_text,
                next_state="optimisation" if ARGS.treatment_level > 2 else "baseline",
                max_attempt=ARGS.refinement_iterations,
                attempt=0,
                history=[],
                optimiser_seed=SEEDS[i],
                evaluator_seed=EVALUATOR_SEED,
                client=self.client,
            )

            while handler := self.STATE_HANDLERS.get(state["next_state"]):
                await handler(state)

            if self.log_file:
                loggable_state = {k: v for k, v in state.items() if k not in ("client")}
                _ = self.log_file.write(
                    json.dumps(loggable_state, ensure_ascii=False, indent=4) + "\n"
                )
                self.log_file.flush()


async def main():
    log_args(ARGS)

    LOGGER.info("Starting translation experiment...")

    embedder = Embedder(ARGS.embedding_model, ARGS.rerank_model)

    if ARGS.vectorise:
        embedder.generate_vectors()
        exit(0)

    input_files = [Path(p) for p in ARGS.input.split(",")]
    root = Path(__file__).parent
    output_files = [
        get_next_available_path(
            root
            / (
                "baseline_attempts"
                if ARGS.treatment_level == 1
                else "baseline_with_persona_attempts"
                if ARGS.treatment_level == 2
                else "evaluator_optimiser_attempts"
            )
            / f"{p.stem}_translated_{ARGS.model}_attempt.jsonl"
        )
        for p in input_files
    ]

    event = asyncio.Event()
    signal.signal(signal.SIGINT, lambda *_args: signal_handler(event))  # pyright: ignore[reportUnusedCallResult, reportUnknownArgumentType, reportUnknownLambdaType]

    async with aiohttp.ClientSession() as client:
        for file_idx, (input_file, output_file) in enumerate(
            zip(input_files, output_files)
        ):
            try:
                async with FileProcessor(
                    id=file_idx,
                    input_file=input_file,
                    output_file=output_file,
                    client=client,
                    embedder=embedder,
                ) as processor:
                    await wait(
                        processor.process(),
                        event,
                    )

            except IOError as e:
                LOGGER.error(
                    "Could not write to file %s. Reason: %s", output_file, e, exc_info=e
                )
                return
            except Bail:
                LOGGER.info("Experiment interrupted by user.")
                return

    if ARGS.save_output and not ARGS.generate_hints:
        LOGGER.info(
            "Experiment complete. Results saved to %s.",
            ", ".join([str(i) for i in output_files]),
        )


if __name__ == "__main__":
    asyncio.run(main())
