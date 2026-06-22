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
import csv
import json
import logging
import re
import signal
import time
from collections.abc import Sequence
from io import TextIOWrapper
from pathlib import Path
from types import TracebackType
from typing import Final, Literal, cast, overload

import aiohttp

from _types import (
    CSVWriter,
    Corpus,
    IdiomEntry,
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
    run_inference,
    signal_handler,
    wait,
)

EVALUATOR_SEED = 727
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
ARGS = get_parsed_args()


def format_external_knowledge(external_knowledge: list[str]) -> str:
    if not external_knowledge:
        return ""

    nl = "\n"
    return f"""
External retrieved knowledge:
{nl.join([f"- {i}" for i in external_knowledge])}
"""


def format_idiom_knowledge(idioms: Sequence[IdiomEntry]) -> str:
    nl = "\n"
    if not idioms:
        return ""

    def format_senses(senses: list[str]) -> str:
        return "\n".join([f"  {j}. {k} " for j, k in enumerate(senses, start=1)])

    def format_translations(translations: dict[str, str]) -> str:
        if not translations:
            return ""

        return "\n\n    Example translations:\n" + "\n".join(
            [f"        {k}: {v}" for k, v in translations.items()]
        )

    return f"""
Known idiom definitions:
{nl.join([f"- {i['idiom']}:{nl}{format_senses(i['senses'])}{format_translations(i['translations'])}" for i in idioms])}
"""


def format_context(state: State) -> str:
    source_text = state["source_text"]
    return f"""
Text type: {source_text["type"]}

Translation direction: from {source_text["source_lang"]} to {source_text["target_lang"]}

{format_external_knowledge(source_text.get("external_knowledge", []))}

{format_idiom_knowledge(source_text.get("idiom_matches", []))}
""".strip()


def format_rubric(rubric: Rubric) -> str:
    return f"""
- accuracy: {rubric["accuracy"]["score"]}. {rubric["accuracy"]["feedback"]}
- acceptability: {rubric["acceptability"]["score"]}. {rubric["acceptability"]["feedback"]}
- readability: {rubric["readability"]["score"]}. {rubric["readability"]["feedback"]}
""".strip()


IDIOM_EXTRACTION_GRAMMAR = r"""
root        ::= "[" string-list "]"
string-list ::= (string (", " string)*)?
string      ::= "\"" [^"\\]* "\""   
""".strip()


IDIOM_EXTRACTION_SYSTEM_PROMPT = """
Extract idioms, fixed metaphorical phrases, or non-compositional expressions present within the provided text as a JSON list. Do not explain. No code blocks.

Constraints:
1. Extract the phrase exactly as it appears in the text.
2. Broaden your criteria: include physical expressions used metaphorically and idiomatic word pairings.
3. Bias toward over-extraction. If a phrase is even slightly figurative or non-literal, extract it. The downstream system will handle filtering; it is critical that you do not miss any candidate expressions.
4. Output a valid flat JSON array of strings: ["extracted_phrase_1", "extracted_phrase_2"]
5. If absolutely no figurative expressions are present, output exactly: []
6. Provide NO explanations and NO conversational filler.
"""


OPTIMISER_SYSTEM_PROMPT = """
Translate the provided source text into natural, idiomatic Indonesian using the given external knowledge and evaluation history. You must prioritize figurative accuracy, proper narrative register, and contextual phrasing over literal word-for-word translation. Incorporate any corrections or suggested alternatives from the interaction history seamlessly during refinement turns.
""".strip()


EVALUATOR_SYSTEM_PROMPT = """
Evaluate the provided translation against the original source text strictly across accuracy, acceptability, and readability. Be highly critical of idiomatic nuances, register, and narrative context; do not default to a perfect score of 3 if any literal calques, flat phrasing, or contextual mismatches are present.

Mandatory Instruction: If you assign a score of 1 or 2 to any metric, you must explicitly include 1-2 highly natural, contextually accurate Indonesian alternatives at the end of that metric's feedback text. Do not just describe the error conceptually; provide the exact phrasing the optimiser should use. If the score is 3, no suggestions are required.

Format your output exactly as follows:
- accuracy: <score 1-3>. <detailed explanation of error>. Suggested alternatives: <concrete Indonesian phrasing>
- acceptability: <score 1-3>. <detailed explanation of error>. Suggested alternatives: <concrete Indonesian phrasing>
- readability: <score 1-3>. <detailed explanation of error>. Suggested alternatives: <concrete Indonesian phrasing>
""".strip()


OPTIMISER_INIT_PROMPT = """
{CONTEXT}

Instruction: Translate the Source Text naturally. Look at the 'Known idiom definitions' provided above. If any of those phrases appear in the Source Text, you must apply their defined meaning. Do not translate them literally word-for-word.

Critical Grounding Constraint: Before translating, identify the exact physical action, dialogue topic, or narrative event occurring immediately around any identified idioms. You must translate the figurative meaning so that it anchors perfectly to that specific situational context, rather than defaulting to a generic or abstract dictionary definition.

Source text: {SOURCE_TEXT}

Provide exactly one translation:
""".strip()


EVALUATOR_INIT_PROMPT = """
{CONTEXT}

Instruction: Grade the Translation against the Source Text following the rubric format. 

Critical Grounding Constraint: Before evaluating an idiom, identify the exact physical action or event occurring in the narrative immediately preceding the phrase. The idiom must be evaluated based on how it applies to that specific, immediate event—not as an abstract or generic dictionary definition. 

If the translation uses a literal calque or selects the wrong contextual sense, you must penalize both Accuracy and Acceptability. If you assign a score of 1 or 2, you must provide highly natural, context-specific Indonesian alternatives inside the feedback block.

Source text: {SOURCE_TEXT}

Translation: {TRANSLATION_ATTEMPT}

Provide the grades following the rubric:
""".strip()


OPTIMISER_RETRY_PROMPT = """
Grades:
{GRADES}

Provide your revision in the exact output:
Planned changes:
- <brief reasoning>

Revision: <the complete updated translation block containing all sentences>
""".strip()


EVALUATOR_RETRY_PROMPT = """
Revision: {TRANSLATION_ATTEMPT}

Regrade my revision following the rubric:
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
        assert "raw_output" in s
        messages.append(("user", s["prompt"], "user"))
        messages.append(("assistant", s["raw_output"], "assistant"))

    messages.append(("user", user_prompt, "user"))
    return messages


def parse_rubric(text: str) -> Rubric:
    rubric: Rubric = {
        "accuracy": {"score": 0, "feedback": "Missing feedback."},
        "acceptability": {"score": 0, "feedback": "Missing feedback."},
        "readability": {"score": 0, "feedback": "Missing feedback."},
    }

    pattern = re.compile(
        r"-?\s*\*?\*?(accuracy|acceptability|readability)\*?\*?\s*:\s*\*?\*?(\d+)(?:\.|\b)\*?\*?\s*([\s\S]*?)(?=-\s*\*?\*?(?:accuracy|acceptability|readability)\b|\Z)",
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


async def handle_baseline_state(state: State) -> None:
    state["next_state"] = ""
    state["attempt"] += 1

    LOGGER.info(
        "Generating baseline translation for text %d", state["source_text"]["id"]
    )
    prompt = f"Provide exactly one translation of the following text into {state['source_text']['target_lang']}:\n{state['source_text']['text']}\n\nTranslation:\n"
    temp = ARGS.optimiser_init_temperature
    seed = state["optimiser_seed"]
    output = (
        await run_inference(
            state["client"],
            ARGS.endpoint,
            ARGS.model,
            temp,
            seed,
            timeout=ARGS.timeout,
            cache_prompt=ARGS.cache_prompt,
            messages=[("user", prompt, "user")],
        )
    ).strip()
    state["history"].append(
        {
            "type": "attempt",
            "translation": output,
            "raw_output": output,
            "prompt": prompt,
            "seed": seed,
            "temp": temp,
        }
    )

    if csv_writer := state.get("csv_writer"):
        csv_writer.writerow(
            (
                state["source_text"]["id"],
                state["iteration_id"],
                state["attempt"],
                seed,
                temp,
                0,
                0,
                state["source_text"]["text"],
                output,
                {},
                output,
                "N/A",
                time.ctime(),
                "N/A",
                prompt,
                "N/A",
                "N/A",
            )
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
    if is_draft:
        prompt = OPTIMISER_INIT_PROMPT.format(
            SOURCE_TEXT=state["source_text"]["text"], CONTEXT=context
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
    messages = build_messages(state, OPTIMISER_SYSTEM_PROMPT, prompt)
    output = (
        await run_inference(
            state["client"],
            ARGS.endpoint,
            ARGS.model,
            temp,
            seed,
            timeout=ARGS.timeout,
            cache_prompt=ARGS.cache_prompt,
            messages=messages,
        )
    ).strip()
    match = re.search(r"Revision:\s*(.*)", output, re.IGNORECASE | re.DOTALL)
    translation = match.group(1).strip() if match else output.strip()
    state["history"].append(
        {
            "type": "attempt",
            "translation": translation,
            "raw_output": output,
            "prompt": prompt,
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
    assert "translation" in last_attempt

    prompt = (EVALUATOR_RETRY_PROMPT if is_retrying else EVALUATOR_INIT_PROMPT).format(
        SOURCE_TEXT=state["source_text"]["text"],
        TRANSLATION_ATTEMPT=last_attempt["translation"],
        CONTEXT=format_context(state),
    )
    messages = build_messages(state, EVALUATOR_SYSTEM_PROMPT, prompt)
    output = (
        await run_inference(
            state["client"],
            ARGS.endpoint,
            ARGS.model,
            ARGS.evaluator_temperature,
            seed,
            timeout=ARGS.timeout,
            cache_prompt=ARGS.cache_prompt,
            messages=messages,
        )
    ).strip()
    rubric = parse_rubric(output.strip())
    evaluation: TranslationEvaluation = {
        "type": "evaluation",
        "prompt": prompt,
        "seed": seed,
        "temp": ARGS.evaluator_temperature,
        "rubric": rubric,
        "raw_output": output,
    }
    state["history"].append(evaluation)

    if csv_writer := state.get("csv_writer"):
        assert "translation" in last_attempt
        assert "raw_output" in last_attempt
        assert "rubric" in evaluation
        assert "raw_output" in evaluation

        csv_writer.writerow(
            (
                state["source_text"]["id"],
                state["iteration_id"],
                state["attempt"],
                last_attempt.get("seed", -1),
                last_attempt.get("temp", -1),
                seed,
                ARGS.evaluator_temperature,
                state["source_text"]["text"],
                last_attempt["translation"],
                evaluation["rubric"],
                last_attempt["raw_output"],
                evaluation["raw_output"],
                time.ctime(),
                last_attempt.get("system_prompt", "Not available."),
                last_attempt.get("prompt", "Not available."),
                "",
                prompt,
            )
        )

    state["next_state"] = "optimisation"
    if (
        sum(rubric[i]["score"] for i in ("accuracy", "acceptability", "readability"))
        == 9
        or state["attempt"] >= state["max_attempt"]
    ):
        state["next_state"] = ""


class FileProcessor:
    CSV_HEADER: tuple[str, ...] = (
        "text_id",
        "iteration_id",
        "attempt",
        "optimiser_seed",
        "optimiser_temp",
        "evaluator_seed",
        "evaluator_temp",
        "source_text",
        "translation_attempt",
        "grade",
        "raw_translation",
        "raw_evaluation",
        "timestamp",
        "optimiser_system_prompt",
        "optimiser_user_prompt",
        "evaluator_system_prompt",
        "evaluator_user_prompt",
    )

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

        self.csv_file: TextIOWrapper | None = None
        self.csv_writer: CSVWriter | None = None
        self.log_file: TextIOWrapper | None = None

        self.client: aiohttp.ClientSession = client
        self.embedder: Embedder = embedder

    def open(self) -> None:
        if not ARGS.save_output:
            return

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_file:
            self.csv_file = open(self.output_file, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(self.CSV_HEADER)
            LOGGER.info("Output will be saved to: %s", self.output_file)

        if not self.log_file:
            self.log_file = open(
                self.output_file.with_suffix(".jsonl"), "w", encoding="utf-8"
            )

    async def __aenter__(self) -> "FileProcessor":
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self.csv_writer = None

    async def _get_idiom_matches(self, excerpt: str) -> list[IdiomMatchResult]:
        if ARGS.baseline:
            return []

        output = await run_inference(
            self.client,
            ARGS.endpoint,
            ARGS.model,
            0.0,
            SEEDS[0],
            0,
            IDIOM_EXTRACTION_GRAMMAR,
            True,
            False,
            [
                ("system", IDIOM_EXTRACTION_SYSTEM_PROMPT, "system"),
                ("user", excerpt, "user"),
            ],
        )

        if not (extracted_phrases := cast("list[str]", json.loads(output))):
            return []

        LOGGER.info("Extracted idioms: %s", extracted_phrases)
        return await self.embedder.get_idiom_definitions(excerpt, extracted_phrases)

    async def process(self) -> None:
        if not self.input_file.exists():
            LOGGER.error("Input file '%s' does not exist.", self.input_file)
            return

        LOGGER.info("Processing input file: %s", self.input_file)

        self.open()

        input_json = cast(
            Corpus, json.loads(self.input_file.read_text("utf-8").strip())
        )

        if not ARGS.baseline:
            self.embedder.load_vectors()

        for text_idx, text in enumerate(input_json["texts"]):
            LOGGER.info(
                "--- Translating text %d out of %d ---",
                text_idx + 1,
                len(input_json["texts"]),
            )

            if ARGS.match_idioms_only:
                LOGGER.info("Source text: %s", text["content"])

            source_text: SourceTextEntry = {
                "source_lang": input_json["source_lang"],
                "target_lang": input_json["target_lang"],
                "text": text["content"],
                "type": input_json.get("type", "general"),
                "id": text_idx + 1,
                "external_knowledge": input_json.get("external_knowledge", [])
                + text.get("external_knowledge", []),
                "idiom_matches": await self._get_idiom_matches(text["content"]),
            }

            await self._process_text(source_text)

    async def _process_text(self, source_text: SourceTextEntry) -> None:
        if ARGS.match_idioms_only:
            if self.log_file:
                _ = self.log_file.write(
                    json.dumps(source_text, ensure_ascii=False, indent=4) + "\n"
                )
                self.log_file.flush()
            return

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
                next_state="baseline" if ARGS.baseline else "optimisation",
                max_attempt=ARGS.refinement_iterations,
                attempt=0,
                history=[],
                optimiser_seed=SEEDS[i],
                evaluator_seed=EVALUATOR_SEED,
                client=self.client,
                csv_writer=self.csv_writer,
            )

            while handler := self.STATE_HANDLERS.get(state["next_state"]):
                await handler(state)
                _ = self.csv_file and self.csv_file.flush()

            if self.log_file:
                loggable_state = {
                    k: v for k, v in state.items() if k not in ("client", "csv_writer")
                }
                _ = self.log_file.write(
                    json.dumps(loggable_state, ensure_ascii=False, indent=4) + "\n"
                )
                self.log_file.flush()


async def main():
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
            / ("baseline_attempts" if ARGS.baseline else "evaluator_optimiser_attempts")
            / f"{p.stem}_translated_{ARGS.model}_attempt.csv"
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

    if ARGS.save_output:
        LOGGER.info(
            "Experiment complete. Results saved to %s.",
            ", ".join([str(i) for i in output_files]),
        )


if __name__ == "__main__":
    asyncio.run(main())
