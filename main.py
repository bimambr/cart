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
    ExampleEntry,
    IdiomEntry,
    Rubric,
    RubricEntry,
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

EVALUATOR_TEMP = 0.7
OPTIMIZER_TEMP = 0.7
OPTIMIZER_ALT_TEMP = 0.7
EVALUATOR_SEED = 727
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
ARGS = get_parsed_args()


# FIXME: maybe dynamically inject examples
EXAMPLES: list[ExampleEntry] = [
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="He finally spilled the beans about the surprise party, ruining everything.",
        translation="Dia akhirnya menumpahkan kacang tentang pesta kejutan itu, merusak segalanya.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=1,
                feedback="The idiom 'spilled the beans' was translated literally as 'menumpahkan kacang', completely losing the intended meaning of revealing a secret.",
            ),
            acceptability=RubricEntry(
                score=1,
                feedback="The phrase 'menumpahkan kacang' is nonsensical in Indonesian within this context.",
            ),
            readability=RubricEntry(
                score=3,
                feedback="The sentence structure itself is readable, despite the semantic error.",
            ),
        ),
        revision="""Planned Changes:
- The phrase 'menumpahkan kacang' is a literal calque of the English idiom and makes no sense. I will replace it with the natural Indonesian equivalent 'membocorkan rahasia'.

Revision: Dia akhirnya membocorkan rahasia tentang pesta kejutan itu, merusak segalanya.""",
        known_idioms=[
            IdiomEntry(
                idiom="spill the beans",
                senses=[
                    "to reveal a secret.",
                    "to disclose information prematurely.",
                ],
                translations={},
            )
        ],
    ),
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="After years of hard work, she finally hit the nail on the head with her new business idea.",
        translation="Setelah bertahun-tahun bekerja keras, dia akhirnya menemukan ide bisnis yang tepat sasaran.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=3,
                feedback="The idiom 'hit the nail on the head' is accurately rendered as 'tepat sasaran'.",
            ),
            acceptability=RubricEntry(
                score=3, feedback="The translation is acceptable and idiomatic."
            ),
            readability=RubricEntry(
                score=3, feedback="The sentence is fluent and natural."
            ),
        ),
        revision=None,
        known_idioms=[
            IdiomEntry(
                idiom="hit the nail on the head",
                senses=[
                    "to describe exactly what is causing a situation or problem.",
                    "to do or say something exactly right.",
                ],
                translations={},
            )
        ],
    ),
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="When he heard the news, he felt under the weather and decided to stay home.",
        translation="Ketika dia mendengar kabar itu, dia memutuskan untuk tetap di rumah karena cuaca buruk.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=1,
                feedback="The idiom 'under the weather' means feeling ill, but the translation misinterpreted it as literal bad weather ('cuaca buruk').",
            ),
            acceptability=RubricEntry(
                score=3,
                feedback="The sentence is grammatically correct and acceptable as a standalone Indonesian sentence.",
            ),
            readability=RubricEntry(
                score=3,
                feedback="The sentence flows naturally and is easy to understand.",
            ),
        ),
        revision="""Planned Changes:
- The translation incorrectly interpreted 'under the weather' literally as bad weather ('cuaca buruk'). I will correct this to reflect the true meaning of feeling ill by using 'merasa tidak enak badan'.

Revision: Ketika dia mendengar kabar itu, dia merasa tidak enak badan dan memutuskan untuk tetap di rumah.""",
        known_idioms=[
            IdiomEntry(
                idiom="under the weather",
                senses=["feeling ill.", "slightly unwell or in low spirits."],
                translations={},
            )
        ],
    ),
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="She let the cat out of the bag during dinner, and everyone was shocked.",
        translation="Dia secara tidak sengaja telah menyingkap sebuah tabir kerahasiaan pada saat perjamuan makan malam, dan semua hadirin terperanjat.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=3,
                feedback="The core meaning of revealing a secret is successfully retained.",
            ),
            acceptability=RubricEntry(
                score=1,
                feedback="The phrasing ('menyingkap sebuah tabir kerahasiaan', 'perjamuan', 'hadirin terperanjat') is excessively poetic and formal for a simple dinner setting.",
            ),
            readability=RubricEntry(
                score=2,
                feedback="The sentence is overly wordy and cumbersome to read.",
            ),
        ),
        revision="""Planned Changes:
- The phrasing 'menyingkap sebuah tabir kerahasiaan' is too formal and poetic; I will change it to 'membocorkan rahasia'.
- The terms 'perjamuan makan malam' and 'hadirin terperanjat' are too stiff; I will simplify them to 'makan malam' and 'semua orang terkejut'.

Revision: Dia tanpa sengaja membocorkan rahasia saat makan malam, dan semua orang terkejut.""",
        known_idioms=[
            IdiomEntry(
                idiom="let the cat out of the bag",
                senses=[
                    "to accidentally reveal a secret.",
                    "to disclose something that was meant to be hidden.",
                ],
                translations={},
            )
        ],
    ),
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="Instead of addressing the client's complaints, the manager just passed the buck to the marketing team.",
        translation="Alih-alih menangani keluhan klien, sang manajer malah mengoper uang ke tim pemasaran.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=1,
                feedback="The idiom 'passed the buck' means to shift responsibility, but it was translated literally as 'mengoper uang' (passing money), losing the intended meaning entirely.",
            ),
            acceptability=RubricEntry(
                score=1,
                feedback="The phrase 'mengoper uang' is nonsensical in the context of handling client complaints.",
            ),
            readability=RubricEntry(
                score=3,
                feedback="The sentence structure itself is readable, despite the severe semantic error.",
            ),
        ),
        revision="""Planned Changes:
- The literal translation 'mengoper uang' misinterprets the idiom and makes no sense in context. I will replace it with the culturally appropriate and natural Indonesian phrasing 'lepas tangan dan melimpahkannya' to accurately convey the abdication of responsibility.

Revision: Alih-alih menangani keluhan klien, sang manajer malah lepas tangan dan melimpahkannya ke tim pemasaran.""",
        known_idioms=[
            IdiomEntry(
                idiom="pass the buck",
                senses=[
                    "to shift the responsibility or blame to someone else.",
                ],
                translations={},
            )
        ],
    ),
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="We went to that new restaurant last night. The service was terrible, so we just left before ordering.",
        translation="Kami pergi ke restoran baru itu semalam. Layanannya sangat buruk, jadi kami baru saja pergi sebelum memesan.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=2,
                feedback="The translation uses 'baru saja' for 'just', which implies the action is happening right now. This breaks the timeline established by 'semalam' (last night).",
            ),
            acceptability=RubricEntry(
                score=2,
                feedback="The use of 'baru saja' creates a confusing temporal clash for the reader, making the phrasing unnatural for a recounting of a past event.",
            ),
            readability=RubricEntry(
                score=3,
                feedback="The sentence is grammatically standard, despite the logical error.",
            ),
        ),
        revision="""Planned Changes:
- The translation incorrectly maps 'just' to 'baru saja' (recently), which contradicts the 'semalam' timeframe. In this context, 'just' implies taking a simple or immediate alternative action. I will replace 'baru saja pergi' with 'langsung pergi saja' to maintain the correct past narrative flow.

Revision: Kami pergi ke restoran baru itu semalam. Layanannya sangat buruk, jadi kami langsung pergi saja sebelum memesan.""",
        known_idioms=[],
    ),
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="I was going to tell you the truth, but I panicked.",
        translation="Aku akan mengatakan yang sebenarnya padamu, tapi aku panik.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=2,
                feedback="The English phrase 'was going to' expresses a past intention that was not fulfilled. Translating it simply as 'akan' without a temporal marker makes it sound like a general or future statement. The contrasting past outcome ('panicked') also lacks emphasis.",
            ),
            acceptability=RubricEntry(
                score=2,
                feedback="While grammatically acceptable, it lacks the natural narrative flow a native speaker uses to express a failed past intention followed by an unexpected reaction.",
            ),
            readability=RubricEntry(
                score=3, feedback="The sentence is straightforward and readable."
            ),
        ),
        revision="""Planned Changes:
- The source text relies on 'was going to' and 'panicked' to contrast a past intention with an unexpected outcome. To map this past tense dynamic naturally into Indonesian, I will add 'tadinya' (initially) to anchor the unfulfilled plan in the past, and 'malah' (instead) to emphasize the contrary reaction of panicking.

Revision: Tadinya aku mau mengatakan yang sebenarnya padamu, tapi aku malah panik.""",
        known_idioms=[],
    ),
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="The store had been in the red for three months. We just sat behind the counter cooling our heels, waiting for a stray customer to walk through the door.",
        translation="Toko itu sudah berada di dalam merah selama tiga bulan. Kami hanya duduk di belakang meja kasir mendinginkan tumit kami, menunggu pelanggan tersesat berjalan melewati pintu.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=1,
                feedback="The translation relies on literal word-for-word mappings for two distinct idioms. 'In the red' is translated as 'di dalam merah' instead of its financial meaning (merugi), and 'cooling our heels' is translated as 'mendinginkan tumit kami', completely destroying the figurative meaning of waiting idly.",
            ),
            acceptability=RubricEntry(
                score=1,
                feedback="The phrase 'mendinginkan tumit kami' is severe translationese. No native Indonesian speaker uses this phrasing; it fails to convey natural narrative flow or cultural equivalence.",
            ),
            readability=RubricEntry(
                score=2,
                feedback="The syntax is understandable, but the literal translation of the idioms causes significant cognitive friction and breaks the narrative illusion.",
            ),
        ),
        revision="""Planned Changes:
- 'In the red' indicates financial deficit, which maps naturally to the accounting term 'merugi'.
- 'Cooling our heels' describes a state of forced, idle waiting. To avoid a stiff dictionary substitution, I will apply situational paraphrasing to describe what is actually happening in the scene: 'duduk termangu... tanpa melakukan apa-apa' (sitting blankly... doing nothing).
- Adjust the final clause ('waiting for a stray customer...') to read more fluidly in Indonesian narrative prose ('berharap ada satu-dua pelanggan yang tersesat masuk').

Revision: Toko itu sudah merugi selama tiga bulan. Kami hanya duduk termangu di balik meja kasir tanpa melakukan apa-apa, berharap ada satu-dua pelanggan yang tersesat masuk ke dalam toko.""",
        known_idioms=[],
    ),
]


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

    # def format_translations(translations: dict[str, str]) -> str:
    #     if not translations:
    #         return ""

    #     return "\n\n    Translations:\n" + "\n".join(
    #         [f"        {k}: {v}" for k, v in translations.items()]
    #     )

    return f"""
Known idiom definitions:
{nl.join([f"- {i['idiom']}:{nl}{format_senses(i['senses'])}" for i in idioms])}
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


def format_examples(examples: list[ExampleEntry], depth: int) -> str:
    def format_example(entry: ExampleEntry) -> str:
        buffers = [
            f"""
Source text: {entry["source_text"]}

Type: novel

{format_idiom_knowledge(entry["known_idioms"])}

Translation: {entry["translation"]}
""".strip(),
            f"""Grade the translation based on a 3-point rubric: accuracy, acceptability, and readability.

Grades:
{format_rubric(entry["rubric"])}
""".strip(),
            f"""
Based on the grades, now provide a revision.

{entry["revision"]}
""".strip(),
        ]
        return "\n".join(buffers[:depth])

    nl = "\n"
    return f"""
=== EXAMPLES START ===

{f"{nl}---{nl}".join([format_example(i) for i in examples])}

=== EXAMPLES END ===
""".strip()


OPTIMIZER_INIT_PROMPT = f"""
{format_examples(EXAMPLES, 1)}

{{CONTEXT}}

Source text: {{SOURCE_TEXT}}

Situational Paraphrasing: If a literal translation of an idiom sounds stiff, or if a direct target-language idiomatic equivalent does not exist, unpack the idiom into its natural situational meaning. Translate what is actually happening in the scene rather than translating the words of the phrase mechanically.

Translation:
""".strip()


EVALUATOR_PROMPT = f"""
{format_examples(EXAMPLES, 2)}

{{CONTEXT}}

Source text: {{SOURCE_TEXT}}

Translation: {{TRANSLATION_ATTEMPT}}

Evaluate the translation based on a strict 3-point rubric. You MUST penalize literal translations (calques) of English idioms or phrases that sound unnatural in {{TARGET_LANGUAGE}}.

Scoring Criteria:
Accuracy:
- 3: Meaning is perfectly preserved.
- 2: Minor shifts in meaning, but core message remains.
- 1: Severe mistranslation, hallucination, or literal translation of an idiom that loses the figurative meaning.

Acceptability (Naturalness):
- 3: Reads like a text originally written by a native Indonesian speaker.
- 2: Grammatically correct, but phrasing is slightly awkward or overly formal.
- 1: "Translationese" - grammatically correct but utilizes phrasing nobody uses in real life (e.g., word-for-word literal translations like "penipu kepercayaan" for "confidence trickster").

Readability:
- 3: Flows smoothly and effortlessly.
- 2: understandable, but requires slight cognitive effort due to clunky syntax.
- 1: Difficult to read or confusing.

Target language evaluation:
1. Identify any idioms or complex phrases in the source text.
2. State how a native {{TARGET_LANGUAGE}} speaker would naturally express that concept, ignoring the {{SOURCE_LANGUAGE}} phrasing.
3. Compare the provided translation against the native expectation. If the provided translation uses literal phrasing ("translationese") instead of the native equivalent, you MUST score Acceptability as a 1 or 2.

Grades:
""".strip()


OPTIMIZER_RETRY_PROMPT = f"""
{format_examples(EXAMPLES, 3)}

{{CONTEXT}}

Source text: {{SOURCE_TEXT}}
 
Situational Paraphrasing: If a literal translation of an idiom sounds stiff, or if a direct target-language idiomatic equivalent does not exist, unpack the idiom into its natural situational meaning. Translate what is actually happening in the scene rather than translating the words of the phrase mechanically.

Translation: {{TRANSLATION_ATTEMPT}}

Grade the translation based on a 3-point rubric: accuracy, acceptability, and readability.

Grades:
{{GRADES}}

Based on the grades, provide a revision. You MUST format your response exactly as follows:
Planned Changes:
- <your reasoning>

Revision: <your final translated string>
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


def fill_in_messages(state: State, messages: list[tuple[str, str, str]]) -> None:
    if ARGS.keep_n_messages == 0:
        return

    roles: dict[str, str] = {
        "attempt": "translator",
        "evaluation": "editor",
    }

    history = state["history"]

    if ARGS.keep_n_messages > 0:
        history = history[-ARGS.keep_n_messages - 1 :]

    LOGGER.info("Filling in %d messages from history.", len(history))

    for entry in history:
        assert "type" in entry
        assert "prompt" in entry
        assert "system_prompt" in entry
        assert "raw_output" in entry

        messages.append(("user", entry["prompt"], "user"))
        messages.append(("assistant", entry["raw_output"], roles[entry["type"]]))


def build_messages(
    state: State, system_prompt: str, user_prompt: str
) -> list[tuple[str, str, str]]:
    messages: list[tuple[str, str, str]] = []
    if system_prompt:
        messages.append(("system", system_prompt, "system"))
    fill_in_messages(state, messages)
    messages.append(("user", user_prompt, "user"))
    return messages


def parse_rubric(text: str) -> Rubric:
    rubric: Rubric = {
        "accuracy": {"score": 0, "feedback": "Missing feedback."},
        "acceptability": {"score": 0, "feedback": "Missing feedback."},
        "readability": {"score": 0, "feedback": "Missing feedback."},
    }

    for line in text.strip().splitlines():
        # match: - key: score. feedback
        match = re.match(r"- (\w+): (\d+)\.\s*(.*)", line.strip())
        if not match:
            continue

        key, score, feedback = match.groups()
        rubric[key] = {
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
    prompt = f"Translate the following text into {state['source_text']['target_lang']}:\n{state['source_text']['text']}\n\nTranslation:\n"
    temp = OPTIMIZER_TEMP
    seed = state["optimizer_seed"]
    messages = build_messages(state, "", prompt)
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


async def handle_optimization_state(state: State) -> None:
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
        prompt = OPTIMIZER_INIT_PROMPT.format(
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
        prompt = OPTIMIZER_RETRY_PROMPT.format(
            SOURCE_TEXT=state["source_text"]["text"],
            TRANSLATION_ATTEMPT=last_attempt["translation"],
            GRADES=format_rubric(last_evaluation["rubric"]),
            CONTEXT=context,
        )

    temp = OPTIMIZER_TEMP if is_draft else OPTIMIZER_ALT_TEMP
    seed = state["optimizer_seed"] * 10 + state["attempt"]
    messages = build_messages(state, "", prompt)
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

    # do not mutate the original evaluator seed
    seed = state["evaluator_seed"] + state["iteration_id"] * 100
    last_attempt = state["history"][-1]

    assert last_attempt.get("type") == "attempt"
    assert "translation" in last_attempt

    prompt = EVALUATOR_PROMPT.format(
        SOURCE_TEXT=state["source_text"]["text"],
        TRANSLATION_ATTEMPT=last_attempt["translation"],
        CONTEXT=format_context(state),
        TARGET_LANGUAGE=state["source_text"]["target_lang"],
        SOURCE_LANGUAGE=state["source_text"]["source_lang"],
    )
    messages = build_messages(state, "", prompt)
    output = (
        await run_inference(
            state["client"],
            ARGS.endpoint,
            ARGS.model,
            EVALUATOR_TEMP,
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
        "temp": EVALUATOR_TEMP,
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
                EVALUATOR_TEMP,
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

    state["next_state"] = "optimization"
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
        "optimizer_seed",
        "optimizer_temp",
        "evaluator_seed",
        "evaluator_temp",
        "source_text",
        "translation_attempt",
        "grade",
        "raw_translation",
        "raw_evaluation",
        "timestamp",
        "optimizer_system_prompt",
        "optimizer_user_prompt",
        "evaluator_system_prompt",
        "evaluator_user_prompt",
    )

    STATE_HANDLERS: Final = {
        "baseline": handle_baseline_state,
        "optimization": handle_optimization_state,
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

            source_text: SourceTextEntry = {
                "source_lang": input_json["source_lang"],
                "target_lang": input_json["target_lang"],
                "text": text["content"],
                "type": input_json.get("type", "general"),
                "id": text_idx + 1,
                "external_knowledge": input_json.get("external_knowledge", [])
                + text.get("external_knowledge", []),
                "idiom_matches": await self.embedder.get_idiom_definitions(
                    text["content"]
                ),
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
                next_state="baseline" if ARGS.baseline else "optimization",
                max_attempt=ARGS.refinement_iterations,
                attempt=0,
                history=[],
                optimizer_seed=SEEDS[i],
                evaluator_seed=EVALUATOR_SEED,
                client=self.client,
                csv_writer=self.csv_writer,
            )

            while handler := self.STATE_HANDLERS.get(state["next_state"]):
                await handler(state)
                _ = self.csv_file and self.csv_file.flush()

                # let llama-server disconnect the previous connection
                await asyncio.sleep(0.1)

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

    embedder = Embedder(ARGS.embedding_model)

    if ARGS.vectorise:
        embedder.generate_vectors()
        exit(0)

    input_files = [Path(p) for p in ARGS.input.split(",")]
    root = Path(__file__).parent
    output_files = [
        get_next_available_path(
            root
            / ("baseline_attempts" if ARGS.baseline else "evaluator_optimizer_attempts")
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
