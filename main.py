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
    ExampleEntry,
    IdiomEntry,
    IdiomMatchResult,
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

EVALUATOR_SEED = 727
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
ARGS = get_parsed_args()


# FIXME: maybe dynamically inject examples
EXAMPLES: list[ExampleEntry] = [
    # 1. misses the idiom definition (translates literally despite context)
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="He finally spilled the beans about the surprise party, ruining everything.",
        translation="Dia akhirnya menumpahkan kacang tentang pesta kejutan itu, merusak segalanya.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=1,
                feedback="The context provided the idiom 'spill the beans' as revealing a secret, but it was ignored and translated literally as 'menumpahkan kacang'.",
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
- The previous translation ignored the provided external knowledge. I will replace the literal calque 'menumpahkan kacang' with the correct Indonesian equivalent 'membocorkan rahasia'.

Revision: Dia akhirnya membocorkan rahasia tentang pesta kejutan itu, merusak segalanya.""",
        known_idioms=[
            IdiomEntry(
                idiom="spill the beans",
                senses=[
                    "to reveal a secret.",
                    "to disclose information prematurely.",
                ],
                translations={},
                master_key="spill the beans",
            )
        ],
    ),
    # 2. picks the wrong sense
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="The new graphics card completely blew me away.",
        translation="Kartu grafis baru itu benar-benar menerbangkanku.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=1,
                feedback="The idiom 'blow away' was translated using the physical sense of moving by wind ('menerbangkanku'), missing the intended figurative sense of impressing someone.",
            ),
            acceptability=RubricEntry(
                score=1,
                feedback="Saying a graphics card 'menerbangkanku' is comical and incorrect in Indonesian tech contexts.",
            ),
            readability=RubricEntry(
                score=3,
                feedback="The grammar is technically correct.",
            ),
        ),
        revision="""Planned Changes:
- I selected the wrong sense of 'blow away' from the provided idiom dictionary. The context is technology impressing a user, so I need to use the second sense ('to impress greatly'). I will use 'membuatku sangat terkesan' or 'memukauku'.

Revision: Kartu grafis baru itu benar-benar memukauku.""",
        known_idioms=[
            IdiomEntry(
                idiom="blow away",
                senses=[
                    "to kill someone.",
                    "to impress someone greatly.",
                    "to move something physically by wind.",
                ],
                translations={},
                master_key="blow away",
            )
        ],
    ),
    # 3. false positive (literal meaning intended, but idiom definition incorrectly applied)
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="The mechanic asked me to step on it to see if the pedal was stuck.",
        translation="Mekanik itu memintaku untuk bergegas untuk melihat apakah pedalnya tersangkut.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=1,
                feedback="The phrase 'step on it' was translated using its idiomatic sense ('bergegas' / hurry up), but the context dictates a literal physical action of stepping on a pedal.",
            ),
            acceptability=RubricEntry(
                score=2,
                feedback="The sentence feels logically disconnected because hurrying does not test a stuck pedal.",
            ),
            readability=RubricEntry(
                score=3,
                feedback="The phrasing flows well natively.",
            ),
        ),
        revision="""Planned Changes:
- I incorrectly applied the figurative idiom 'step on it' (to hurry). The presence of 'pedal' dictates that the literal meaning was intended. I will translate this as a physical action: 'menginjaknya'.

Revision: Mekanik itu memintaku untuk menginjaknya untuk melihat apakah pedalnya tersangkut.""",
        known_idioms=[
            IdiomEntry(
                idiom="step on it",
                senses=[
                    "to go faster or hurry up.",
                ],
                translations={},
                master_key="step on it",
            )
        ],
    ),
    # 4. correct execution (idiom successfully adapted stylistically)
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="Instead of addressing the client's complaints, the manager just passed the buck to the marketing team.",
        translation="Alih-alih menangani keluhan klien, sang manajer malah lepas tangan dan melimpahkannya ke tim pemasaran.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=3,
                feedback="The idiom 'pass the buck' is accurately rendered based on the provided definition of shifting responsibility.",
            ),
            acceptability=RubricEntry(
                score=3,
                feedback="The phrase 'lepas tangan dan melimpahkannya' is highly idiomatic and natural in corporate Indonesian contexts.",
            ),
            readability=RubricEntry(
                score=3,
                feedback="The narrative flow is flawless.",
            ),
        ),
        revision=None,
        known_idioms=[
            IdiomEntry(
                idiom="pass the buck",
                senses=[
                    "to shift the responsibility or blame to someone else.",
                ],
                translations={},
                master_key="pass the buck",
            )
        ],
    ),
    # === false positives ===
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="The frustrated janitor kicked the bucket, spilling dirty water all over the freshly mopped floor.",
        translation="Petugas kebersihan yang frustrasi itu meninggal dunia, menumpahkan air kotor ke seluruh lantai yang baru dipel.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=1,
                feedback="The translator incorrectly applied the provided idiom definition for 'kick the bucket' (to die). The surrounding context of dirty water and a mopped floor indicates this is a literal physical action of kicking a pail, making 'meninggal dunia' a severe semantic error.",
            ),
            acceptability=RubricEntry(
                score=1,
                feedback="Saying the janitor died and then spilled water is illogical and breaks the narrative.",
            ),
            readability=RubricEntry(
                score=3,
                feedback="The sentence structure is technically sound.",
            ),
        ),
        revision="""Planned Changes:
- I blindly applied the figurative idiom definition ('meninggal dunia') provided in the external knowledge. The context clearly requires the literal meaning. I will revert to translating the physical action: 'menendang ember'.

Revision: Petugas kebersihan yang frustrasi itu menendang ember, menumpahkan air kotor ke seluruh lantai yang baru dipel.""",
        known_idioms=[
            IdiomEntry(
                idiom="kick the bucket",
                senses=[
                    "to die.",
                ],
                translations={},
                master_key="kick the bucket",
            )
        ],
    ),
    # 8. another false positive
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="After finishing his heavy steak dinner, he asked the waiter for a piece of cake.",
        translation="Setelah menghabiskan makan malam steik yang mengenyangkan, dia meminta hal yang sangat mudah kepada pelayan.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=1,
                feedback="The external knowledge for 'piece of cake' (an easy task) was inappropriately applied. In a restaurant setting directly following a meal, the source text refers to literal dessert. It must be translated as 'sepotong kue'.",
            ),
            acceptability=RubricEntry(
                score=1,
                feedback="Asking a waiter for 'an easy thing' in this context is completely unnatural and nonsensical.",
            ),
            readability=RubricEntry(
                score=3,
                feedback="The grammar is correct.",
            ),
        ),
        revision="""Planned Changes:
- The translation used the figurative sense ('hal yang sangat mudah') from the idiom dictionary, ignoring the restaurant context. I will ignore the external knowledge and translate 'piece of cake' literally to 'sepotong kue'.

Revision: Setelah menghabiskan makan malam steik yang mengenyangkan, dia meminta sepotong kue kepada pelayan.""",
        known_idioms=[
            IdiomEntry(
                idiom="piece of cake",
                senses=[
                    "something easily achieved.",
                    "a simple task.",
                ],
                translations={},
                master_key="piece of cake",
            )
        ],
    ),
    # 9. another false positive
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="While eating the wild game meat, he accidentally bit the bullet that was still lodged in the tissue, cracking his tooth.",
        translation="Saat memakan daging hewan liar itu, dia tidak sengaja menahan penderitaan yang masih bersarang di jaringannya, membuat giginya retak.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=1,
                feedback="The idiom 'bite the bullet' was mistakenly translated using its figurative sense ('menahan penderitaan'). The presence of hunting, meat, and a cracked tooth means this refers to a literal piece of ammunition.",
            ),
            acceptability=RubricEntry(
                score=1,
                feedback="The phrasing 'menahan penderitaan yang masih bersarang di jaringannya' makes absolutely no sense in Indonesian.",
            ),
            readability=RubricEntry(
                score=2,
                feedback="The confusing semantics disrupt readability.",
            ),
        ),
        revision="""Planned Changes:
- The provided idiom definition ('to endure a painful situation') is a trap here. The context demands a literal translation of biting a physical bullet left in hunted meat. I will replace the figurative attempt with the literal action: 'menggigit peluru'.

Revision: Saat memakan daging hewan liar itu, dia tidak sengaja menggigit peluru yang masih bersarang di jaringannya, membuat giginya retak.""",
        known_idioms=[
            IdiomEntry(
                idiom="bite the bullet",
                senses=[
                    "to endure a painful or otherwise unpleasant situation that is seen as unavoidable.",
                ],
                translations={},
                master_key="bite the bullet",
            )
        ],
    ),
    # === technically accurate but clunky/ungrammatical translations ===
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="She opened her bag, took out her wallet, and paid for her coffee. The coffee was very hot.",
        translation="Dia membuka tasnya, mengeluarkan dompetnya, dan membayar untuk kopinya. Kopi itu adalah sangat panas.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=3,
                feedback="All informational elements are translated.",
            ),
            acceptability=RubricEntry(
                score=1,
                feedback="Heavy translationese. The repetitive '-nya' suffixes feel highly unnatural. 'Membayar untuk' is a literal calque of 'paid for'. Using 'adalah' as a direct translation of the copula 'was' before an adjective is grammatically incorrect in Indonesian.",
            ),
            readability=RubricEntry(
                score=2,
                feedback="The sentence is comprehensible but reads like a machine translation, causing stylistic friction.",
            ),
        ),
        revision="""Planned Changes:
- Remove redundant possessive suffixes ('-nya') since ownership is already established by the subject.
- Drop the literal preposition 'untuk' after 'membayar'.
- Remove the incorrect copula 'adalah' before the adjective 'panas'.

Revision: Dia membuka tas, mengeluarkan dompet, lalu membayar kopinya. Kopi tersebut sangat panas.""",
        known_idioms=[],
    ),
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="The students must submit all the required documents to the teachers.",
        translation="Para siswa-siswa harus mengumpulkan semua dokumen-dokumen yang dibutuhkan kepada para guru-guru.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=3,
                feedback="The core meaning is intact.",
            ),
            acceptability=RubricEntry(
                score=1,
                feedback="Severe grammatical redundancy. Indonesian does not require strict pluralization of every noun. Combining the plural marker 'para'/'semua' with noun reduplication ('siswa-siswa', 'dokumen-dokumen') is a common structural error and highly unidiomatic.",
            ),
            readability=RubricEntry(
                score=1,
                feedback="The excessive reduplication makes the sentence extremely clunky and exhausting to read.",
            ),
        ),
        revision="""Planned Changes:
- Eliminate the double pluralizations. Once a quantifier like 'para' or 'semua' is used, the following noun must remain singular. I will change 'para siswa-siswa' to 'para siswa' and 'semua dokumen-dokumen' to 'semua dokumen'.

Revision: Para siswa harus mengumpulkan semua dokumen yang dibutuhkan kepada guru.""",
        known_idioms=[],
    ),
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="Because of the fact that the project was behind schedule, the manager forced us to burn the midnight oil.",
        translation="Dikarenakan oleh fakta bahwa proyek itu berada di belakang jadwal, manajer memaksa kami untuk membakar minyak tengah malam.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=1,
                feedback="The idiom 'burn the midnight oil' was translated literally as 'membakar minyak tengah malam', missing the provided definition entirely.",
            ),
            acceptability=RubricEntry(
                score=1,
                feedback="The syntax is heavily influenced by English structure. 'Dikarenakan oleh fakta bahwa' is a bloated, literal calque of 'because of the fact that'. 'Berada di belakang jadwal' is also a stiff literal translation of 'behind schedule'.",
            ),
            readability=RubricEntry(
                score=2,
                feedback="The clunky structural phrasing combined with the literal idiom makes the sentence feel highly artificial.",
            ),
        ),
        revision="""Planned Changes:
- Simplify the bloated conjunction 'Dikarenakan oleh fakta bahwa' to the natural Indonesian equivalent 'Karena'.
- Replace the literal 'berada di belakang jadwal' with the standard term 'terlambat dari jadwal'.
- Apply the external knowledge for 'burn the midnight oil' and translate it contextually as 'bekerja lembur hingga larut malam'.

Revision: Karena proyek tersebut terlambat dari jadwal, manajer memaksa kami untuk bekerja lembur hingga larut malam.""",
        known_idioms=[
            IdiomEntry(
                idiom="burn the midnight oil",
                senses=[
                    "to read, study, or work late into the night.",
                ],
                translations={},
                master_key="burn the midnight oil",
            )
        ],
    ),
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="The terms and conditions must be read and accepted by the user before creating an account.",
        translation="Syarat dan ketentuan harus dibaca dan diterima oleh pengguna sebelum membuat sebuah akun.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=3,
                feedback="The meaning is fully preserved.",
            ),
            acceptability=RubricEntry(
                score=2,
                feedback="While technically correct, the phrasing feels like a rigid manual. Using the active voice ('pengguna harus membaca...') or streamlining the subject/verb relationship would make this interface copy feel more native. Furthermore, 'sebuah akun' is a literal translation of the English indefinite article 'an', which is unnecessary here.",
            ),
            readability=RubricEntry(
                score=2,
                feedback="The sentence is slightly stiff due to the formal passive structure paired with the unneeded article.",
            ),
        ),
        revision="""Planned Changes:
- Remove the unnecessary indefinite article 'sebuah' before 'akun'.
- Restructure the sentence from a stiff passive format into a more direct active voice, which is preferred for user instructions in Indonesian UI/UX contexts.

Revision: Pengguna harus membaca dan menyetujui syarat dan ketentuan sebelum membuat akun.""",
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
- The source text relies on 'was going to' and 'panicked' to contrast a past intention with an unexpected outcome. To map this past tense dynamic naturally into Indonesian, I will add 'tadinya' (initially) to anchor the unfulfilled plan in the past, and 'malah' (instead) to emphasise the contrary reaction of panicking.

Revision: Tadinya aku mau mengatakan yang sebenarnya padamu, tapi aku malah panik.""",
        known_idioms=[],
    ),
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="My brother called me yesterday to ask for some money.",
        translation="Kakak laki-lakiku meneleponku kemarin untuk meminta uang.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=2,
                feedback="The English word 'brother' only denotes gender, not relative age. The translation assumed 'kakak' (older brother). Without surrounding context, forcing an age relationship is a semantic hallucination.",
            ),
            acceptability=RubricEntry(
                score=2,
                feedback="While grammatically perfect, the translation forces specificity that does not exist in the source text.",
            ),
            readability=RubricEntry(
                score=3,
                feedback="The sentence reads smoothly.",
            ),
        ),
        revision="""Planned Changes:
- I incorrectly assumed the brother was older ('kakak'). Since the source text does not specify age, I will use the age-neutral kinship term 'saudara' combined with the gender marker 'laki-laki'.

Revision: Saudara laki-lakiku meneleponku kemarin untuk meminta uang.""",
        known_idioms=[],
    ),
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="The startup has a strong vision and a high chance of success.",
        translation="Perusahaan rintisan itu memiliki sebuah visi yang kuat dan sebuah peluang sukses yang tinggi.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=3,
                feedback="All words are accounted for.",
            ),
            acceptability=RubricEntry(
                score=1,
                feedback="Translating the English article 'a' directly to the Indonesian numeral counter 'sebuah' for abstract concepts like 'vision' and 'chance' is a severe case of translationese. Indonesian does not use classifiers for non-physical nouns.",
            ),
            readability=RubricEntry(
                score=2,
                feedback="The repetitive use of 'sebuah' disrupts the natural flow of the sentence.",
            ),
        ),
        revision="""Planned Changes:
- Remove the counter 'sebuah' before 'visi' and 'peluang' because they are abstract nouns. Indonesian does not require articles in these positions.

Revision: Perusahaan rintisan itu memiliki visi yang kuat dan peluang sukses yang tinggi.""",
        known_idioms=[],
    ),
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="When I was a kid, I walked to school every day.",
        translation="Ketika aku adalah seorang anak, aku telah berjalan kaki ke sekolah setiap hari.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=2,
                feedback="The English past tense was incorrectly mapped to the Indonesian perfective aspect marker 'telah' (already completed). 'Telah' does not work for habitual, repeated actions in the past.",
            ),
            acceptability=RubricEntry(
                score=1,
                feedback="The phrase 'Ketika aku adalah seorang anak' is a rigid, word-for-word calque of 'When I was a kid'. Combined with the incorrect use of 'telah', the sentence is highly unnatural.",
            ),
            readability=RubricEntry(
                score=2,
                feedback="The sentence is understandable but feels entirely machine-generated.",
            ),
        ),
        revision="""Planned Changes:
- Replace the rigid calque 'Ketika aku adalah seorang anak' with the natural phrase 'Saat aku masih kecil'.
- Remove the perfective marker 'telah'. The past context is already established by 'saat aku masih kecil', and 'setiap hari' denotes a habit, making 'telah' grammatically incompatible.

Revision: Saat aku masih kecil, aku berjalan kaki ke sekolah setiap hari.""",
        known_idioms=[],
    ),
    ExampleEntry(
        source_lang="en",
        target_lang="id",
        source_text="You must verify your email before we can process the refund.",
        translation="Kamu harus memverifikasi email Anda sebelum kami bisa memproses pengembalian dana.",
        rubric=Rubric(
            accuracy=RubricEntry(
                score=3,
                feedback="The translation captures the correct instructions.",
            ),
            acceptability=RubricEntry(
                score=1,
                feedback="The translation mixes the informal subject pronoun 'kamu' with the formal possessive pronoun 'Anda' within the exact same sentence. English 'you' must be mapped to a consistent politeness register.",
            ),
            readability=RubricEntry(
                score=2,
                feedback="The jarring shift in formality breaks the professional tone.",
            ),
        ),
        revision="""Planned Changes:
- The sentence suffers from a register mismatch. Given the context is customer service (processing a refund), I need to use the formal register consistently. I will change 'Kamu' to 'Anda'.

Revision: Anda harus memverifikasi email Anda sebelum kami bisa memproses pengembalian dana.""",
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


IDIOM_EXTRACTION_GRAMMAR = """
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
Translate the provided source text using the given external knowledge and interaction history. Pay close attention to how corrections, structural boundaries, and stylistic adjustments are resolved across the multi-turn interaction examples.
""".strip()


EVALUATOR_SYSTEM_PROMPT = """
Evaluate the provided translation against the original source text by matching the scoring distributions and feedback patterns demonstrated in the few-shot examples. Assess accuracy, acceptability, and readability uniformly.

Format your output exactly as follows:
- accuracy: <score 1-3>. <feedback>
- acceptability: <score 1-3>. <feedback>
- readability: <score 1-3>. <feedback>
""".strip()


OPTIMISER_INIT_PROMPT = """
{CONTEXT}

Source text: {SOURCE_TEXT}

Provide exactly one translation:
""".strip()


EVALUATOR_INIT_PROMPT = """
{CONTEXT}

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


def get_few_shot_turns(state: State) -> list[tuple[str, str, str]]:
    ret: list[tuple[str, str, str]] = []

    is_evaluating = state["history"] and state["history"][-1].get("type") == "attempt"

    for idx, entry in enumerate(EXAMPLES):
        is_initial = idx == 0
        context_str = format_idiom_knowledge(entry["known_idioms"])

        if is_evaluating:
            ret.append(
                (
                    "user",
                    (
                        "Please grade my translation based on the rubric.\n\n"
                        if is_initial
                        else "Now grade this one.\n\n"
                    )
                    + f"{context_str}\n\n" * bool(context_str)
                    + f"Source text: {entry['source_text']}\n\n"
                    + f"Translation: {entry['translation']}\n\n"
                    + "Grades:\n",
                    "user",
                )
            )
            ret.append(("assistant", format_rubric(entry["rubric"]), "assistant"))
            continue

        init_req = (
            ("Please translate this " if is_initial else "Now translate this ")
            + f"from {entry['source_lang']} to {entry['target_lang']}.\n\n"
            + f"{context_str}\n\n" * bool(context_str)
            + f"Source text: {entry['source_text']}\n\n"
            + "Translation:\n"
        )
        ret.append(("user", init_req, "user"))
        ret.append(("assistant", entry["translation"], "assistant"))

        if not entry["revision"]:
            continue

        ret.append(
            (
                "user",
                "Okay, please adjust the translation based on my feedback\n\n"
                + format_rubric(entry["rubric"]),
                "user",
            )
        )
        ret.append(("assistant", entry["revision"], "assistant"))

    return ret


def build_messages(
    state: State, system_prompt: str, user_prompt: str
) -> list[tuple[str, str, str]]:
    messages: list[tuple[str, str, str]] = []

    if system_prompt:
        messages.append(("system", system_prompt, "system"))

    messages.extend(get_few_shot_turns(state))

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
