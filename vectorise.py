import json
import pickle
from typing import cast

# https://github.com/microsoft/pylance-release/issues/7615
from sentence_transformers import SentenceTransformer  # type: ignore

from _types import IdiomEntry
from lib import VECTORISED_DICTIONARY_PATH

normalised_dict: dict[str, IdiomEntry] = {}

with open("idiom_dict/cherrypicked.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
    for phrase, data in json_data.items():
        normalised_dict[phrase] = IdiomEntry(
            idiom=cast(str, phrase),
            senses=data.get("senses", []),  # pyright: ignore[reportAny]
            translations={},
        )

with open("idiom_dict/idiomKB.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
    for entry in json_data:
        phrase = entry.get("idiom")
        en_meaning = entry.get("en_meaning")

        assert isinstance(phrase, str)
        assert isinstance(en_meaning, str)

        translations: dict[str, str] = {}
        if "zh_meaning" in entry:
            translations["zh"] = entry["zh_meaning"]
        if "ja_meaning" in entry:
            translations["ja"] = entry["ja_meaning"]

        if phrase in normalised_dict:
            if en_meaning and en_meaning not in normalised_dict[phrase]["senses"]:
                normalised_dict[phrase]["senses"].append(en_meaning)
            normalised_dict[phrase]["translations"].update(translations)
        else:
            normalised_dict[phrase] = IdiomEntry(
                idiom=phrase,
                senses=[en_meaning] if en_meaning else [],
                translations=translations,
            )

phrases = list(normalised_dict.keys())
print("Loading model and computing", len(normalised_dict), "embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
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

print(f"Saved to {VECTORISED_DICTIONARY_PATH}")
