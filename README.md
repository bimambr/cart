# Context-Augmented Refinement for (LLM) Translation

This repository contains the scripts and methodology for a thesis project analysing the quality of translations produced by a local Large Language Model (default: gemma-4-E2B).
The framework uses RAG and an iterative workflow to generate, evaluate, and refine translations, providing a rich dataset for analysis.

## Workflow Graph

<p align="center">
  <img alt="workflow graph" src="docs/graph.png" />
</p>

The workflow was inspired by Aman Madaan's [self-refine](https://github.com/madaan/self-refine)
and LangChain's [evaluator-optimizer](https://docs.langchain.com/oss/python/langgraph/workflows-agents#evaluator-optimizer),
further grounded using RAG to dynamically inject contexts (e.g., idiom definitions, etc).

## Requirements

- Python 3.11 or higher (Python 3.14 is untested as of writing).
- Python package: `aiohttp`.
- llama.cpp (llama-server).
- LLM Model: A GGUF-compatible model. The experiments for the thesis were conducted using `unsloth/gemma-4-E2B-it-GGUF` (Q8_0 quant).
- Memory:
  - At least 16GB RAM if running on CPU, or
  - 6GB VRAM (NVIDIA GPU recommended).

## Quick Setup

<details open>

<summary>Using Scoop on Windows</summary>

1. Install Scoop from the official [website](https://scoop.sh).
2. Add `versions` bucket:

```sh
scoop bucket add versions
```

3. Install Python and llama.cpp (Vulkan, or any other variant that your hardware supports):

```sh
scoop install python311 llama.cpp-vulkan
```

4. Install Python dependencies:

```sh
python311 -m pip install aiohttp
```

5. There are two options to get the model:

  - Download the model directly: Download the [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) model from Hugging Face and place it in the root directory of this project (we use specifically the Q8_0 quant).
  - Or, install `just`, `aria2`, and `bash` for runner and model downloader:
```sh
scoop install just aria2 git

# Run this to download the model
just setup
```

</details>

<details>

<summary>Using Nix on WSL/Darwin/Unix</summary>

If you have [Nix](https://nix.dev) with flakes enabled, you can get a fully reproducible environment:

```sh
# Enter the development shell (required before running any `just` commands)
nix develop

# Then download the model (run once)
just setup
```

</details>

## Usage

The experiment is run in three stages: preparing the data, running the LLM server, and executing the main Python script.

##### Step 1: Prepare the corpus

Create a JSON file in the corpus directory (e.g., [corpus/literature.json](corpus/literature.json)). It must have `source_lang`, `target_lang`, `type`, `external_knowledge`, and lastly `texts` containing an array of text objects (`content` and `external_knowledge`).

> [!NOTE]
> Additional keys will be ignored. For example, the `page_on_translated_book` tracks where the passage appears in the translated book. You could embed the translated text directly instead, but doing so might imply that the model needs to see the reference, which it does not.

```json
{
    "source_lang": ...,
    "target_lang": ...,
    "type": ...,
    "external_knowledge": [..., ..., ...],
    "texts": [
        {
            "content": ...,
            "external_knowledge": [..., ..., ...]
        },
        ...
    ]
}
```

Then, vectorise the idioms in the json files. The json paths are hardcoded due to differing schematics between idiom providers.
But you can add more idioms inside `idiom_dict/cherrypicked.json` with the following structure:

```json
{
    "<idiom 1>": {
        "senses": [
            "<sense 1>",
            "<sense 2>"
        ]
    },
    "<idiom 2>": {
        "senses": [
            "<sense 1>"
        ]
    },
    ...
}
```

Run this to vectorise:

```sh
python vectorise_dictionary.py

# Or using just:
just vectorise
```

##### Step 2: Run the LLM Server

Open a terminal in the project's root directory and run the `llama-server`. This will load the model into memory and open an API endpoint for the script to use.

```sh
# Example command to run llama-server
llama-server -m .\gemma-4-E2B-it.gguf --port 8000 -c 32768 -fa on --cache-ram 0 --repeat-penalty 1.0 --min-p 0.01 --top-k 64 --top-p 0.95 --no-webui -ngl 99

# Or using just:
just serve
```

##### Step 3: Run the Experiment Script

Open a second terminal in the project's root directory. Run the `evaluator_optimizer.py` script, pointing it to your input corpus.

```sh
python evaluator_optimizer.py --input "corpus/literature.json" --timeout 0 --iterations 1 --refinement-iterations 3 --preserve-last-n-messages 2

# Or using just:
just run
```

## Output

The script will generate a .csv file in a newly created `evaluator_optimizer_attempts` directory. This CSV file contains a detailed log of every attempt in the refinement loop, allowing for a thorough analysis of the AI's translation products.
