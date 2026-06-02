model_url := "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q8_0.gguf?download=true"
model_file := "gemma-4-E2B-it-Q8_0.gguf"
port := "8127"
ctx := "32768"

default:
    @just --list

setup:
    @echo "Checking for model..."
    @if [ ! -f {{model_file}} ]; then \
        echo "Downloading {{model_file}}..."; \
        aria2c -x 16 -s 16 -o {{model_file}} "{{model_url}}"; \
    else \
        echo "Model already exists."; \
    fi

serve mf="" *args:
    @model="{{mf}}"; \
    if [ -z "$model" ]; then model="{{model_file}}"; fi; \
    echo "Starting llama-server with model: $model"; \
    llama-server -m ./{{model_file}} --port {{port}} -c {{ctx}} -fa on --cache-ram 2048 --repeat-penalty 1.0 --min-p 0.01 --top-k 64 --top-p 0.95 {{args}}

run input_file="corpus/literature.json":
    python main.py --input "{{input_file}}" --timeout 0 --iterations 1 --refinement-iterations 3 --preserve-last-n-messages 0 --cache-prompt

vectorise:
    python vectorise.py
