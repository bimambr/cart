model_url := "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q8_0.gguf?download=true"
model_file := "gemma-4-E2B-it-Q8_0.gguf"
embed_model_file := "all-MiniLM-L6-v2"
port := "8127"
ctx := "32768"

default:
    @just --list

setup:
    @echo "Checking for Gemma model..."
    @if [ ! -f {{model_file}} ]; then \
        echo "Downloading {{model_file}}..."; \
        aria2c -x 16 -s 16 -o {{model_file}} "{{model_url}}"; \
    else \
        echo "Model already exists."; \
    fi
    @echo "Checking for embedding model..."
    @if [ ! -d {{embed_model_file}} ]; then \
        echo "Downloading embedding model..."; \
        python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('{{embed_model_file}}'); model.save('./{{embed_model_file}}')"; \
    else \
        echo "Embedding model already exists."; \
    fi

serve mf="" *args:
    @model="{{mf}}"; \
    if [ -z "$model" ]; then model="{{model_file}}"; fi; \
    echo "Starting llama-server with model: $model"; \
    llama-server -m "./$model" --port {{port}} -c {{ctx}} -fa on --cache-ram 2048 --repeat-penalty 1.0 --min-p 0.01 --top-k 64 --top-p 0.95 {{args}}

run input_file="corpus/literature.json" *args:
    python main.py --input "{{input_file}}" --timeout 0 --iterations 1 --refinement-iterations 3 --preserve-last-n-messages 0 --cache-prompt {{args}}

vectorise mf="":
    @model="{{mf}}"; \
    if [ -z "$model" ]; then model="{{embed_model_file}}"; fi; \
    python main.py --input "./" --embedding-model "./$model" --vectorise
