set shell := ["bash", "-c"]

model_url := "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q8_0.gguf?download=true"
model_file := "gemma-4-E2B-it-Q8_0.gguf"
embedder_file := "all-MiniLM-L6-v2"
reranker_file := "mixedbread-ai/mxbai-rerank-base-v2"
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
    @if [ ! -d {{embedder_file}} ]; then \
        echo "Downloading embedding model..."; \
        python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('{{embedder_file}}'); model.save('./{{embedder_file}}')"; \
    else \
        echo "Embedding model already exists."; \
    fi
    @echo "Checking for rerank model..."
    @if [ ! -d {{reranker_file}} ]; then \
        echo "Downloading rerank model..."; \
        python -c "from sentence_transformers import CrossEncoder; model = CrossEncoder('{{reranker_file}}'); model.save('./{{reranker_file}}')"; \
    else \
        echo "Rerank model already exists."; \
    fi

serve mf="" *args:
    @model="{{mf}}"; \
    if [ -z "$model" ]; then model="{{model_file}}"; fi; \
    echo "Starting llama-server with model: $model"; \
    llama-server -m "./$model" --port {{port}} -c {{ctx}} -fa on --cache-ram 2048 --repeat-penalty 1.0 --min-p 0.01 --top-k 64 --top-p 0.95 {{args}}

run input_file="corpus/literature.json" *args:
    python main.py --input "{{input_file}}" --timeout 0 --iterations 1 --refinement-iterations 3 --cache-prompt {{args}}

vectorise ef="" rf="":
    @embedder="{{ef}}"; \
    reranker="{{rf}}"
    if [ -z "$embedder" ]; then embedder="{{embedder_file}}"; fi; \
    if [ -z "$reranker" ]; then reranker="{{reranker_file}}"; fi; \
    python main.py --input "./" --embedding-model "./$embedder" --rerank-model "./$reranker" --vectorise
