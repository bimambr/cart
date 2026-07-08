set shell := ["bash", "-c"]

model_url := "https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF/resolve/main/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf?download=true"
model_file := "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
corpus_embedder_file := "Snowflake/snowflake-arctic-embed-m-v1.5"
query_embedder_file := "MongoDB/mdbr-leaf-ir"
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
    @echo "Checking for corpus embedding model..."
    @if [ ! -d {{corpus_embedder_file}} ]; then \
        echo "Downloading corpus embedding model..."; \
        python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('{{corpus_embedder_file}}'); model.save('./{{corpus_embedder_file}}')"; \
    else \
        echo "Corpus embedding model already exists."; \
    fi
    @echo "Checking for query embedding model..."
    @if [ ! -d {{query_embedder_file}} ]; then \
        echo "Downloading query embedding model..."; \
        python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('{{query_embedder_file}}'); model.save('./{{query_embedder_file}}')"; \
    else \
        echo "Query embedding model already exists."; \
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
    llama-server -m "./$model" --port {{port}} -c {{ctx}} -fa off --cache-ram 0 --repeat-penalty 1.0 --min-p 0.01 --top-k 64 --top-p 0.95 --parallel 1 --threads 1 --threads-batch 1 {{args}}

run input_file="corpus/literature.json" *args:
    python main.py --input "{{input_file}}" --timeout 0 --iterations 1 --refinement-iterations 3 {{args}}

vectorise ef="" rf="":
    @embedder="{{ef}}";
    if [ -z "$embedder" ]; then embedder="{{corpus_embedder_file}}"; fi; \
    python main.py --input "./" --embedding-model "./$embedder" --vectorise
