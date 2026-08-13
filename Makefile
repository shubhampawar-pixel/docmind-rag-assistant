.PHONY: install run test pull-model clean-db

install:
	pip install -r requirements.txt

run:
	streamlit run ui/streamlit_app.py

test:
	pytest tests/ -v

pull-model:
	ollama pull qwen2.5:0.5b

clean-db:
	rm -rf data/chroma_db data/bm25_cache
