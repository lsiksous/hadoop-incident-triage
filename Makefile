.PHONY: venv install run pull-model

VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

pull-model:
	ollama pull qwen2.5:1.5b

run:
	$(VENV)/bin/streamlit run app.py