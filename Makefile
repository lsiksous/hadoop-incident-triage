.PHONY: venv install run pull-model

VENV := .venv

# Détection simple Windows vs Unix
ifeq ($(OS),Windows_NT)
	PY         := $(VENV)/Scripts/python.exe
	PIP        := $(VENV)/Scripts/pip.exe
	STREAMLIT  := $(VENV)/Scripts/streamlit.exe
else
	PY         := $(VENV)/bin/python
	PIP        := $(VENV)/bin/pip
	STREAMLIT  := $(VENV)/bin/streamlit
endif

venv:
	python3 -m venv $(VENV)

install: venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

pull-model:
	ollama pull qwen2.5:1.5b

run:
	$(STREAMLIT) run app.py
