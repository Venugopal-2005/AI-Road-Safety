# Makefile for AI-Road-Safety System
# Works on Linux, macOS, and Windows with make / nmake / git bash

.PHONY: help install test smoke samples run clean docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install      - Install required dependencies"
	@echo "  make samples      - Generate synthetic sample images and video"
	@echo "  make test         - Run pytest unit test suite"
	@echo "  make smoke        - Run end-to-end smoke test"
	@echo "  make run          - Launch Streamlit interactive web dashboard"
	@echo "  make clean        - Remove temporary caches and outputs"
	@echo "  make docker-build - Build Docker container image"
	@echo "  make docker-run   - Run container on port 8501"

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

samples:
	python scripts/generate_samples.py

test:
	python -m pytest tests/

smoke:
	python tests/smoke_test.py

run:
	python -m streamlit run app.py

clean:
	rm -rf .pytest_cache/ __pycache__/ src/**/__pycache__/ tests/__pycache__/ outputs/*.jpg outputs/*.mp4

docker-build:
	docker build -t road-safety-system:latest .

docker-run:
	docker run -p 8501:8501 -v $(PWD)/outputs:/app/outputs road-safety-system:latest
