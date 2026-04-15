# Makefile for LLM Knowledge Distillation Project
# ===============================================

.PHONY: help install train optimize-round1 optimize-round2 evaluate infer app jupyter docker-build docker-train docker-app clean

# Default target
help:
	@echo "LLM Knowledge Distillation - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          - Install dependencies"
	@echo "  make docker-build     - Build Docker image"
	@echo ""
	@echo "Training:"
	@echo "  make train            - Run training"
	@echo "  make optimize-round1  - Run Round 1 Optuna optimization"
	@echo "  make optimize-round2  - Run Round 2 Optuna optimization"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate         - Evaluate trained model"
	@echo "  make infer            - Run interactive inference"
	@echo ""
	@echo "Deployment:"
	@echo "  make app              - Run Streamlit demo app"
	@echo "  make jupyter          - Run Jupyter notebook server"
	@echo "  make docker-app       - Run Streamlit app in Docker"
	@echo "  make docker-train     - Run training in Docker"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            - Clean generated files"
	@echo "  make format           - Format code with black"
	@echo "  make lint             - Lint code with flake8"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

# Training
train:
	python scripts/train.py --config configs/default.yaml

# Optimization
optimize-round1:
	python scripts/optimize_round1.py --config configs/default.yaml

optimize-round2:
	python scripts/optimize_round2.py --config configs/default.yaml --round1-results artifacts/optuna/round1_best_params.json

# Evaluation
evaluate:
	python scripts/evaluate.py --model-path artifacts/best_model/final --generate

# Inference
infer:
	python scripts/infer.py --model-path artifacts/best_model/final --interactive

# Streamlit app
app:
	streamlit run app/streamlit_app.py

# Jupyter notebook
jupyter:
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Docker commands
docker-build:
	docker-compose build

docker-train:
	docker-compose up train

docker-optimize-round1:
	docker-compose up optimize-round1

docker-optimize-round2:
	docker-compose up optimize-round2

docker-evaluate:
	docker-compose up evaluate

docker-app:
	docker-compose up app

docker-jupyter:
	docker-compose up jupyter

# Code quality
format:
	black src/ scripts/ app/ tests/
	isort src/ scripts/ app/ tests/

lint:
	flake8 src/ scripts/ app/ tests/
	mypy src/

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src --cov-report=html

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name "*.egg-info" -delete
	rm -rf build/ dist/
	rm -rf artifacts/checkpoints/* artifacts/logs/* artifacts/plots/*

# Full pipeline
pipeline: train evaluate
	@echo "Training and evaluation complete!"

# Full optimization pipeline
optimization-pipeline: optimize-round1 optimize-round2
	@echo "Optimization pipeline complete!"
