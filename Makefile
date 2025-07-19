.PHONY: help install install-dev test test-cov lint format type-check clean docs

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

test:  ## Run tests
	pytest tests/

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=core --cov=nn --cov=optim --cov-report=html --cov-report=term-missing

lint:  ## Run linting
	flake8 core nn optim examples
	mypy core nn optim --ignore-missing-imports

format:  ## Format code
	black core nn optim examples
	isort core nn optim examples

type-check:  ## Run type checking
	mypy core nn optim --ignore-missing-imports

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:  ## Build documentation
	cd docs && make html

benchmark:  ## Run performance benchmarks
	pytest tests/benchmarks/ -v

install-hooks:  ## Install pre-commit hooks
	pre-commit install

run-hooks:  ## Run pre-commit hooks on all files
	pre-commit run --all-files
