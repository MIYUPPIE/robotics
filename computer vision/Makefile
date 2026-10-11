.PHONY: setup download-models check-env prerender-audio test eval lint \
        run-kiosk run-registration run-admin seed-demo

ENV_NAME := voter-kiosk-cv

# This machine sources ROS2 (/opt/ros/jazzy/setup.bash) in .bashrc, which sets
# PYTHONPATH globally. conda does NOT isolate against an inherited PYTHONPATH,
# so without this override, pytest/python here can pick up ROS's Python 3.12
# site-packages (and its pytest11 entry points) and crash with unrelated
# ModuleNotFoundErrors (e.g. `lark`). Every recipe below runs with a clean
# PYTHONPATH so this project's env is actually isolated, as intended.
export PYTHONPATH :=

setup:
	conda create -y -n $(ENV_NAME) python=3.11
	conda run -n $(ENV_NAME) pip install -e ".[dev]"
	$(MAKE) download-models

download-models:
	bash scripts/download_models.sh

check-env:
	python scripts/check_env.py

prerender-audio:
	python -m voice.generate_prompts

test:
	pytest

eval:
	pytest -m eval

lint:
	ruff check .
	mypy contracts storage biometric voice hand_ui kiosk_app registration_app admin_app

run-kiosk:
	python -m kiosk_app.main

run-registration:
	python -m registration_app.app

run-admin:
	python -m admin_app.app

seed-demo:
	python scripts/seed_demo_data.py
