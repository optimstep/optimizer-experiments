SHELL := /bin/bash
PYTHON ?= python3
VENV ?= .venv
CONFIG ?= configs/muon.yaml
REMOTE ?=
REMOTE_DIR ?= /workspace/optimizer-experiments
RESULTS ?= results

.PHONY: bootstrap install submodules prepare smoke run next-token-smoke next-token-suite nanogpt-install nanogpt-data nanogpt-dry-run nanogpt-run test lint validate-configs remote-setup remote-run sync dashboard hard-stop

bootstrap: submodules
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e '.[dev]'
	$(VENV)/bin/pip install 'git+https://github.com/b0nce/MemoryEfficientCLIP.git'

install:
	$(PYTHON) -m pip install -e '.[dev]'
	$(PYTHON) -m pip install 'git+https://github.com/b0nce/MemoryEfficientCLIP.git'

submodules:
	git submodule update --init --recursive

prepare:
	$(VENV)/bin/python experiments/train.py --config configs/base.yaml --prepare-only

smoke:
	$(VENV)/bin/python experiments/train.py --config $(CONFIG) --set training.steps=2 --set training.batch_size=256 --set training.chunk_size=128

run:
	$(VENV)/bin/python experiments/train.py --config $(CONFIG)

next-token-smoke:
	$(VENV)/bin/python -m experiments.next_token.train \
		--config configs/next_token/muon.yaml \
		--set data.synthetic=true --set training.steps=2 \
		--set training.batch_size=4 --set training.sequence_length=16 \
		--set logging.validation_batches=0 --set logging.checkpoint_every=0 \
		--set logging.tensorboard_dir=results/next_token/smoke/tensorboard

next-token-suite:
	PYTHON=$(VENV)/bin/python bash scripts/run_next_token_suite.sh

NANOGPT_CONFIG ?= configs/nanogpt_speedrun/baseline.yaml

nanogpt-install: submodules
	$(VENV)/bin/python -m experiments.nanogpt_speedrun.run install --config $(NANOGPT_CONFIG)

nanogpt-data: submodules
	$(VENV)/bin/python -m experiments.nanogpt_speedrun.run data --config $(NANOGPT_CONFIG)

nanogpt-dry-run: submodules
	$(VENV)/bin/python -m experiments.nanogpt_speedrun.run run --config $(NANOGPT_CONFIG) --dry-run

nanogpt-run: submodules
	$(VENV)/bin/python -m experiments.nanogpt_speedrun.run run --config $(NANOGPT_CONFIG)

test:
	$(VENV)/bin/pytest -q

lint:
	$(VENV)/bin/ruff check .

validate-configs:
	$(VENV)/bin/python scripts/validate_configs.py

remote-setup:
	test -n "$(REMOTE)"
	rsync -az --delete --exclude results --exclude artifacts --exclude .venv --exclude third_party/modded-nanogpt/ ./ $(REMOTE):$(REMOTE_DIR)/
	ssh $(REMOTE) 'cd $(REMOTE_DIR) && make bootstrap prepare'

remote-run:
	test -n "$(REMOTE)"
	ssh $(REMOTE) 'cd $(REMOTE_DIR) && nohup make run CONFIG=$(CONFIG) > remote-run.log 2>&1 &'

sync:
	test -n "$(REMOTE)"
	mkdir -p $(RESULTS)
	rsync -az $(REMOTE):$(REMOTE_DIR)/results/ $(RESULTS)/

dashboard:
	$(VENV)/bin/tensorboard --logdir $(RESULTS) --bind_all --port 8088

hard-stop:
	test -n "$(VAST_INSTANCE_ID)"
	test -n "$(MAX_DOLLARS)"
	bash scripts/vast_hard_stop.sh
