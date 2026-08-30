SHELL := /bin/bash
PYTHON ?= python3
VENV ?= .venv
CONFIG ?= configs/muon.yaml
REMOTE ?=
REMOTE_DIR ?= /workspace/optimizer-experiments
RESULTS ?= results

.PHONY: bootstrap install prepare smoke run test lint remote-setup remote-run sync dashboard hard-stop

bootstrap:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e '.[dev]'
	$(VENV)/bin/pip install 'git+https://github.com/b0nce/MemoryEfficientCLIP.git'

install:
	$(PYTHON) -m pip install -e '.[dev]'
	$(PYTHON) -m pip install 'git+https://github.com/b0nce/MemoryEfficientCLIP.git'

prepare:
	$(VENV)/bin/python experiments/train.py --config configs/base.yaml --prepare-only

smoke:
	$(VENV)/bin/python experiments/train.py --config $(CONFIG) --set training.steps=2 --set training.batch_size=256 --set training.chunk_size=128

run:
	$(VENV)/bin/python experiments/train.py --config $(CONFIG)

test:
	$(VENV)/bin/pytest -q

lint:
	$(VENV)/bin/ruff check .

remote-setup:
	test -n "$(REMOTE)"
	rsync -az --delete --exclude results --exclude artifacts --exclude .venv ./ $(REMOTE):$(REMOTE_DIR)/
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

