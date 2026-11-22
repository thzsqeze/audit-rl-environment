PY ?= python3
IMAGE ?= audit-rl-env

.PHONY: install lint test check analysis episodes trajectories demo docker-build docker-check

install:
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install --no-deps -e .

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest -q

check:
	$(PY) -m audit_env check

analysis: check
	mkdir -p docs && cp reports/failure_analysis.md docs/failure_analysis.md

episodes:
	rm -rf episodes && $(PY) -m audit_env generate --split train --out episodes/train && \
	$(PY) -m audit_env generate --split dev --out episodes/dev

trajectories:
	$(PY) -m audit_env run --split train --mode train --policies random,reconciler,flag_all,oracle,rules_auditor --out artifacts/train
	$(PY) -m audit_env run --split dev --mode closed --policies random,reconciler,flag_all,oracle,rules_auditor --out artifacts/dev

demo:
	$(PY) -m audit_env demo --seed 9 --config interacting

docker-build:
	docker build -t $(IMAGE) .

docker-check: docker-build
	docker run --rm --network none --cpus 2 --memory 4g $(IMAGE)
