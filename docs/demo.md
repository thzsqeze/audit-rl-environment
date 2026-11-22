# Демо: сценарий показа

Всё ниже выполняется офлайн; выводы — реальные (ENV_VERSION 0.3.0). Для записи экрана:
`asciinema rec docs/demo.cast` (Linux/macOS) или любая запись терминала, затем шаги 1–6 по порядку.
Длительность ≈ 3 минуты (шаг 6 в Docker ≈ 1–2 минуты).

## 1. Скриптовая политика на эпизоде с ошибками (train-режим, награда видна)

```bash
python -m audit_env demo --seed 9 --config interacting
```

```
=== policy=reconciler episode=ep-7bd21d1106 ===
step   1: list_records {'table': 'ledger', 'limit': 50} -> reward=0.0 terminated=False truncated=False
...
P=0.21 R=0.50 F1=0.30 return=-0.422 steps=12 statuses=['affected_ids', ..., 'tp', 'tp']
=== policy=rules_auditor ...
P=1.00 R=1.00 F1=1.00 return=0.978 steps=12 statuses=['tp', 'tp', 'tp', 'tp', 'tp', 'tp']
```

Что сказать: наивная сверка (`reconciler`) ловится на отвлекающих случаях и получает отрицательную награду;
решатель по открытым правилам находит все 6 ошибок за 11 исследовательских шагов (награда = 2·1 − 1 − 0.002·11).
Промежуточная награда всегда 0 — подсказок о правильности нет.

## 2. Протокол жюри, закрытый режим: reset → чтение → submit без награды

```python
import judge_adapter as ja
ja.generate_episode(9, "type_period_cutoff_receipt", "/tmp/ep")        # открытые + закрытые файлы
env = ja.make_env()
obs, info = env.reset(0, {"mode": "closed", "episode_dir": "/tmp/ep"})
# info: {'episode_id': 'ep-fe350f9b82', 'env_version': '0.3.0', 'schema_version': '1.0.0', 'mode': 'closed'}
env.step({"name": "list_records", "args": {"table": "payments", "cursor": None, "limit": 50}})
# reward 0.0, terminated False, info {}
env.step({"name": "submit", "args": {"findings": {"episode_id": obs["manifest"]["episode_id"], "findings": []}}})
# ({'result': {'accepted': True}, 'error': None, 'remaining_steps': 98}, 0.0, True, False, {})
ja.score_episode("/tmp/ep", {"episode_id": ..., "findings": []}, 1)   # только у оценщика
# {'tp': 0, 'fp': 0, 'fn': 1, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'return': -1.0}
```

Что сказать: политика видит `reward=0` и пустой `info` даже на submit; истинная оценка — только у оценщика.

## 3. Изоляция: политика в отдельном процессе

```bash
python -m audit_env run --split dev --mode closed --isolated --policies rules_auditor,flag_all --out runs/demo
```

Политика запускается как `python -I -B -c … policy_worker <имя>` с пустым рабочим каталогом, без путей/seed/config
в argv и окружении; audit-hook запрещает чтение файлов вне кода, запись, процессы, сеть
(`tests/test_isolation.py::test_locked_down_worker_cannot_read_episode_files_or_use_network_or_processes`).
Зависшая/упавшая политика → `truncated=true`, пустой ответ, оценщик продолжает работу.

## 4. stdio-сервер (агент на любом языке)

```bash
printf '%s\n' '{"cmd": "reset", "seed": 9, "config": "interacting"}' \
  '{"cmd": "step", "action": {"name": "calculator", "args": {"expression": "0.1+0.2"}}}' \
  | python -m audit_env serve --mode closed --grader-log grader.jsonl
```

```
{"observation": {"task": "Audit the books of one fictional company ...", "manifest": {...}, ...}, "info": {...}}
{"observation": {"result": "0.3", "error": null, "remaining_steps": 99}, "reward": 0.0, "terminated": false, "truncated": false, "info": {}}
```

Оценка эпизода пишется только в `grader.jsonl`, никогда в stdout.

## 5. Разбор сбоев и экспериментальная награда

```bash
python -m audit_env analyze --run artifacts/dev --episodes episodes/dev --out reports/dev_analysis.md
```

Показать в [failure_analysis.md](failure_analysis.md): ложные находки `reconciler` по видам отвлекающих случаев
(повторяющиеся счета → `duplicate_invoice`, граница периода → `period_cutoff`/`fx_conversion`) и пропуски по
(тип, вариант). Колонка `exp. return` — экспериментальная награда, общая не меняется.

## 6. Одна команда проверки в контейнере без сети

```bash
docker build -t audit-rl-env .
docker run --rm --network none --cpus 2 --memory 4g audit-rl-env
```

```
[PASS] tests
[PASS] episode_validation
[PASS] baselines
[PASS] schemas
[PASS] analysis
[PASS] paired_checks
...
oracle          0.750  0.750  1.000    1.000    1.000    1.0   0.200
paired checks: 20/20 passed
CHECK PASSED
```

Что сказать: тесты → самопроверка всех пресетов → матрица D01–D20 жюри (5 чистых, 10 одиночных, 5 взаимодействующих)
в закрытом режиме с изолированными политиками и поиском утечек → валидация схем → измеренный разбор сбоев →
20 парных проверок (перестановка / смена ID / оба). Oracle = 1.0 на всех 20, «пометить всё» = −1.0.
