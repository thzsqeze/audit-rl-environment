# audit-rl-environment

Воспроизводимая RL-среда для дообучения и оценки ИИ-аудитора (кейс FinTech №1).
Среда генерирует согласованную скрытую историю вымышленной компании за один отчётный период,
вносит в книги ошибки с записанным причинным изменением и оценивает выводы агента с доказательствами.

* runtime — только стандартная библиотека Python ≥ 3.11: без сети, GPU и платных API;
* контракт в стиле Gymnasium: `reset(seed, config) -> (obs, info)`, `step(action) -> (obs, reward, terminated, truncated, info)`;
* 4 типа ошибок, 2 сценария взаимодействующих ошибок, чистые компании и 6 видов отвлекающих случаев;
* общий matcher и награда `clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)` с тестами на атакующие ответы;
* все данные синтетические: никаких реальных компаний, людей, счетов и платежей.

## Быстрый старт

```bash
# единая команда проверки в контейнере (офлайн, 2 CPU, 4 ГБ)
docker build -t audit-rl-env .
docker run --rm --network none --cpus 2 --memory 4g audit-rl-env      # == python -m audit_env check

# локально
make install
make check        # тесты + валидация данных + 4 базовые политики + воспроизводимость
make demo         # скриптовая политика на эпизоде со взаимодействующими ошибками
```

`check` пишет `reports/check_report.json` и возвращает ненулевой код при любом провале.

## Использование из Python

```python
from audit_env.env import AuditEnv

env = AuditEnv(mode="train")            # "eval": награда и диагностика остаются у оценщика
obs, info = env.reset(seed=7, config="dev")
episode_id = info["episode_id"]
obs, reward, terminated, truncated, _ = env.step(
    {"tool": "list_records", "args": {"table": "ledger", "limit": 50}})
obs, reward, terminated, truncated, _ = env.step(
    {"tool": "submit", "args": {"episode_id": episode_id, "findings": []}})
```

Изоляция процессов (агент не делит память с оценщиком):

```bash
python -m audit_env serve --mode eval --grader-log reports/grader.jsonl
# stdin:  {"cmd": "reset", "seed": 3, "config": "clean"}
#         {"cmd": "step", "action": {"tool": "get_document", "args": {"doc_id": "DOC-..."}}}
```

## Действия

| tool | args | примечание |
|---|---|---|
| `list_records` | `table`, `cursor?`, `limit?` | `ledger`/`invoices`/`payments`/`fx`, ≤ 50 строк, стабильные курсоры `"<table>:<offset>"` |
| `get_record` | `table`, `id` | `row_id` / `invoice_id` / `payment_id` / `FX:<date>:<src>:<dst>` |
| `get_document` | `doc_id` | документ счёта или платежа; также `manifest`, `policy` |
| `calculator` | `expression` | только числа, `+ - * / ( )`, десятичная арифметика |
| `submit` | `episode_id`, `findings` | `findings.json`; завершает эпизод (`terminated=true`) |

Любой шаг (включая ошибочный) расходует бюджет; 100 шагов максимум, 60 с на эпизод → `truncated=true`
и оценка пустого ответа. Ошибки ограничены кодом и коротким сообщением без путей и отладочной информации.

## Данные эпизода

```
episodes/<split>/<episode_id>/
  manifest.json  policy.json  ledger.csv  invoices.jsonl  payments.jsonl  fx.csv   # открытые
  _grader/truth.json  _grader/provenance.json                                    # закрытые
```

* время — ISO-8601 со смещением (счета — в часовом поясе компании, платежи и проводки — в UTC);
* деньги — целые минимальные единицы + код валюты (`JPY` — 0 знаков, `KWD` — 3);
* курсы — десятичные строки, `1 source = rate target`, только рабочие дни;
* отсутствующие ссылки — явный `null`.

Открытые сплиты: `configs/splits.json` (10 train + 10 dev, включая чистый и «взаимодействующий» эпизод в каждом).
Закрытая проверка использует другие seed и новые сочетания параметров (`id_salt`, `shuffle_rows`, другие пресеты).

## Команды CLI

```bash
python -m audit_env generate --split train --out episodes/train      # или --seeds 0-99 --config dev
python -m audit_env validate --seeds 0-199 --config interacting       # самопроверка генератора
python -m audit_env run --split dev --mode eval --out runs/dev        # траектории + grades + summary
python -m audit_env check                                             # единая проверка
python -m audit_env demo --seed 9 --config interacting
```

## Базовые политики

| политика | что делает |
|---|---|
| `random` | случайный исследователь с фиксированным seed, затем 0–2 случайные догадки |
| `reconciler` | простая сверка с наивными эвристиками (UTC-даты, дубли по сумме) |
| `flag_all` | атака «пометить всё» (30 находок по всем типам) |
| `oracle` | только для оценщика: отправляет эталон с доказательствами |
| `rules_auditor` | эталонный решатель по открытым правилам — доказывает разрешимость в бюджете шагов |

Последний `check` (20 скрытых-подобных эпизодов: 10 смешанных, 5 чистых, 5 взаимодействующих):

| policy | P | R | F1 | return | steps |
|---|---|---|---|---|---|
| random | 0.000 | 0.000 | 0.200 | -0.610 | 23.0 |
| reconciler | 0.160 | 0.347 | 0.216 | -0.583 | 13.1 |
| flag_all | 0.000 | 0.000 | 0.000 | -1.000 | 4.6 |
| oracle | 0.650 | 0.650 | 1.000 | 1.000 | 1.0 |
| rules_auditor | 0.650 | 0.650 | 1.000 | 0.976 | 13.1 |

P и R усреднены по эпизодам; на чистых эпизодах они по определению равны 0, а F1 = 1 при пустом ответе.

## Структура

```
audit_env/
  generator/   world.py (модель и проводки) history.py (история, отвлекающие случаи)
               errors.py (инъекции и сценарии) episode.py (рендер, ID, эталон, provenance)
  grader/      matcher.py (схема, нормализация, сопоставление) reward.py (метрики, награда)
  env.py       контракт reset/step и инструменты      server.py  stdio-изоляция
  rules.py     правила по открытым данным             validate.py самопроверки эпизодов
  policies/    базовые политики                       runner.py  траектории и сводки
configs/       версионированные пресеты и сплиты
tests/         тесты среды, генератора, сопоставления и награды
```

Подробности устройства, правила сопоставления и ограничения — в [DESIGN.md](DESIGN.md).
Лицензия — MIT.
