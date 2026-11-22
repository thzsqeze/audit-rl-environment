# audit-rl-environment

Воспроизводимая RL-среда для дообучения и оценки ИИ-аудитора (кейс FinTech №1).
Среда генерирует согласованную скрытую историю вымышленной компании за один отчётный период,
вносит в книги ошибки с записанным причинным изменением и оценивает выводы агента с доказательствами.

* runtime — только стандартная библиотека Python ≥ 3.11: без сети, GPU и платных API;
* контракт в стиле Gymnasium и протокол жюри (`judge_adapter.py`, [определения проверок](genesis-fintech-checks.md));
* 4 типа ошибок, 2 сценария взаимодействующих ошибок, чистые компании и 9 видов отвлекающих случаев;
* общий matcher и награда `clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)` с тестами на атакующие ответы;
* все данные синтетические: никаких реальных компаний, людей, счетов и платежей.

## Единая команда проверки

```bash
docker build -t audit-rl-env .
docker run --rm --network none --cpus 2 --memory 4g audit-rl-env      # == python -m audit_env check
```

Рабочая директория в контейнере — `/app` (корень репозитория). Локально: `make install && make check`.

`check` делает: тесты (`pytest`, 260+ тестов, в т. ч. копии опубликованных блоков C01–C20, matcher 1–20,
M01–M12, фаззинг 2 000 действий, изоляция, схемы) → самопроверку эпизодов каждого пресета → генерацию матрицы
D01–D20 в каталоги и прогон 4 базовых политик (+ `rules_auditor`) в закрытом режиме через `episode_dir`, каждая
политика — в отдельном заблокированном процессе, с поиском утечек → валидацию файлов и траекторий по `schemas/` →
измеренный разбор сбоев (`reports/failure_analysis.md`) → 20 парных проверок P01–P20 (перестановка / смена ID /
оба) на файлах. Отчёт — `reports/check_report.json` (+ `trajectories.jsonl`, `grades.jsonl`), ненулевой код при провале.

Сценарий показа (reset → чтение → submit, закрытый режим, изоляция, stdio-сервер, разбор сбоев, Docker) — [docs/demo.md](docs/demo.md); измеренный разбор сбоев — [docs/failure_analysis.md](docs/failure_analysis.md).

## Подключение жюри (`judge_adapter.py`)

```python
import judge_adapter as ja

ja.generate_episode(seed, config, output_dir)   # config: пресет, dict конфигурации или {"team_config": ...}
env = ja.make_env()
obs, info = env.reset(seed, {"mode": "closed", "episode_dir": output_dir})   # или "train"
episode_id = obs["manifest"]["episode_id"]
obs, reward, terminated, truncated, info = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": None}})
obs, reward, terminated, truncated, info = env.step(
    {"name": "submit", "args": {"findings": {"episode_id": episode_id, "findings": [...]}}})
ja.score_episode(output_dir, findings_json, inspection_steps)
# -> {"tp", "fp", "fn", "precision", "recall", "f1", "return"}; невалидный submit: return=-1, остальное null
```

* `reset` с `episode_dir` загружает именно этот эпизод (ID и порядок не меняются), повторный `reset` начинает заново;
  с `team_config` — генерирует. Без `mode` можно передать конфиг напрямую: `env.reset(7, "dev")`.
* Наблюдения: после `reset` — `{task, manifest, policy, tools, remaining_steps: 100}`;
  после `step` — `{result, error, remaining_steps}`, где `error` = `null` или `{code, message}` (без путей, seed, эталона).
* `closed`: награда политике всегда `0.0`, `info = {}`; отчёт оценщика — только `env.grader_report()` после конца
  эпизода и `score_episode`. `train`: терминальная награда и `info["grade"]` доступны политике.
* Невалидный submit записывается в журнал оценщика (`AUDIT_GRADER_LOG`, по умолчанию stderr).
* Изоляция процессов: `python -m audit_env serve --mode closed --grader-log grader.jsonl` — JSON-lines по stdin/stdout,
  оценка пишется только в журнал; путь в качестве config запрещён.

## Действия

| name | args | result |
|---|---|---|
| `list_records` | `table`, `cursor?` (null/строка), `limit?` (1..50) | `{"records": [...], "next_cursor": null \| "<table>:<offset>"}` |
| `get_record` | `table`, `id` | объект записи; `id`: `row_id` / `invoice_id` / `payment_id` / `FX:<date>:<src>:<dst>` |
| `get_document` | `doc_id` | документ счёта/платежа; также `"manifest"`, `"policy"` |
| `calculator` | `expression` | десятичная строка, только числа и `+ - * / ( )` (`"0.1+0.2"` → `"0.3"`) |
| `submit` | `findings` (объект `findings.json`) | `{"accepted": true}` или ошибка структуры; `terminated=true` |

Таблицы: `ledger`, `invoices`, `payments`, `fx`. Любой шаг (в т. ч. ошибочный) расходует бюджет; submit входит в лимит
100 действий, но не в `inspection_steps`; 100 исследовательских действий без submit → `truncated=true` и пустой ответ;
60 с на эпизод.

## Данные эпизода

```
<episode_dir>/manifest.json policy.json ledger.csv invoices.jsonl payments.jsonl fx.csv   # открытые
<episode_dir>/truth.json provenance.json                                                 # закрытые
```

* время — ISO-8601 со смещением (счета — в часовом поясе компании, платежи и проводки — в UTC);
  деловая дата и границы периода — по `manifest.timezone`;
* деньги — целые минимальные единицы + код валюты (`JPY` — 0 знаков, `KWD` — 3, остальные — 2);
* курсы — десятичные строки, `1 source = rate target`, только рабочие дни; правило выбора даты — в `policy.fx`;
* отсутствующие ссылки — явный `null` (в CSV — строка `null`);
* дополнительные поля: `ledger.journal_id/txn_currency/txn_amount_minor/fx_rate`, `payments.entity_id/direction/refund_of`;
* `provenance.json`: журнал генерации, для каждой ошибки — проводки до/после изменения в публичных ID,
  а также `clean_books` — книга тех же документов до внесения ошибок.

## Схемы (`schemas/`)

JSON Schema draft 2020-12 для всех обменов и файлов: [`manifest`](schemas/manifest.schema.json),
[`policy`](schemas/policy.schema.json), [`ledger_row`](schemas/ledger_row.schema.json),
[`invoice`](schemas/invoice.schema.json), [`payment`](schemas/payment.schema.json), [`fx_row`](schemas/fx_row.schema.json),
[`findings`](schemas/findings.schema.json) (объект submit), [`action`](schemas/action.schema.json),
[`observation_reset`](schemas/observation_reset.schema.json), [`observation_step`](schemas/observation_step.schema.json),
[`truth`](schemas/truth.schema.json) (закрытый), [`trajectory_line`](schemas/trajectory_line.schema.json);
общие форматы — [`common`](schemas/common.schema.json). Проверяются офлайн собственным валидатором подмножества
draft 2020-12 (`audit_env/schema.py`, только stdlib; неизвестные ключевые слова схемы — ошибка загрузки):
`tests/test_schemas.py` (эпизоды, траектории, живые наблюдения, findings oracle, негативные случаи, совпадение
схемы findings с `matcher.parse_submission`) и шаг `schemas` в `check` (матрица D01–D20 и её траектории).

```python
from audit_env.schema import SchemaStore
SchemaStore().errors(findings_json, "findings")   # [] = валидно
```

## Типы ошибок и канонические ответы

| type | affected_ids | claimed_amount_minor / currency |
|---|---|---|
| `duplicate_invoice` | оба `invoice_id` | сумма счёта / валюта счёта |
| `payment_allocation` | `[payment_id, invoice_id]` | недоприменённая сумма / валюта платежа |
| `fx_conversion` | `[invoice_id]` или `[payment_id]` | \|проведено − по policy\| / базовая валюта |
| `period_cutoff` | `[invoice_id]` (счёт) или `[payment_id]` (поступление) | сумма документа / валюта документа |

Сравнение точное: ID, тип и валюта — строки как есть, сумма — целое, допуск τ = 0 для всех типов
(ожидаемые суммы вычисляются по policy без неоднозначного округления). Подробно — [DESIGN.md](DESIGN.md).

## Конфигурации (`configs/`)

Пресет — имя файла; dict может наследовать пресет через `"extends"`. Каждый тип включается независимо.

| нужен эпизод | config |
|---|---|
| чистый, обычная сверка / разбитый платёж / возврат / повторяющиеся суммы / граница периода и TZ | `clean_basic` / `clean_split_payment` / `clean_refund` / `clean_recurring` / `clean_boundary` |
| чистый, «почти дубль» (тот же клиент, строки и сумма, другой PO) | `clean_near_duplicate` |
| чистый, частичный возврат переплаты (остаток — кредит клиента в 2200) | `clean_partial_refund` |
| чистый, поступление следующего периода (опубликовано, корректно не проведено) | `clean_next_period_receipt` |
| одна ошибка заданного типа | `type_duplicate_invoice`, `type_payment_allocation[_missing\|_partial]`, `type_fx_conversion`, `type_period_cutoff` |
| вариант FX: счёт пересчитан по курсу даты срока оплаты | `type_fx_conversion_due_date` |
| вариант периода: поступление следующего периода проведено последним днём текущего | `type_period_cutoff_receipt` |
| сценарий взаимодействия 1: частичный валютный платёж на границе периода (+ курс UTC-даты) | `scenario_partial_fx_payment_boundary` |
| сценарий взаимодействия 2: дубль счёта + платёж зачтён на дубль | `scenario_duplicate_with_misapplied_payment` |
| смешанный | `default`, `train`, `dev`, `interacting`, `clean` |
| верх предложенных диапазонов: ≤ 200 документов, 12 ошибок (~330–370 проводок) | `large` |
| нагрузочный: ~1 900 проводок, ~950 документов (выше предложенных 200 документов; только для лимитов) | `stress` |
| безвредные преобразования (внутри генератора) | `{"extends": "...", "id_salt": "x", "shuffle_rows": true}` |

Основные поля: `error_counts`, `scenario_counts`, `distractor_counts` (диапазоны `[min, max]`), `clean_probability`,
`fx_variants`, `allocation_variants`, `cutoff_variants`, `base_currencies`, `timezones`, `months`, размеры.

## Прочие команды

```bash
python -m audit_env generate --split train --out episodes/train      # или --seeds 0-99 --config dev
python -m audit_env validate --seeds 0-199 --config interacting       # самопроверка генератора
python -m audit_env run --split dev --mode closed --out runs/dev      # trajectories.jsonl + grades.jsonl + summary.json
python -m audit_env demo --seed 9 --config interacting                # скриптовая политика на эпизоде
```

Открытые сплиты (`configs/splits.json`): 10 train + 10 dev эпизодов в `episodes/`, траектории базовых политик —
`artifacts/{train,dev}/trajectories.jsonl` (поля `episode_id, step, action, observation, reward, terminated, truncated, info`).

## Базовые политики

| policy | что делает |
|---|---|
| `random` | случайный исследователь с фиксированным seed, затем 0–2 случайные догадки |
| `reconciler` | простая сверка с наивными эвристиками (UTC-даты, дубли по сумме) |
| `flag_all` | атака «пометить всё» (30 находок по всем типам) |
| `oracle` | только для оценщика: отправляет эталон с доказательствами обычным submit |
| `rules_auditor` | эталонный решатель по открытым правилам — показывает разрешимость в бюджете шагов |

Последний `check` (ENV_VERSION 0.3.0), закрытый режим, политики в отдельных процессах, матрица D01–D20
(5 чистых, 10 с одной ошибкой, 5 взаимодействующих):

| policy | P | R | F1 | return | exp. return | steps | time, s |
|---|---|---|---|---|---|---|---|
| random | 0.000 | 0.000 | 0.200 | -0.610 | -0.610 | 22.6 | 0.212 |
| reconciler | 0.052 | 0.375 | 0.189 | -0.635 | -0.633 | 12.8 | 0.223 |
| flag_all | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | 4.3 | 0.201 |
| oracle | 0.750 | 0.750 | 1.000 | 1.000 | 1.000 | 1.0 | 0.206 |
| rules_auditor | 0.750 | 0.750 | 1.000 | 0.976 | 0.980 | 12.8 | 0.209 |

Средние по эпизодам; на чистых эпизодах P = R = 0 по определению, F1 = 1 при пустом ответе. `return` — общая
награда §05, `exp. return` — экспериментальная (DESIGN §8). Время включает запуск процесса политики (~0,2 с).
Это диагностика, а не заявление о качестве модели. Полный разбор — [docs/failure_analysis.md](docs/failure_analysis.md).

## Структура

```
judge_adapter.py  точки входа жюри
audit_env/
  generator/   world.py (модель и проводки) history.py (история, отвлекающие случаи)
               errors.py (инъекции и сценарии) episode.py (рендер, ID, эталон, provenance, load/write)
  grader/      matcher.py (схема, сопоставление) reward.py (метрики, награда)
  env.py       контракт reset/step и инструменты      server.py    stdio-изоляция
  rules.py     правила по открытым данным             validate.py  самопроверки эпизодов
  transform.py перестановка/смена ID на файлах        policies/    базовые политики, runner.py
configs/  пресеты и сплиты      tests/  тесты      episodes/  открытые эпизоды      artifacts/  траектории
```

Версии: `audit_env/version.py` (`ENV_VERSION`, `SCHEMA_VERSION`), `config_version` в пресетах, зависимости
закреплены в `requirements-dev.txt`. Лицензия — MIT.
