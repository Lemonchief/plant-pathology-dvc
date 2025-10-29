PY=python

# --- БАЗОВОЕ ---
setup:
	python -m venv .venv
	.\.venv\Scripts\python.exe -m pip install -U pip
	.\.venv\Scripts\python.exe -m pip install -r requirements.txt
	.\.venv\Scripts\dvc init -q

prepare:
	dvc repro prepare

augment:
	dvc repro augment

stats:
	dvc repro stats

train:
	dvc repro train

eval:
	dvc repro evaluate

all:
	dvc repro

# --- ЭКСПЕРИМЕНТЫ (RAW) ---
exp-raw-logreg:
	dvc exp run -S model.type=logreg -S data_path="data/raw"

exp-raw-cnn:
	dvc exp run -S model.type=simple_cnn -S data_path="data/raw"

exp-raw-resnet:
	dvc exp run -S model.type=resnet18 -S data_path="data/raw"

series-raw: exp-raw-logreg exp-raw-cnn exp-raw-resnet

# --- ЭКСПЕРИМЕНТЫ (AUGMENTED) ---
exp-aug-logreg:
	dvc exp run -S model.type=logreg -S data_path="data/augmented"

exp-aug-cnn:
	dvc exp run -S model.type=simple_cnn -S data_path="data/augmented"

exp-aug-resnet:
	dvc exp run -S model.type=resnet18 -S data_path="data/augmented"

series-aug: exp-aug-logreg exp-aug-cnn exp-aug-resnet

# --- ПОКАЗ/ЭКСПОРТ РЕЗУЛЬТАТОВ ---
exps-show:
	dvc exp show --precision 4

exps-csv:
	dvc exp show --precision 4 --csv > artifacts/experiments.csv

exps-show-compact:
    dvc exp show --precision 4 --only-changed --hide-workspace

open-live:
    start dvclive\report.html

# --- ЧИСТКА (wipe) ---
clean-outputs:
	rm -rf data/raw data/augmented artifacts metrics models plots dvclive dvc.lock

# 2) подчистить историю экспериментов (и мусорные очереди)
clean-exps:
	dvc exp remove -A || true     # удалить все выполненные эксперименты (DVC 3.x)
	dvc exp clean || true         # подчистить временные файлы очереди

# 3) собрать мусор в локальном кэше DVC — оставить только то, что нужно воркспейсу
clean-cache:
	dvc gc --workspace -f || true

# Полный wipe
clean-all: clean-exps clean-outputs clean-cache

# --- ЗАПУСК С НУЛЯ УДОБНЫМИ ШОРТАМИ ---
fresh-raw: clean-all
	dvc exp run -S data_path="data/raw"

fresh-aug: clean-all
	dvc repro augment
	dvc exp run -S data_path="data/augmented"

help:
	@echo "Targets:"
	@echo "  make clean-all        # wipe эксперименты, артефакты и неиспользуемый DVC cache"
	@echo "  make fresh-raw        # полный прогон пайплайна на RAW"
	@echo "  make fresh-aug        # создать augmented и прогнать пайплайн"
	@echo "  make series-raw       # запустить 3 модели на RAW"
	@echo "  make series-aug       # запустить 3 модели на AUG"
	@echo "  make exps-show        # таблица экспериментов"
	@echo "  make exps-csv         # экспорт таблицы в CSV (artifacts/experiments.csv)"
	@echo "  make open-live        # открыть HTML-отчет DVCLive"
.PHONY: setup prepare augment stats train eval all \
        exp-raw-logreg exp-raw-cnn exp-raw-resnet series-raw \
        exp-aug-logreg exp-aug-cnn exp-aug-resnet series-aug \
        exps-show exps-csv exps-show-compact open-live \
        clean-outputs clean-exps clean-cache clean-all \
        fresh-raw fresh-aug help
