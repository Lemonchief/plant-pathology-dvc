import pandas as pd
from pathlib import Path
from shutil import copy2
from sklearn.model_selection import train_test_split
from src.utils import CLASSES

# Пути
RAW_DIR = Path("data/raw")
TRAIN_DIR = RAW_DIR / "train"
VAL_DIR = RAW_DIR / "val"
TEST_DIR = RAW_DIR / "test"
IMAGES_DIR = Path("data/images")  # здесь лежат все JPG
CSV_DIR = Path("data/csv")        # если ты туда положишь train.csv и test.csv
PLACEHOLDER_DIR = TEST_DIR / "placeholder"

# Создаём нужные директории
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# Загружаем данные
train_df = pd.read_csv(CSV_DIR / "train.csv")
test_df = pd.read_csv(CSV_DIR / "test.csv")

# Создаём подпапки для каждого класса
for cls in CLASSES:
    (TRAIN_DIR / cls).mkdir(parents=True, exist_ok=True)

# Разбиваем train на обучающую и валидационную выборки
train_split, val_split = train_test_split(train_df, test_size=0.2, random_state=42)

# Копируем train
for _, row in train_split.iterrows():
    img_name = f"{row['image_id']}.jpg"
    img_src = IMAGES_DIR / img_name
    label = row[CLASSES].idxmax()
    img_dst = TRAIN_DIR / label / img_name
    copy2(img_src, img_dst)

# Копируем валидацию
for cls in CLASSES:
    (VAL_DIR / cls).mkdir(parents=True, exist_ok=True)

for _, row in val_split.iterrows():
    img_name = f"{row['image_id']}.jpg"
    img_src = IMAGES_DIR / img_name
    label = row[CLASSES].idxmax()
    img_dst = VAL_DIR / label / img_name
    copy2(img_src, img_dst)

# Копируем тестовые
PLACEHOLDER_DIR.mkdir(parents=True, exist_ok=True)
for _, row in test_df.iterrows():
    img_name = PLACEHOLDER_DIR / f"{row['image_id']}.jpg"
    img_src = IMAGES_DIR / f"{row['image_id']}.jpg"
    copy2(img_src, img_name)
