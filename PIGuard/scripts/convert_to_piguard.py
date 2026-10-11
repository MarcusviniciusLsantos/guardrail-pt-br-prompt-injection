import argparse
import json
import os
import random

import pandas as pd

SPLITS = ["NotInject_one", "NotInject_two", "NotInject_three"]

TRANSLATED_DIR = os.path.join(os.path.dirname(__file__), "translated")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "pt_br")


def load_split(model: str, split: str) -> pd.DataFrame:
    path = os.path.join(TRANSLATED_DIR, model, f"{split}_pt_br.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_parquet(path)


def to_eval_records(df: pd.DataFrame) -> list[dict]:
    return [
        {"prompt": row["prompt"], "label": False, "category": row["category"]}
        for _, row in df.iterrows()
    ]


def to_train_records(df: pd.DataFrame) -> list[dict]:
    return [{"prompt": row["prompt"], "label": 0} for _, row in df.iterrows()]


def save_json(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  Salvo: {path} ({len(records)} registros)")


def main():
    parser = argparse.ArgumentParser(description="Converte parquets traduzidos para o formato PIGuard.")
    parser.add_argument("--model", default="gpt-4o", help="Subdiretório do modelo em scripts/translated/")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="Fração dos dados para validação (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = os.path.join(OUTPUT_DIR, args.model)

    print(f"\n=== Gerando arquivos de AVALIAÇÃO ===")
    all_train: list[dict] = []
    for split in SPLITS:
        df = load_split(args.model, split)
        eval_records = to_eval_records(df)
        save_json(eval_records, os.path.join(out, "eval", f"{split}.json"))
        all_train.extend(to_train_records(df))

    print(f"\n=== Gerando arquivos de TREINO / VALIDAÇÃO ===")
    random.shuffle(all_train)
    cut = int(len(all_train) * (1 - args.valid_ratio))
    train_records = all_train[:cut]
    valid_records = all_train[cut:]

    save_json(train_records, os.path.join(out, "train.json"))
    save_json(valid_records, os.path.join(out, "valid.json"))

    print(f"\nPronto! Total: {len(all_train)} amostras | treino={len(train_records)} | validação={len(valid_records)}")
    print(f"Diretório de saída: {os.path.abspath(out)}")
    print("\nPara usar no PIGuard:")
    print(f"  Avaliação : copie {os.path.join(out, 'eval', '*.json')} → PIGuard/datasets/")
    print(f"  Treinamento: python train.py --train_set {os.path.join(out, 'train.json')} --valid_set {os.path.join(out, 'valid.json')}")


if __name__ == "__main__":
    main()
