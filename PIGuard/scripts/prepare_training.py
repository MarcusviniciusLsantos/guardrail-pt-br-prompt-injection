"""
Prepara o dataset de treinamento para o PIGuard mesclando os dados originais
com as amostras NotInject traduzidas para pt-BR.

Modos (--mode):
  augment   [padrão] — adiciona as amostras pt-BR ao dataset original
  ptbr_only — usa apenas injeções do original + benign pt-BR (experimento isolado)

Saída (dentro de PIGuard/datasets/):
  train_ptbr.json
  valid_ptbr.json
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
PIGUARD_DS = ROOT / "PIGuard" / "datasets"
PTBR_DIR = ROOT / "datasets" / "pt_br"


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    labels = [r["label"] for r in data]
    print(f"  Salvo: {path.name} | {len(data)} amostras | label=0: {labels.count(0)}, label=1: {labels.count(1)}")


def load_ptbr_samples(model: str) -> list[dict]:
    train_path = PTBR_DIR / model / "train.json"
    valid_path = PTBR_DIR / model / "valid.json"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Não encontrado: {train_path}\n"
            f"Execute: uv run python scripts/convert_to_piguard.py --model {model}"
        )
    samples = load_json(train_path) + load_json(valid_path)
    return [{"prompt": s["prompt"], "label": s["label"]} for s in samples]


def load_ptbr_injections(model: str) -> list[dict]:
    path = ROOT / "scripts" / "translated" / model / "injections_pt_br.json"
    if not path.exists():
        return []
    samples = load_json(path)
    print(f"  Injeções pt-BR carregadas: {len(samples)} de {path}")
    return [{"prompt": s["prompt"], "label": 1} for s in samples]


def split_train_valid(data: list[dict], valid_ratio: float, seed: int) -> tuple[list, list]:
    random.seed(seed)
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * (1 - valid_ratio))
    return data[:cut], data[cut:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o", help="Modelo de tradução (subdir em datasets/pt_br/)")
    parser.add_argument(
        "--mode",
        choices=["augment", "ptbr_only"],
        default="augment",
        help="augment: adiciona pt-BR ao original | ptbr_only: só injeções orig + benign pt-BR",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nModo: {args.mode} | Modelo de tradução: {args.model}")

    ptbr_benign = load_ptbr_samples(args.model)
    ptbr_inject = load_ptbr_injections(args.model)
    print(f"\nAmostras pt-BR: {len(ptbr_benign)} benign + {len(ptbr_inject)} injeções")

    orig_train = load_json(PIGUARD_DS / "train.json")
    orig_valid = load_json(PIGUARD_DS / "valid.json")
    orig_all = orig_train + orig_valid

    if args.mode == "augment":
        combined = orig_all + ptbr_benign + ptbr_inject
        print(f"Dataset combinado: {len(orig_all)} (original) + {len(ptbr_benign)} benign (pt-BR) + {len(ptbr_inject)} injeções (pt-BR) = {len(combined)}")

    elif args.mode == "ptbr_only":
        injection_only = [r for r in orig_all if r["label"] == 1]
        combined = injection_only + ptbr_inject + ptbr_benign
        print(f"Dataset pt-BR only: {len(injection_only)} injeções (orig) + {len(ptbr_inject)} injeções (pt-BR) + {len(ptbr_benign)} benign (pt-BR) = {len(combined)}")

    train_data, valid_data = split_train_valid(combined, args.valid_ratio, args.seed)

    out_train = PIGUARD_DS / f"train_{args.mode}_{args.model}.json"
    out_valid = PIGUARD_DS / f"valid_{args.mode}_{args.model}.json"

    print()
    save_json(train_data, out_train)
    save_json(valid_data, out_valid)

    print(f"\nPara treinar, execute dentro de PIGuard/:")
    print(f"  cd PIGuard")
    print(f"  python train.py \\")
    print(f"    --name {args.mode}_{args.model} \\")
    print(f"    --train_set datasets/{out_train.name} \\")
    print(f"    --valid_set datasets/{out_valid.name}")


if __name__ == "__main__":
    main()
