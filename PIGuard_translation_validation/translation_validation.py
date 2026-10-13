"""
Validação de qualidade de tradução EN -> PT-BR das bases de prompt-injection.

Lê pares de arquivos (og/<nome> e translated_pt-br_gemma-4-31B/<nome>),
normaliza as 3 estruturas de dataset diferentes, roda 3 métricas
(SONAR/BLASER-2, XCOMET, chrF) e salva um .jsonl por dataset em --output-dir.
As métricas pesadas são opcionais para permitir uma instalação enxuta:
`--metrics sonar,chrf` não carrega XCOMET.

Resumível: se o job cair no meio no cluster, rodar de novo continua de
onde parou (conta quantas linhas já existem no .jsonl de saída e pula
esses itens).

Uso:
    python translation_validation.py \
        --og-dir datasets/pi-guard-experiment/og \
        --tr-dir datasets/pi-guard-experiment/translated_pt-br_gemma-4-31B \
        --output-dir outputs/translation_validation \
        --batch-size 16 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_DATASETS = [
    "BIPIA_code.json",
    "BIPIA_text.json",
    "NotInject_one.json",
    "NotInject_two.json",
    "NotInject_three.json",
    "train.json",
    "valid.json",
    "wildguard.json",
]


# --------------------------------------------------------------------------
# Carregamento e normalização das 3 estruturas de dataset
# --------------------------------------------------------------------------

def _load_raw(path: Path):
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    if path.suffix == ".parquet":
        from importlib import import_module

        pd = import_module("pandas")
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Formato não suportado: {path.suffix}")


def load_items(path: Path) -> list[dict]:
    """Normaliza qualquer uma das 3 estruturas em:
    [{"id": str, "text": str, "meta": {...}}, ...]
    """
    raw = _load_raw(path)

    # Formato BIPIA: {"categoria": ["texto", "texto", ...], ...}
    if isinstance(raw, dict):
        items = []
        for category, texts in raw.items():
            for idx, text in enumerate(texts):
                items.append({"id": f"{category}__{idx}", "text": text, "meta": {"category": category}})
        return items

    # Formato lista: train/valid/wildguard (prompt/label/source)
    # e NotInject (prompt/word_list/category) — ambos usam a chave "prompt".
    if isinstance(raw, list):
        items = []
        for idx, entry in enumerate(raw):
            meta = {k: v for k, v in entry.items() if k != "prompt"}
            items.append({"id": str(idx), "text": entry["prompt"], "meta": meta})
        return items

    raise ValueError(f"Estrutura de dataset não reconhecida em {path}")


def align_pairs(og_items: list[dict], tr_items: list[dict]) -> list[dict]:
    """Alinha por posição (assume mesma ordem/tamanho nos dois arquivos)."""
    n = min(len(og_items), len(tr_items))
    if len(og_items) != len(tr_items):
        print(f"  [aviso] tamanhos diferentes (og={len(og_items)}, tr={len(tr_items)}) "
              f"— usando os {n} primeiros pares")
    pairs = []
    for i in range(n):
        og, tr = og_items[i], tr_items[i]
        pairs.append({
            "id": og["id"],
            "source_text": og["text"],
            "translated_text": tr["text"],
            "meta": og["meta"],
        })
    return pairs


def validate_datasets(dataset_names: list[str], og_dir: Path, tr_dir: Path) -> int:
    """Valida arquivos, estruturas e alinhamento sem carregar modelos."""
    total = 0
    for dataset_name in dataset_names:
        name = dataset_name if Path(dataset_name).suffix else f"{dataset_name}.json"
        og_path, tr_path = og_dir / name, tr_dir / name
        if not og_path.exists() or not tr_path.exists():
            raise FileNotFoundError(f"{name}: arquivo ausente em og/ ou translated/")
        pairs = align_pairs(load_items(og_path), load_items(tr_path))
        if not pairs:
            raise ValueError(f"{name}: nenhum par encontrado")
        total += len(pairs)
        print(f"[ok] {name}: {len(pairs)} pares alinhados")
    print(f"Preflight concluído: {total} pares em {len(dataset_names)} datasets")
    return total


# --------------------------------------------------------------------------
# Processamento com retomada (resume) por item
# --------------------------------------------------------------------------

def already_done_count(output_path: Path) -> int:
    if not output_path.exists():
        return 0
    with output_path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def process_dataset(name: str, og_dir: Path, tr_dir: Path, output_dir: Path,
                     scorers: dict, batch_size: int) -> None:
    from utils.chrf_metric import chrf_score_batch

    og_path, tr_path = og_dir / name, tr_dir / name
    if not og_path.exists() or not tr_path.exists():
        print(f"[skip] {name}: não encontrado em og/ ou translated/")
        return

    pairs = align_pairs(load_items(og_path), load_items(tr_path))
    output_path = output_dir / f"{Path(name).stem}.jsonl"
    done = already_done_count(output_path)

    if done >= len(pairs):
        print(f"[ok] {name}: já processado por completo ({done} itens) — pulando")
        return

    pending = pairs[done:]
    print(f"[run] {name}: {done} já processados, {len(pending)} restantes")

    with output_path.open("a", encoding="utf-8") as out_f:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            src_texts = [p["source_text"] for p in batch]
            tr_texts = [p["translated_text"] for p in batch]

            sonar_out = scorers["sonar"].score_batch(src_texts, tr_texts) if "sonar" in scorers else None
            xcomet_out = scorers["xcomet"].score_batch(src_texts, tr_texts, batch_size=len(batch)) if "xcomet" in scorers else None
            chrf_out = chrf_score_batch(tr_texts, src_texts)

            if xcomet_out is not None and len(xcomet_out["xcomet_qe"]) != len(batch):
                raise RuntimeError(
                    f"XCOMET retornou {len(xcomet_out['xcomet_qe'])} scores para {len(batch)} pares em {name}"
                )

            for i, p in enumerate(batch):
                record = {
                    "id": p["id"],
                    "meta": p["meta"],
                    "sonar_cosine": sonar_out["sonar_cosine"][i] if sonar_out else None,
                    "blaser_qe": sonar_out["blaser_qe"][i] if sonar_out else None,
                    "xcomet_qe": xcomet_out["xcomet_qe"][i] if xcomet_out is not None else None,
                    "xcomet_error_spans": xcomet_out["xcomet_error_spans"][i] if xcomet_out is not None else None,
                    "chrf": chrf_out[i],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()  # grava em disco a cada lote — importante para o cluster

            processed = done + min(start + batch_size, len(pending))
            print(f"  ... {processed}/{len(pairs)}")


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--og-dir", default="datasets/pi-guard-experiment/og")
    parser.add_argument("--tr-dir", default="datasets/pi-guard-experiment/translated_pt-br_gemma-4-31B")
    parser.add_argument("--output-dir", default="outputs/translation_validation")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--xcomet-model", default="Unbabel/XCOMET-XL")
    parser.add_argument(
        "--metrics",
        default="sonar,xcomet,chrf",
        help="Metricas separadas por virgula: sonar, xcomet, chrf. Padrao: sonar,xcomet,chrf.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Valida arquivos e alinhamento sem carregar SONAR/XCOMET.",
    )
    args = parser.parse_args()

    og_dir, tr_dir = Path(args.og_dir), Path(args.tr_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_datasets(args.datasets, og_dir, tr_dir)
    if args.validate_only:
        return

    metrics = {metric.strip().lower() for metric in args.metrics.split(",") if metric.strip()}
    allowed_metrics = {"sonar", "xcomet", "chrf"}
    unknown_metrics = metrics - allowed_metrics
    if unknown_metrics or not metrics:
        raise ValueError(f"Metricas invalidas: {sorted(unknown_metrics)}. Use sonar, xcomet e/ou chrf.")

    print("Carregando modelos (uma única vez para todos os datasets)...")
    scorers = {}
    if "sonar" in metrics:
        from utils.sonar_blaser import SonarBlaserScorer
        scorers["sonar"] = SonarBlaserScorer(device=args.device)
    if "xcomet" in metrics:
        from utils.metricx_xcomet import XCometScorer
        scorers["xcomet"] = XCometScorer(model_name=args.xcomet_model, gpus=1 if args.device == "cuda" else 0)

    for dataset_name in args.datasets:
        name = dataset_name if Path(dataset_name).suffix else f"{dataset_name}.json"
        process_dataset(name, og_dir, tr_dir, output_dir, scorers, args.batch_size)

    print("Concluído.")


if __name__ == "__main__":
    main()
