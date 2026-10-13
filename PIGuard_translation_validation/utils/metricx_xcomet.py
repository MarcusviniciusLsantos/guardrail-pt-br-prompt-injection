"""
Quality Estimation supervisionado sem referência.

Padrão: XCOMET (Unbabel) — API Python direta e estável, recomendada aqui
por simplicidade. Instalação: pip install unbabel-comet

Alternativa: MetricX-24 (Google) — SOTA nas comparações mais recentes (WMT24),
mas só é distribuída via CLI do repositório oficial (precisa ser clonado e
instalado à parte: https://github.com/google-research/metricx). A função
`run_metricx24_cli` abaixo é um wrapper de conveniência; confira o README do
repo clonado para os nomes exatos dos argumentos na versão que você instalar,
pois esse detalhe muda entre releases.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

from comet import download_model, load_from_checkpoint


class XCometScorer:
    def __init__(self, model_name: str = "Unbabel/XCOMET-XL", gpus: int = 1):
        model_path = download_model(model_name)
        self.model = load_from_checkpoint(model_path)
        self.gpus = gpus

    def score_batch(
        self, src_texts: list[str], tr_texts: list[str], batch_size: int = 8
    ) -> dict[str, list]:
        """Modo QE (reference-free): só usa fonte (src) e tradução (mt)."""
        data = [{"src": s, "mt": t} for s, t in zip(src_texts, tr_texts)]
        output = self.model.predict(data, batch_size=batch_size, gpus=self.gpus)
        scores = getattr(output, "scores", None)
        if scores is None:
            scores = getattr(output, "segment_scores", None)
        if scores is None:
            raise RuntimeError(
                "XCOMET nao retornou scores por segmento; "
                f"campos disponiveis: {sorted(vars(output)) if hasattr(output, '__dict__') else type(output)}"
            )

        scores = list(scores)
        if len(scores) != len(data) or any(
            score is None or not math.isfinite(float(score)) for score in scores
        ):
            raise RuntimeError(
                "XCOMET retornou scores invalidos: "
                f"esperados={len(data)}, recebidos={len(scores)}, amostra={scores[:3]}"
            )
        metadata = getattr(output, "metadata", None)
        error_spans = getattr(metadata, "error_spans", None)
        if error_spans is None:
            error_spans = [[] for _ in data]
        else:
            error_spans = list(error_spans)

        if len(error_spans) != len(data):
            raise RuntimeError(
                "XCOMET retornou quantidade invalida de error_spans: "
                f"esperados={len(data)}, recebidos={len(error_spans)}"
            )

        return {
            "xcomet_qe": [float(score) for score in scores],
            "xcomet_error_spans": error_spans,
        }


def run_metricx24_cli(
    src_texts: list[str],
    tr_texts: list[str],
    model_name_or_path: str = "google/metricx-24-hybrid-xl-v2p6",
    tokenizer: str = "google/mt5-xl",
    max_input_length: int = 1536,
    batch_size: int = 1,
) -> list[float]:
    """Alternativa opcional ao XCOMET. Requer o pacote metricx24 instalado
    (clone https://github.com/google-research/metricx e `pip install -e .`).
    Modo QE: passamos reference vazia, já que só temos fonte + tradução.
    """
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "input.jsonl"
        out_path = Path(tmp) / "output.jsonl"

        with in_path.open("w", encoding="utf-8") as f:
            for s, t in zip(src_texts, tr_texts):
                f.write(json.dumps({"source": s, "hypothesis": t, "reference": ""}) + "\n")

        subprocess.run(
            [
                "python", "-m", "metricx24.predict",
                "--tokenizer", tokenizer,
                "--model_name_or_path", model_name_or_path,
                "--max_input_length", str(max_input_length),
                "--batch_size", str(batch_size),
                "--input_file", str(in_path),
                "--output_file", str(out_path),
            ],
            check=True,
        )

        scores = []
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                scores.append(json.loads(line)["prediction"])
    return scores
