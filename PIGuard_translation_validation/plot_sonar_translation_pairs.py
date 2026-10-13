#!/usr/bin/env python3
"""Build an interactive review panel for SONAR translation pairs."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from sklearn.decomposition import IncrementalPCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi_guard_translation_validation.translation_validation import load_items

LANGUAGES = {"original_en", "translated_ptbr"}
METRICS = ("blaser_qe", "chrf", "xcomet_qe", "xcomet_error_spans")


def source_color_map(sources: set[str] | list[str]) -> dict[str, str]:
    ordered = sorted(sources)
    return {
        source: to_hex(plt.get_cmap("hsv")(index / max(len(ordered), 1) * .85))
        for index, source in enumerate(ordered)
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSON invalido em {path}:{number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Registro invalido em {path}:{number}")
            rows.append(row)
    return rows


def load_metrics(directory: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for path in sorted(directory.glob("*.jsonl")):
        for row in read_jsonl(path):
            if "id" not in row:
                raise ValueError(f"Registro sem id em {path}")
            key = (path.stem, str(row["id"]))
            if key in result:
                raise ValueError(f"Metrica duplicada: {key[0]}/{key[1]}")
            result[key] = {field: row.get(field) for field in METRICS}
    return result


def load_outliers(directory: Path) -> set[tuple[str, str]]:
    result = set()
    for path in sorted(directory.glob("*_outliers.jsonl")):
        dataset = path.name.removesuffix("_outliers.jsonl")
        for row in read_jsonl(path):
            if "id" not in row:
                raise ValueError(f"Registro sem id em {path}")
            result.add((dataset, str(row["id"])))
    return result


def load_texts(directory: Path, datasets: set[str]) -> dict[tuple[str, str], str]:
    result = {}
    for dataset in sorted(datasets):
        path = directory / f"{dataset}.json"
        if not path.exists():
            raise FileNotFoundError(f"Dataset ausente: {path}")
        for item in load_items(path):
            key = (dataset, str(item["id"]))
            if key in result:
                raise ValueError(f"ID duplicado no dataset: {dataset}/{key[1]}")
            if not isinstance(item.get("text"), str):
                raise ValueError(f"Texto invalido para {dataset}/{key[1]}")
            result[key] = item["text"]
    return result


def metadata_pairs(metadata: pd.DataFrame, vector_count: int) -> list[dict[str, Any]]:
    required = {"dataset", "id", "language", "computed_sonar_cosine"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata sem colunas obrigatorias: {sorted(missing)}")
    if len(metadata) != vector_count:
        raise ValueError(f"Contagem incompativel: metadata={len(metadata)}, embeddings={vector_count}")
    groups: dict[tuple[str, str], dict[str, pd.Series]] = defaultdict(dict)
    ordered = []
    for _, row in metadata.iterrows():
        key = (str(row["dataset"]), str(row["id"]))
        language = str(row["language"])
        if language not in LANGUAGES:
            raise ValueError(f"Idioma invalido para {key[0]}/{key[1]}: {language}")
        if key not in groups:
            ordered.append(key)
        if language in groups[key]:
            raise ValueError(f"Idioma duplicado para {key[0]}/{key[1]}: {language}")
        groups[key][language] = row
    count = len(ordered)
    stored_indexes = "embedding_index" in metadata.columns
    pairs = []
    for position, key in enumerate(ordered):
        group = groups[key]
        if set(group) != LANGUAGES:
            raise ValueError(f"Par sem os dois idiomas para {key[0]}/{key[1]}")
        original, translated = group["original_en"], group["translated_ptbr"]
        cosine, translated_cosine = float(original["computed_sonar_cosine"]), float(translated["computed_sonar_cosine"])
        if not math.isclose(cosine, translated_cosine, abs_tol=1e-6):
            raise ValueError(f"SONAR divergente entre idiomas para {key[0]}/{key[1]}")
        # Legacy compatibility: old NPY is [all EN, all PT-BR], while its CSV
        # alternates rows. New files include explicit indexes.
        oi = int(original["embedding_index"]) if stored_indexes else position
        ti = int(translated["embedding_index"]) if stored_indexes else count + position
        if not (0 <= oi < vector_count and 0 <= ti < vector_count):
            raise ValueError(f"Indice de embedding invalido para {key[0]}/{key[1]}")
        pairs.append({"key": key, "original_index": oi, "translated_index": ti, "metadata": original, "cosine": cosine})
    return pairs


def project(vectors: np.ndarray, batch_size: int) -> np.ndarray:
    if vectors.ndim != 2 or len(vectors) < 2:
        raise ValueError("Embeddings devem ser matriz com ao menos dois vetores")
    if batch_size < vectors.shape[1]:
        raise ValueError(f"--pca-batch-size deve ser >= {vectors.shape[1]}")
    # Keep every partial_fit batch at least as large as the embedding
    # dimension; IncrementalPCA cannot fit a smaller final batch.
    bounds = list(range(0, len(vectors), batch_size))
    if len(bounds) > 1 and len(vectors) - bounds[-1] < vectors.shape[1]:
        bounds.pop()
    bounds.append(len(vectors))
    pca = IncrementalPCA(n_components=2, batch_size=batch_size)
    for start, end in zip(bounds[:-1], bounds[1:]):
        pca.partial_fit(np.asarray(vectors[start:end], dtype=np.float32))
    output = np.empty((len(vectors), 2), dtype=np.float32)
    for start, end in zip(bounds[:-1], bounds[1:]):
        output[start:end] = pca.transform(np.asarray(vectors[start:end], dtype=np.float32))
    return output


def scalar(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def build_pairs(metadata: pd.DataFrame, vectors: np.ndarray, coords: np.ndarray, originals: dict, translations: dict, metrics: dict, outliers: set, threshold: float, datasets: set[str] | None, max_pairs: int) -> list[dict[str, Any]]:
    result = []
    for item in metadata_pairs(metadata, len(vectors)):
        key = item["key"]
        if datasets and key[0] not in datasets:
            continue
        if key not in originals or key not in translations:
            raise ValueError(f"Texto ausente para {key[0]}/{key[1]}")
        row, cosine = item["metadata"], item["cosine"]
        # The source embeddings are L2-normalized.  The norm of their midpoint
        # is therefore a direct, monotonic transform of EN/PT-BR cosine
        # similarity, but calculating it from the stored vectors also keeps the
        # diagnostic valid for legacy embedding files.
        centroid = (
            np.asarray(vectors[item["original_index"]], dtype=np.float32)
            + np.asarray(vectors[item["translated_index"]], dtype=np.float32)
        ) / 2
        
        # Corrige o mapeamento de source para o dataset wildguard que não possui
        # a coluna source explicitamente, caindo em 'unknown'.
        raw_source = scalar(row.get("source", "unknown"))
        if key[0] == "wildguard" and raw_source == "unknown":
            raw_source = "WildGuard"

        result.append({
            "dataset": key[0], "id": key[1], "label": scalar(row.get("label", "")), "source": raw_source,
            "outlier": key in outliers, "sonar_cosine": cosine, "sonar_distance": 1 - cosine,
            "centroid_norm": float(np.linalg.norm(centroid)), "needs_attention": cosine < threshold,
            "original_x": float(coords[item["original_index"], 0]), "original_y": float(coords[item["original_index"], 1]),
            "translated_x": float(coords[item["translated_index"], 0]), "translated_y": float(coords[item["translated_index"], 1]),
            "original_prompt": originals[key], "translated_prompt": translations[key], **metrics.get(key, {}),
        })
    result.sort(key=lambda item: (item["sonar_cosine"], item["dataset"], item["id"]))
    return result[:max_pairs] if max_pairs else result


def write_csv(pairs: list[dict], path: Path) -> None:
    fields = ["dataset", "id", "label", "source", "sonar_cosine", "sonar_distance", "centroid_norm", "needs_attention", "outlier", *METRICS, "original_prompt", "translated_prompt"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in pairs:
            row = {field: item.get(field) for field in fields}
            row["xcomet_error_spans"] = json.dumps(row["xcomet_error_spans"], ensure_ascii=False)
            writer.writerow(row)


def write_png(pairs: list[dict], path: Path, threshold: float, projection_label: str = "PCA") -> None:
    figure, (hist, plane) = plt.subplots(1, 2, figsize=(18, 7))
    hist.hist([p["sonar_cosine"] for p in pairs], bins=70, color="#4c78a8")
    hist.axvline(threshold, color="#e45756", linewidth=2, label=f"limiar {threshold:.2f}")
    hist.set(title="Distribuição de similaridade SONAR", xlabel="similaridade cosseno", ylabel="pares")
    hist.legend()
    candidates = [p for p in pairs if p["needs_attention"] or p["outlier"]]
    sources = sorted({str(p["source"]) for p in candidates})
    source_colors = source_color_map(sources)
    for p in candidates:
        color = "#e45756" if p["needs_attention"] else "#999999"
        plane.plot([p["original_x"], p["translated_x"]], [p["original_y"], p["translated_y"]], color=color, alpha=.5, linewidth=.6)
    for source in sources:
        group = [p for p in candidates if str(p["source"]) == source]
        color = source_colors[source]
        plane.scatter([p["original_x"] for p in group], [p["original_y"] for p in group], c=[color], s=7, alpha=.7, label=source)
        plane.scatter([p["translated_x"] for p in group], [p["translated_y"] for p in group], c=[color], marker="D", s=7, alpha=.7)
    plane.set(title="Pares para revisão / outliers (cor: source; círculo: EN; losango: PT-BR)", xlabel=f"{projection_label} 1", ylabel=f"{projection_label} 2")
    plane.legend(markerscale=2, fontsize="x-small")
    figure.tight_layout(); figure.savefig(path, dpi=220); plt.close(figure)


def write_overview_png(pairs: list[dict], path: Path, projection_label: str) -> None:
    """Write one point per pair, colored by source, without pair connections."""
    figure, plane = plt.subplots(figsize=(16, 9))
    sources = sorted({str(p["source"]) for p in pairs})
    source_colors = source_color_map(sources)
    for source in sources:
        group = [p for p in pairs if str(p["source"]) == source]
        plane.scatter(
            [p["original_x"] for p in group],
            [p["original_y"] for p in group],
            c=[source_colors[source]], s=3, alpha=.65, linewidths=0,
            rasterized=True, label=source,
        )
    plane.set(
        title=f"{projection_label} — centróide EN/PT-BR; cor: source; sem conexões",
        xlabel="UMAP 1", ylabel="UMAP 2",
    )
    plane.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="x-small", markerscale=2)
    figure.tight_layout(rect=(0, 0, .80, 1)); figure.savefig(path, dpi=220); plt.close(figure)


def write_translation_delta_png(points: list[dict], path: Path, projection_label: str) -> None:
    """Visualize directions of translation changes, by source and SONAR score."""
    figure, (by_source, by_sonar) = plt.subplots(1, 2, figsize=(20, 9))
    sources = sorted({str(point["source"]) for point in points})
    source_colors = source_color_map(sources)
    for source in sources:
        group = [point for point in points if str(point["source"]) == source]
        by_source.scatter(
            [point["x"] for point in group], [point["y"] for point in group],
            c=[source_colors[source]], s=4, alpha=.65, linewidths=0,
            rasterized=True, label=source,
        )
    scores = np.asarray([point["sonar_cosine"] for point in points], dtype=float)
    low, high = np.quantile(scores, [.01, .99])
    if low == high:
        low, high = float(scores.min()), float(scores.max()) + 1e-6
    scatter = by_sonar.scatter(
        [point["x"] for point in points], [point["y"] for point in points],
        c=scores, cmap="RdYlGn", vmin=low, vmax=high, s=4, alpha=.7,
        linewidths=0, rasterized=True,
    )
    by_source.set(title=f"{projection_label} — cor: source", xlabel="UMAP 1", ylabel="UMAP 2")
    by_sonar.set(title=f"{projection_label} — cor: SONAR", xlabel="UMAP 1", ylabel="UMAP 2")
    by_source.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="x-small", markerscale=2)
    colorbar = figure.colorbar(scatter, ax=by_sonar)
    colorbar.set_label("similaridade SONAR (escala q01–q99)")
    figure.suptitle("Vetores de diferença normalizados: PT-BR − EN; sem conexões", y=.99)
    figure.tight_layout(rect=(0, 0, .93, .97)); figure.savefig(path, dpi=220); plt.close(figure)


def short(text: str) -> str:
    value = " ".join(text.split())
    return value if len(value) <= 280 else value[:279] + "…"


def write_html(pairs: list[dict], path: Path, threshold: float, projection_label: str = "PCA") -> None:
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as error:
        raise RuntimeError("Plotly e obrigatorio; instale as dependencias.") from error
    data = []
    for p in pairs:
        # Full prompts stay in the local payload for the panel below the plot;
        # only the hover label uses the short snippets.
        item = {k: v for k, v in p.items() if k not in {"xcomet_error_spans"}}
        item["original_snippet"], item["translated_snippet"] = short(p["original_prompt"]), short(p["translated_prompt"])
        item["xcomet_error_count"] = len(p.get("xcomet_error_spans") or [])
        data.append(item)
    # JSON lives inside a script tag; escape HTML-significant characters so a
    # prompt such as </script> cannot terminate it.
    payload = (json.dumps(data, ensure_ascii=False, separators=(",", ":"))
               .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
               .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    source_colors = json.dumps(source_color_map({str(p["source"]) for p in pairs}), ensure_ascii=False, separators=(",", ":"))
    js = get_plotlyjs()
    path.write_text(f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Revisão SONAR</title>
<style>body{{font-family:system-ui;margin:0;color:#202124}}header,#controls,#details{{padding:12px 20px}}header{{background:#f5f7fa}}#controls{{display:flex;gap:9px;flex-wrap:wrap;border-bottom:1px solid #ddd}}label{{display:flex;flex-direction:column;font-size:12px}}input,select{{padding:4px}}#plot{{height:680px}}#details{{white-space:pre-wrap;background:#fafafa;border-top:1px solid #ddd;max-height:420px;overflow:auto;font-size:13px;line-height:1.4}}</style>
<script>{js}</script><body><header><b>Pares SONAR: original EN → tradução PT-BR</b><div id="count"></div></header><div id="controls">
<label>Dataset<select id="dataset"><option value="all">Todos</option></select></label><label>Label<select id="label"><option value="all">Todos</option></select></label><label>Origem<select id="source"><option value="all">Todas</option></select></label><label>Status<select id="status"><option value="all">Todos</option><option value="candidates">Atenção ou outlier</option><option value="attention">Só atenção</option><option value="outlier">Só outlier</option></select></label><label>SONAR mín.<input id="minimum" type="number" value="0" min="0" max="1" step=".01"></label><label>SONAR máx.<input id="maximum" type="number" value="1" min="0" max="1" step=".01"></label><label>Buscar dataset/id<input id="search" placeholder="train/294"></label><label>Conexões<input id="connections" type="checkbox" checked> Exibir linhas dos pares</label></div><div id="plot"></div><pre id="details">Clique em um ponto para ver o texto completo do par.</pre>
<script>const pairs={payload},threshold={threshold},sourceColors={source_colors};const $=id=>document.getElementById(id),vals=k=>[...new Set(pairs.map(p=>String(p[k])))].sort(),sources=vals('source');for(const k of ['dataset','label','source'])for(const v of vals(k))$(k).add(new Option(v,v));const esc=s=>String(s??'').replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
function visible(){{let st=$('status').value,q=$('search').value.trim().toLowerCase(),a=+$('minimum').value,b=+$('maximum').value;return pairs.filter(p=>($('dataset').value==='all'||p.dataset===$('dataset').value)&&($('label').value==='all'||String(p.label)===$('label').value)&&($('source').value==='all'||String(p.source)===$('source').value)&&(st==='all'||st==='attention'&&p.needs_attention||st==='outlier'&&p.outlier||st==='candidates'&&(p.needs_attention||p.outlier))&&p.sonar_cosine>=a&&p.sonar_cosine<=b&&(!q||`${{p.dataset}}/${{p.id}}`.toLowerCase().includes(q)))}}
function tip(p){{return `<b>${{esc(p.dataset)}}/${{esc(p.id)}}</b><br>SONAR: ${{p.sonar_cosine.toFixed(4)}} | distância: ${{p.sonar_distance.toFixed(4)}}<br>label: ${{esc(p.label)}} | origem: ${{esc(p.source)}}<br>atenção: ${{p.needs_attention?'sim':'não'}} | outlier: ${{p.outlier?'sim':'não'}}<br><br><b>EN:</b> ${{esc(p.original_snippet)}}<br><br><b>PT-BR:</b> ${{esc(p.translated_snippet)}}`}}
function showPair(index){{let p=pairs[index];if(!p)return;$('details').textContent=`${{p.dataset}}/${{p.id}}\nSONAR: ${{p.sonar_cosine.toFixed(6)}} | atenção: ${{p.needs_attention}} | outlier: ${{p.outlier}}\nBLASER: ${{p.blaser_qe??'n/a'}} | chrF: ${{p.chrf??'n/a'}} | XCOMET: ${{p.xcomet_qe??'n/a'}} | spans: ${{p.xcomet_error_count}}\n\nORIGINAL EN\n${{p.original_prompt}}\n\nTRADUÇÃO PT-BR\n${{p.translated_prompt}}`}}
function draw(){{let v=visible(),bad=v.filter(p=>p.needs_attention),ok=v.filter(p=>!p.needs_attention),lines=(a,c)=>{{let x=[],y=[];for(const p of a)x.push(p.original_x,p.translated_x,null),y.push(p.original_y,p.translated_y,null);return {{x,y,mode:'lines',type:'scattergl',line:{{color:c,width:c==='#e45756'?1.4:.5}},hoverinfo:'skip',showlegend:false}}}},points=(a,tr)=>sources.map(source=>{{let group=a.filter(p=>String(p.source)===source);if(!group.length)return null;return {{x:group.map(p=>tr?p.translated_x:p.original_x),y:group.map(p=>tr?p.translated_y:p.original_y),text:group.map(tip),customdata:group.map(p=>pairs.indexOf(p)),mode:'markers',type:'scattergl',name:source,legendgroup:`source:${{source}}`,showlegend:!tr,marker:{{color:sourceColors[source],symbol:tr?'diamond':'circle',size:6,opacity:.78}},hovertemplate:'%{{text}}<extra></extra>'}}}}).filter(Boolean),traces=[...($('connections').checked?[lines(ok,'#999'),lines(bad,'#e45756')]:[]),...points(v,false),...points(v,true)];Plotly.react('plot',traces,{{title:`{projection_label} global — cores: source; círculo: EN; losango: PT-BR`,margin:{{t:45}},xaxis:{{title:'{projection_label} 1'}},yaxis:{{title:'{projection_label} 2'}},legend:{{orientation:'h'}}}},{{responsive:true}}).then(()=>{{const plot=$('plot');plot.removeAllListeners('plotly_click');plot.on('plotly_click',e=>showPair(e.points[0].customdata))}});$('count').textContent=`${{v.length.toLocaleString('pt-BR')}} de ${{pairs.length.toLocaleString('pt-BR')}} pares exibidos`}}
for(const id of ['dataset','label','source','status','minimum','maximum','search','connections'])$(id).addEventListener('input',draw);draw();</script></body></html>''', encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=root / "analysis_outputs/sonar_translation_representations/sonar_embeddings.npy")
    parser.add_argument("--metadata", type=Path, default=root / "analysis_outputs/sonar_translation_representations/sonar_embeddings_metadata.csv")
    parser.add_argument("--metrics-dir", type=Path, default=root / "analysis_outputs/translation_validation_gemma-4-31B-merged")
    parser.add_argument("--outlier-dir", type=Path, default=root / "analysis_outputs/outliers")
    parser.add_argument("--original-dir", type=Path, default=root / "datasets/pi-guard-experiment/og")
    parser.add_argument("--translated-dir", type=Path, default=root / "datasets/pi-guard-experiment/translated_pt-br_gemma-4-31B")
    parser.add_argument("--output-dir", type=Path, default=root / "analysis_outputs/sonar_translation_pair_review")
    parser.add_argument("--attention-threshold", type=float, default=.80)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 usa todos; demais mantem os menores SONAR")
    parser.add_argument("--pca-batch-size", type=int, default=1024)
    args = parser.parse_args()
    if not 0 <= args.attention_threshold <= 1: raise ValueError("--attention-threshold deve estar entre 0 e 1")
    metadata, vectors = pd.read_csv(args.metadata), np.load(args.embeddings, mmap_mode="r")
    indexed = metadata_pairs(metadata, len(vectors)); datasets = {x["key"][0] for x in indexed}
    selected = set(args.datasets) if args.datasets else None
    if selected and selected - datasets: raise ValueError(f"Datasets ausentes no metadata: {sorted(selected-datasets)}")
    print(f"Projetando {len(vectors)} vetores via IncrementalPCA...")
    coords = project(vectors, args.pca_batch_size)
    pairs = build_pairs(metadata, vectors, coords, load_texts(args.original_dir, datasets), load_texts(args.translated_dir, datasets), load_metrics(args.metrics_dir), load_outliers(args.outlier_dir), args.attention_threshold, selected, args.max_pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(pairs, args.output_dir / "translation_review.csv"); write_png(pairs, args.output_dir / "translation_review_summary.png", args.attention_threshold); write_html(pairs, args.output_dir / "translation_review_panel.html", args.attention_threshold)
    print(f"Pares: {len(pairs)} | SONAR < {args.attention_threshold:.2f}: {sum(p['needs_attention'] for p in pairs)}")
    print(f"Resultados salvos em {{args.output_dir}}")

if __name__ == "__main__": main()
