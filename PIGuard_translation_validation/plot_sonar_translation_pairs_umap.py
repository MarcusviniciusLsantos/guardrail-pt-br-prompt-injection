#!/usr/bin/env python3
"""Build an interactive SONAR translation-pair review panel using UMAP."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap.umap_ as umap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_sonar_translation_pairs import (
    build_pairs,
    load_metrics,
    load_outliers,
    load_texts,
    metadata_pairs,
    scalar,
    source_color_map,
    write_csv,
    write_html,
    write_overview_png,
    write_png,
    write_translation_delta_png,
)


def project(vectors: np.ndarray, n_neighbors: int, min_dist: float, spread: float) -> np.ndarray:
    if vectors.ndim != 2 or len(vectors) < 3:
        raise ValueError("UMAP requer uma matriz com ao menos tres vetores")
    if n_neighbors < 2:
        raise ValueError("--umap-neighbors deve ser >= 2")
    if spread <= 0:
        raise ValueError("--umap-spread deve ser > 0")
    neighbors = min(n_neighbors, len(vectors) - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=min_dist,
        spread=spread,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(np.asarray(vectors, dtype=np.float32))


def pair_centroids(vectors: np.ndarray, indexed_pairs: list[dict]) -> np.ndarray:
    """Average each EN/PT-BR pair and L2-normalize it for cosine UMAP."""
    result = np.empty((len(indexed_pairs), vectors.shape[1]), dtype=np.float32)
    for position, pair in enumerate(indexed_pairs):
        centroid = (
            np.asarray(vectors[pair["original_index"]], dtype=np.float32)
            + np.asarray(vectors[pair["translated_index"]], dtype=np.float32)
        ) / 2
        norm = np.linalg.norm(centroid)
        if not np.isfinite(norm) or norm == 0:
            dataset, item_id = pair["key"]
            raise ValueError(f"Centróide inválido para {dataset}/{item_id}")
        result[position] = centroid / norm
    return result


def expand_pair_coordinates(indexed_pairs: list[dict], pair_coords: np.ndarray, vector_count: int) -> np.ndarray:
    """Assign a pair centroid coordinate to both language rows for reuse by the panel."""
    result = np.empty((vector_count, 2), dtype=np.float32)
    for position, pair in enumerate(indexed_pairs):
        result[pair["original_index"]] = pair_coords[position]
        result[pair["translated_index"]] = pair_coords[position]
    return result


def translation_deltas(vectors: np.ndarray, indexed_pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return L2-normalized PT-BR − EN directions and a validity mask."""
    result = np.empty((len(indexed_pairs), vectors.shape[1]), dtype=np.float32)
    valid = np.zeros(len(indexed_pairs), dtype=bool)
    for position, pair in enumerate(indexed_pairs):
        delta = (
            np.asarray(vectors[pair["translated_index"]], dtype=np.float32)
            - np.asarray(vectors[pair["original_index"]], dtype=np.float32)
        )
        norm = np.linalg.norm(delta)
        if np.isfinite(norm) and norm > 1e-12:
            result[position] = delta / norm
            valid[position] = True
    return result[valid], valid


def write_centroid_label_png(pairs: list[dict], path: Path, projection_label: str) -> None:
    """Write the centroid UMAP with the binary task label as the visual channel."""
    figure, plane = plt.subplots(figsize=(14, 9))
    labels = sorted({str(pair["label"]) for pair in pairs}, key=lambda value: (not value.isdigit(), value))
    colors = {"0": "#4c78a8", "1": "#e45756"}
    fallback = plt.get_cmap("tab10")
    for index, label in enumerate(labels):
        group = [pair for pair in pairs if str(pair["label"]) == label]
        plane.scatter(
            [pair["original_x"] for pair in group], [pair["original_y"] for pair in group],
            c=colors.get(label, fallback(index % fallback.N)), s=3, alpha=.65,
            linewidths=0, rasterized=True, label=f"label {label} (n={len(group):,})",
        )
    plane.set(
        title=f"{projection_label} — centróide EN/PT-BR; cor: label",
        xlabel="UMAP 1", ylabel="UMAP 2",
    )
    plane.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small", markerscale=3)
    figure.tight_layout(rect=(0, 0, .83, 1)); figure.savefig(path, dpi=220); plt.close(figure)


def write_centroid_norm_png(pairs: list[dict], path: Path, projection_label: str) -> None:
    """Write the centroid UMAP colored by ||(v_EN + v_PT-BR) / 2||_2."""
    figure, plane = plt.subplots(figsize=(14, 9))
    norms = np.asarray([pair["centroid_norm"] for pair in pairs], dtype=float)
    low, high = np.quantile(norms, [.01, .99])
    if low == high:
        low, high = float(norms.min()), float(norms.max()) + 1e-6
    scatter = plane.scatter(
        [pair["original_x"] for pair in pairs], [pair["original_y"] for pair in pairs],
        c=norms, cmap="viridis", vmin=low, vmax=high, s=3, alpha=.7,
        linewidths=0, rasterized=True,
    )
    plane.set(
        title=f"{projection_label} — centróide EN/PT-BR; cor: norma do centróide",
        xlabel="UMAP 1", ylabel="UMAP 2",
    )
    colorbar = figure.colorbar(scatter, ax=plane)
    colorbar.set_label(r"$||(v_{EN} + v_{PT-BR}) / 2||_2$ (escala q01–q99)")
    figure.tight_layout(); figure.savefig(path, dpi=220); plt.close(figure)


def write_distance_by_source(pairs: list[dict], path: Path, summary_path: Path, attention_threshold: float) -> None:
    """Write faceted density histograms and a source-level distance summary CSV."""
    distance_threshold = 1 - attention_threshold
    sources = sorted({str(pair["source"]) for pair in pairs})
    distributions = {
        source: np.asarray([pair["sonar_distance"] for pair in pairs if str(pair["source"]) == source], dtype=float)
        for source in sources
    }
    all_distances = np.concatenate(list(distributions.values()))
    upper = max(float(np.quantile(all_distances, .995)), distance_threshold * 1.05, .01)
    bins = np.linspace(0, upper, 41)
    colors = source_color_map(sources)
    columns = 5
    rows = math.ceil(len(sources) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(22, max(3.2 * rows, 5)), sharex=True)
    summary = []
    for axis, source in zip(np.ravel(axes), sources):
        values = distributions[source]
        axis.hist(values, bins=bins, density=True, color=colors[source], alpha=.82)
        axis.axvline(distance_threshold, color="#202124", linestyle="--", linewidth=.9)
        axis.set_title(source, fontsize=9)
        axis.text(
            .98, .96, f"n={len(values):,}\\nμ={values.mean():.3f}\\nσ={values.std(ddof=0):.3f}",
            transform=axis.transAxes, ha="right", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .78},
        )
        summary.append({
            "source": source, "pair_count": len(values), "distance_mean": values.mean(),
            "distance_std_population": values.std(ddof=0), "distance_median": np.median(values),
            "distance_q90": np.quantile(values, .90), "distance_q95": np.quantile(values, .95),
            "distance_max": values.max(), "count_distance_gt_threshold": int(np.sum(values > distance_threshold)),
            "share_distance_gt_threshold": np.mean(values > distance_threshold),
            "distance_threshold": distance_threshold, "sonar_cosine_threshold": attention_threshold,
        })
    for axis in np.ravel(axes)[len(sources):]:
        axis.set_visible(False)
    figure.supxlabel("distância cosseno SONAR (1 − similaridade EN↔PT-BR)")
    figure.supylabel("densidade")
    figure.suptitle(
        f"Distância cosseno EN↔PT-BR por fonte; linha tracejada: {distance_threshold:.2f} "
        f"(similaridade SONAR < {attention_threshold:.2f})",
        y=1.01,
    )
    figure.tight_layout(); figure.savefig(path, dpi=220, bbox_inches="tight"); plt.close(figure)
    pd.DataFrame(summary).sort_values("distance_mean", ascending=False).to_csv(summary_path, index=False)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=root / "analysis_outputs/sonar_translation_representations/sonar_embeddings.npy")
    parser.add_argument("--metadata", type=Path, default=root / "analysis_outputs/sonar_translation_representations/sonar_embeddings_metadata.csv")
    parser.add_argument("--metrics-dir", type=Path, default=root / "analysis_outputs/translation_validation_gemma-4-31B-merged")
    parser.add_argument("--outlier-dir", type=Path, default=root / "analysis_outputs/outliers")
    parser.add_argument("--original-dir", type=Path, default=root / "datasets/pi-guard-experiment/og")
    parser.add_argument("--translated-dir", type=Path, default=root / "datasets/pi-guard-experiment/translated_pt-br_gemma-4-31B")
    parser.add_argument("--output-dir", type=Path, default=root / "analysis_outputs/sonar_translations_umap_pair_review")
    parser.add_argument("--individual-comparison-output-dir", type=Path, default=root / "analysis_outputs/sonar_translations_umap_individual_n100_pair_review")
    parser.add_argument("--centroid-output-dir", type=Path, default=root / "analysis_outputs/sonar_translations_umap_centroid_pair_review")
    parser.add_argument("--overview-output-dir", type=Path, default=root / "analysis_outputs/sonar_translations_umap_centroid_overviews")
    parser.add_argument("--delta-output-dir", type=Path, default=root / "analysis_outputs/sonar_translation_delta_umap")
    parser.add_argument("--attention-threshold", type=float, default=.80)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 usa todos; demais mantem os menores SONAR")
    parser.add_argument(
        "--umap-neighbors", type=int, default=50,
        help="Vizinhos locais; 50 privilegia grupos semânticos estáveis no mapa global.",
    )
    parser.add_argument(
        "--umap-min-dist", type=float, default=.30,
        help="Separação mínima; 0.30 reduz a sobreposição de pontos dentro dos grupos.",
    )
    parser.add_argument(
        "--umap-spread", type=float, default=1.5,
        help="Escala efetiva entre grupos na projeção.",
    )
    parser.add_argument("--centroid-umap-neighbors", type=int, default=75)
    parser.add_argument("--centroid-umap-min-dist", type=float, default=.35)
    parser.add_argument("--centroid-umap-spread", type=float, default=1.5)
    parser.add_argument(
        "--overview-neighbors", type=int, nargs="+", default=[50, 75, 100],
        help="Valores de n_neighbors para os PNGs gerais sem conexões.",
    )
    parser.add_argument("--individual-comparison-neighbors", type=int, default=100)
    parser.add_argument("--individual-comparison-min-dist", type=float, default=.35)
    parser.add_argument("--individual-comparison-spread", type=float, default=1.5)
    parser.add_argument("--delta-umap-neighbors", type=int, default=75)
    parser.add_argument("--delta-umap-min-dist", type=float, default=.35)
    parser.add_argument("--delta-umap-spread", type=float, default=1.5)
    args = parser.parse_args()

    if not 0 <= args.attention_threshold <= 1:
        raise ValueError("--attention-threshold deve estar entre 0 e 1")
    if args.umap_min_dist < 0:
        raise ValueError("--umap-min-dist deve ser >= 0")
    if args.centroid_umap_min_dist < 0:
        raise ValueError("--centroid-umap-min-dist deve ser >= 0")
    if args.centroid_umap_spread <= 0:
        raise ValueError("--centroid-umap-spread deve ser > 0")
    if args.individual_comparison_min_dist < 0 or args.delta_umap_min_dist < 0:
        raise ValueError("min_dist das projeções comparativas deve ser >= 0")
    if args.individual_comparison_spread <= 0 or args.delta_umap_spread <= 0:
        raise ValueError("spread das projeções comparativas deve ser > 0")
    if args.individual_comparison_neighbors < 2 or args.delta_umap_neighbors < 2:
        raise ValueError("n_neighbors das projeções comparativas deve ser >= 2")
    if any(neighbors < 2 for neighbors in args.overview_neighbors):
        raise ValueError("--overview-neighbors deve conter apenas valores >= 2")

    metadata = pd.read_csv(args.metadata)
    vectors = np.load(args.embeddings, mmap_mode="r")
    indexed = metadata_pairs(metadata, len(vectors))
    datasets = {item["key"][0] for item in indexed}
    selected = set(args.datasets) if args.datasets else None
    if selected and selected - datasets:
        raise ValueError(f"Datasets ausentes no metadata: {sorted(selected - datasets)}")

    originals = load_texts(args.original_dir, datasets)
    translations = load_texts(args.translated_dir, datasets)
    metrics = load_metrics(args.metrics_dir)
    outliers = load_outliers(args.outlier_dir)

    print(f"Projetando {len(vectors)} vetores individuais via UMAP...")
    coords = project(vectors, args.umap_neighbors, args.umap_min_dist, args.umap_spread)
    pairs = build_pairs(
        metadata,
        vectors,
        coords,
        originals, translations, metrics, outliers,
        args.attention_threshold,
        selected,
        args.max_pairs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(pairs, args.output_dir / "translation_review.csv")
    write_png(pairs, args.output_dir / "translation_review_summary.png", args.attention_threshold, "UMAP")
    write_html(pairs, args.output_dir / "translation_review_panel.html", args.attention_threshold, "UMAP")
    print(f"Pares: {len(pairs)} | SONAR < {args.attention_threshold:.2f}: {sum(pair['needs_attention'] for pair in pairs)}")
    print(f"Resultados salvos em {args.output_dir}")

    print(f"Projetando {len(vectors)} vetores individuais para comparação (n_neighbors={args.individual_comparison_neighbors})...")
    comparison_coords = project(
        vectors, args.individual_comparison_neighbors,
        args.individual_comparison_min_dist, args.individual_comparison_spread,
    )
    comparison_pairs = build_pairs(
        metadata, vectors, comparison_coords, originals, translations, metrics, outliers,
        args.attention_threshold, selected, args.max_pairs,
    )
    comparison_label = f"UMAP individual (n={args.individual_comparison_neighbors}, min_dist={args.individual_comparison_min_dist:.2f})"
    args.individual_comparison_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(comparison_pairs, args.individual_comparison_output_dir / "translation_review.csv")
    write_png(comparison_pairs, args.individual_comparison_output_dir / "translation_review_summary.png", args.attention_threshold, comparison_label)
    write_html(comparison_pairs, args.individual_comparison_output_dir / "translation_review_panel.html", args.attention_threshold, comparison_label)
    print(f"Painel individual comparativo salvo em {args.individual_comparison_output_dir}")

    centroids = pair_centroids(vectors, indexed)
    overview_neighbors = sorted(set(args.overview_neighbors))
    if args.centroid_umap_neighbors not in overview_neighbors:
        overview_neighbors.append(args.centroid_umap_neighbors)
    centroid_projections = {}
    for neighbors in overview_neighbors:
        print(f"Projetando {len(centroids)} centróides via UMAP (n_neighbors={neighbors})...")
        centroid_projections[neighbors] = project(
            centroids, neighbors, args.centroid_umap_min_dist, args.centroid_umap_spread,
        )

    centroid_coords = expand_pair_coordinates(
        indexed, centroid_projections[args.centroid_umap_neighbors], len(vectors),
    )
    # A centroid is one position by definition.  Reusing its coordinate for EN
    # and PT-BR makes every pair-connection a zero-length line, so it cannot
    # communicate translation displacement in the interactive review panel.
    # Keep centroid_coords for the one-point-per-pair overview PNGs below, but
    # use the existing individual-vector UMAP for this pair-review artifact.
    connection_pairs = build_pairs(
        metadata, vectors, comparison_coords, originals, translations, metrics, outliers,
        args.attention_threshold, selected, args.max_pairs,
    )
    args.centroid_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(connection_pairs, args.centroid_output_dir / "translation_review.csv")
    centroid_label = (
        f"UMAP individual (n={args.individual_comparison_neighbors}, "
        f"min_dist={args.individual_comparison_min_dist:.2f}) — conexões EN→PT-BR"
    )
    write_png(connection_pairs, args.centroid_output_dir / "translation_review_summary.png", args.attention_threshold, centroid_label)
    write_html(connection_pairs, args.centroid_output_dir / "translation_review_panel.html", args.attention_threshold, centroid_label)
    print(f"Painel de pares com conexões salvo em {args.centroid_output_dir}")

    args.overview_output_dir.mkdir(parents=True, exist_ok=True)
    overview_pairs_for_distance = None
    for neighbors in sorted(set(args.overview_neighbors)):
        overview_coords = expand_pair_coordinates(indexed, centroid_projections[neighbors], len(vectors))
        overview_pairs = build_pairs(
            metadata, vectors, overview_coords, originals, translations, metrics, outliers,
            args.attention_threshold, selected, 0,
        )
        write_overview_png(
            overview_pairs,
            args.overview_output_dir / f"umap_centroids_neighbors_{neighbors}.png",
            f"UMAP centróides (n_neighbors={neighbors}, min_dist={args.centroid_umap_min_dist:.2f})",
        )
        overview_label = f"UMAP centróides (n_neighbors={neighbors}, min_dist={args.centroid_umap_min_dist:.2f})"
        write_centroid_label_png(
            overview_pairs,
            args.overview_output_dir / f"umap_centroids_neighbors_{neighbors}_by_label.png",
            overview_label,
        )
        write_centroid_norm_png(
            overview_pairs,
            args.overview_output_dir / f"umap_centroids_neighbors_{neighbors}_by_centroid_norm.png",
            overview_label,
        )
        overview_pairs_for_distance = overview_pairs
    print(f"PNGs gerais sem conexões salvos em {args.overview_output_dir}")
    if overview_pairs_for_distance is None:
        raise RuntimeError("Nenhuma projeção de centróides foi gerada")
    write_distance_by_source(
        overview_pairs_for_distance,
        args.overview_output_dir / "sonar_cosine_distance_by_source.png",
        args.overview_output_dir / "sonar_cosine_distance_by_source_summary.csv",
        args.attention_threshold,
    )
    print(f"Histograma e resumo por source salvos em {args.overview_output_dir}")

    deltas, valid_deltas = translation_deltas(vectors, indexed)
    if len(deltas) < 3:
        raise ValueError("São necessários ao menos três vetores de diferença não nulos para UMAP")
    print(f"Projetando {len(deltas)} vetores de diferença PT-BR − EN via UMAP (n_neighbors={args.delta_umap_neighbors})...")
    delta_coords = project(deltas, args.delta_umap_neighbors, args.delta_umap_min_dist, args.delta_umap_spread)
    delta_points = []
    for pair_index, coord in zip(np.flatnonzero(valid_deltas), delta_coords):
        pair = indexed[pair_index]
        row = pair["metadata"]
        
        raw_source = scalar(row.get("source", "unknown"))
        if pair["key"][0] == "wildguard" and raw_source == "unknown":
            raw_source = "WildGuard"
            
        delta_points.append({
            "x": float(coord[0]), "y": float(coord[1]),
            "source": raw_source,
            "sonar_cosine": pair["cosine"],
        })
    args.delta_output_dir.mkdir(parents=True, exist_ok=True)
    write_translation_delta_png(
        delta_points, args.delta_output_dir / "umap_translation_delta_source_and_sonar.png",
        f"UMAP diferenças (n_neighbors={args.delta_umap_neighbors}, min_dist={args.delta_umap_min_dist:.2f})",
    )
    print(f"UMAP de vetores de diferença salvo em {args.delta_output_dir}")


if __name__ == "__main__":
    main()
