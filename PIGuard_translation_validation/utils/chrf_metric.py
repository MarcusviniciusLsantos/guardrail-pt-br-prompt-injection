"""
chrF — checagem lexical/estrutural rápida (sacrebleu).

IMPORTANTE: chrF foi desenhada para comparar duas frases NO MESMO idioma
(hipótese vs. referência). Como nosso caso não tem uma referência humana
em pt-br separada da própria tradução, usamos aqui o texto original (EN)
como "pseudo-referência" contra a tradução (PT-BR). Isso não mede adequação
semântica cross-lingual corretamente, mas serve como um sinal barato e
rápido de "quanto sobrou textualmente idêntico" — muito útil especificamente
para o BIPIA_code, onde blocos de código, URLs e nomes de variáveis DEVEM
permanecer intactos após a tradução (só o texto natural ao redor muda).

Instalação: pip install sacrebleu
"""

from __future__ import annotations

from sacrebleu.metrics import CHRF

_chrf = CHRF()  # chrF padrão. Para chrF++, use CHRF(word_order=2).


def chrf_score(hypothesis: str, reference: str) -> float:
    """Score chrF (0-100) entre a tradução (hypothesis) e o texto original (reference)."""
    if not hypothesis or not reference:
        return 0.0
    return _chrf.sentence_score(hypothesis, [reference]).score


def chrf_score_batch(hypotheses: list[str], references: list[str]) -> list[float]:
    """Versão em lote de chrf_score."""
    return [chrf_score(h, r) for h, r in zip(hypotheses, references)]
