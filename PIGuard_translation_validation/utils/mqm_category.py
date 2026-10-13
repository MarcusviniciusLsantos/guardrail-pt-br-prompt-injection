"""
Classificação de categoria MQM (accuracy/fluency/terminology/style/locale)
para os spans já detectados pelo XCOMET. Roda só nos itens flagrados como
outlier — não na base inteira — pra manter o custo de LLM baixo.

`llm_call` é injetado por você: qualquer client (Anthropic, OpenAI, modelo
local) que receba um prompt em string e devolva a resposta em string.
"""

from __future__ import annotations

MQM_CATEGORIES = ["accuracy", "fluency", "terminology", "style", "locale_convention"]

PROMPT_TEMPLATE = """Você é um avaliador de qualidade de tradução seguindo a taxonomia MQM.

Texto original (EN): {source}
Tradução (PT-BR): {translation}
Trecho problemático identificado na tradução: "{span_text}"

Classifique esse erro em EXATAMENTE uma das categorias abaixo e responda
só com a palavra da categoria, nada mais:
- accuracy (sentido errado, omissão, adição, trecho não traduzido)
- fluency (gramática, ortografia, pontuação, registro)
- terminology (termo técnico traduzido de forma inconsistente/errada)
- style (tom, redação estranha, mas sem erro factual)
- locale_convention (formatação, unidades, convenções locais)

Categoria:"""


def classify_span(source: str, translation: str, span_text: str, llm_call) -> str:
    prompt = PROMPT_TEMPLATE.format(source=source, translation=translation, span_text=span_text)
    raw = llm_call(prompt).strip().lower()
    return raw if raw in MQM_CATEGORIES else "uncategorized"


def classify_all_spans(source: str, translation: str, error_spans: list[dict], llm_call) -> list[str]:
    return [classify_span(source, translation, s["text"], llm_call) for s in error_spans]