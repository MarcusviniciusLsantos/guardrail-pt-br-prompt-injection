"""
SONAR + BLASER 2.0 — sucessor do LaBSE.

- SONAR: espaço de embeddings multilíngue (200 idiomas), supera LaBSE/LASER3
  nas tarefas de recuperação cross-lingual (xsim/xsim++).
- BLASER 2.0 (variante QE): métrica supervisionada de qualidade de tradução
  treinada sobre embeddings SONAR, funciona sem referência humana — só
  precisa do texto-fonte e da tradução.

Instalação: pip install sonar-space fairseq2
(os modelos são baixados automaticamente no primeiro uso e ficam em
~/.cache/fairseq2 — garanta que o cluster tenha acesso à internet na
primeira execução, ou baixe o cache antes e leve junto).
"""

from __future__ import annotations

import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model

# Códigos de idioma no padrão FLORES-200/NLLB usados pelo SONAR.
LANG_CODES = {"en": "eng_Latn", "pt": "por_Latn"}


class SonarBlaserScorer:
    def __init__(self, device: str = "cuda"):
        # Se a versão instalada do fairseq2/sonar não aceitar o kwarg
        # `device` no construtor, remova-o e controle o dispositivo via
        # CUDA_VISIBLE_DEVICES no ambiente do cluster.
        self.embedder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=torch.device(device),
        )
        self.blaser_qe = load_blaser_model("blaser_2_0_qe").eval()
        if device == "cuda":
            self.blaser_qe = self.blaser_qe.to(device)

    def _embed(self, texts: list[str], lang: str) -> torch.Tensor:
        return self.embedder.predict(texts, source_lang=LANG_CODES[lang])

    def score_batch(self, src_texts: list[str], tr_texts: list[str]) -> dict[str, list[float]]:
        """Retorna, para cada par (fonte EN, tradução PT-BR):
        - sonar_cosine: similaridade de cosseno entre embeddings (não-supervisionado,
          equivalente direto ao que você já fazia com LaBSE)
        - blaser_qe: nota supervisionada de qualidade (escala ~1-5, sem referência)
        """
        src_embs = self._embed(src_texts, "en")
        tr_embs = self._embed(tr_texts, "pt")

        cosine = torch.nn.functional.cosine_similarity(src_embs, tr_embs).tolist()

        blaser_scores = []
        with torch.inference_mode():
            for s, t in zip(src_embs, tr_embs):
                score = self.blaser_qe(src=s.unsqueeze(0), mt=t.unsqueeze(0)).item()
                blaser_scores.append(score)

        return {"sonar_cosine": cosine, "blaser_qe": blaser_scores}
