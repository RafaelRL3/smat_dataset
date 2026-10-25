"""Testes do gerador. Rodar a partir da raiz do projeto:

    python -m smat_dataset.tests.test_gerador        (ou: python -m pytest smat_dataset/tests)
"""

import json
import random

from smat_dataset.gerar_dataset_sintetico import gerar_split, linhas_de_instancia
from smat_dataset.referencia import instancias_referencia
from smat_dataset.rotulos import analisar, interferencia
from smat_dataset.templates import gerar_instancia


def _flags_ref():
    return {inst.id: analisar(inst)[0] for inst in instancias_referencia()}


def test_exemplos_do_prompt_e_pdf():
    f = _flags_ref()
    # E1: interferência por perda da alteração de Left (condições 1 e 2)
    assert f["ref-E1-cleanText"]["c2_esq"] and f["ref-E1-cleanText"]["c1"]
    # E2: só a condição 3, com especificações satisfeitas
    e2 = f["ref-E2-xyz"]
    assert e2["c3"] and not e2["c0"] and not e2["c2_esq"] and not e2["c2_dir"]
    # E3: escrita sobreposta, condição 2
    assert f["ref-E3-override"]["c2_esq"]
    # E4 e E5: não são interferência
    assert not interferencia(f["ref-E4-chaves"])
    assert not interferencia(f["ref-E5-cleanText-ordem"])
    # getAllEventTypes: Left perde a adição
    assert f["ref-getAllEventTypes"]["c2_esq"]


def test_probabilidades_somam_um():
    rng = random.Random(7)
    for i in range(300):
        linhas, _ = linhas_de_instancia(gerar_instancia(rng, i))
        for row in linhas:
            for qid, alvo in json.loads(row["gold"]).items():
                dist = alvo["probabilities"]
                total = sum(dist.values())
                assert abs(total - 1.0) < 1e-3, (row["id"], qid, total)


def test_determinismo_por_semente():
    def gerar(seed):
        return gerar_split(200, random.Random(seed), "tr-", {}, {})[0]
    assert gerar(42) == gerar(42)
    assert gerar(42) != gerar(43)


def test_treino_e_teste_sem_estado_de_triagem_repetido():
    gold, gold_jul = {}, {}
    treino, _, contr_tr = gerar_split(400, random.Random(1), "tr-", gold, gold_jul)
    teste, _, contr_te = gerar_split(200, random.Random(2), "te-", gold, gold_jul)
    triagem_tr = {r["state"] for r in treino if r["workflow"] == "smat_triagem"}
    triagem_te = {r["state"] for r in teste if r["workflow"] == "smat_triagem"}
    assert not (triagem_tr & triagem_te)
    assert contr_tr == 0 and contr_te == 0
    # julgamento pode repetir entre splits (estado pouco variado), mas sem rótulo conflitante
    assert any(r["workflow"] == "smat_julgamento" for r in treino)


def test_linhas_no_formato_do_notebook():
    rng = random.Random(3)
    for i in range(50):
        linhas, interf = linhas_de_instancia(gerar_instancia(rng, i))
        for row in linhas:
            state = json.loads(row["state"])
            questions = json.loads(row["questions"])
            gold = json.loads(row["gold"])
            assert isinstance(state, (str, dict))
            assert set(gold) <= set(questions)
            assert all(q["type"] in ("choice", "noul") for q in questions.values())
        # julgamento só existe quando há interferência detectada
        assert (len(linhas) == 2) == interf


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {nome}")
