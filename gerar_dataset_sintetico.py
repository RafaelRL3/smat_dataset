"""Gera o dataset sintético de triagem (T1–T4) e julgamento (J1–J2) para a Laya.

Uso (a partir da raiz do projeto):
    python -m smat_dataset.gerar_dataset_sintetico --n-train 3000 --n-test 600 --seed 42

Saída em smat_dataset/data/:
- train.jsonl, test.jsonl: linhas de treino/teste;
- referencia_eval.jsonl: cenários do prompt.md e do PDF (avaliação legível);
- estatisticas.json: contagens e taxa de interferência por família.

Formato de cada linha, compatível com o notebook (state/questions/gold como
strings JSON, como no dataset do HF):
    {"id", "workflow", "state", "questions", "gold", "familia", "modo"}
"""

import argparse
import json
import os
import random
import zlib
from collections import Counter

from smat_dataset.referencia import instancias_referencia
from smat_dataset.rotulos import (
    CONFIGS, CONTEXTOS, P_FORTE, alvo_j2, analisar, choice, heuristica,
    interferencia, noul, pesos_config, pesos_contexto, suave,
)
from smat_dataset.templates import gerar_instancia

PERGUNTAS_TRIAGEM = {
    "T1": {"type": "noul",
           "instructions": "As mudanças de Left e Right podem interagir neste método?"},
    "T2": {"type": "choice",
           "instructions": "Qual configuração tende a gerar um teste que revela a interação?",
           "criteria": {
               "ZS-T0.0": "zero-shot, temperatura 0.0",
               "ZS-T0.7": "zero-shot, temperatura 0.7",
               "1S-T0.0": "one-shot, temperatura 0.0",
               "1S-T0.7": "one-shot, temperatura 0.7"}},
    "T3": {"type": "choice",
           "instructions": "Que contexto incluir no prompt do gerador de testes?",
           "criteria": {
               "corpo": "só o corpo do método",
               "corpo_resumo": "corpo + resumo de mudanças",
               "corpo_resumo_atributos": "corpo + resumo + atributos",
               "corpo_resumo_construtores": "corpo + resumo + construtores"}},
    "T4": {"type": "noul",
           "instructions": "O resumo de Left é consistente com o código desta branch?"},
}

PERGUNTAS_JULGAMENTO = {
    "J1": {"type": "noul",
           "instructions": "Dada a intenção declarada de Left e Right, esta interferência é indesejada?"},
    "J2": {"type": "choice",
           "instructions": "A divergência contradiz o objetivo de: Left, Right, ambos ou nenhum?",
           "criteria": {
               "left": "a mudança de Left",
               "right": "a mudança de Right",
               "ambos": "as duas mudanças",
               "nenhum": "nenhuma das mudanças"}},
}


def _gold(distribuicoes):
    """Formato que laya.train.target_from_gold lê: {qid: {"probabilities": {rótulo: p}}}."""
    return {qid: {"probabilities": dist} for qid, dist in distribuicoes.items()}


def linha_triagem(inst, interf):
    # Metade dos cenários recebe um resumo de Left trocado pelo de Right (T4 = inconsistente).
    corrompido = zlib.crc32(inst.id.encode()) % 2 == 1
    resumo_L = inst.summary_R if corrompido else inst.summary_L
    estado = (
        f"Método: {inst.method}\n"
        f"Corpo da base (B):\n{inst.body['B']}\n"
        f"Corpo de Left (L):\n{inst.body['L']}\n"
        f"Corpo de Right (R):\n{inst.body['R']}\n"
        f"Resumo de Left: {resumo_L}\n"
        f"Resumo de Right: {inst.summary_R}\n"
        f"Atributos: {inst.attributes}\n"
        f"Construtores: {inst.constructors}"
    )
    gold = _gold({
        "T1": noul(P_FORTE if interf else 1 - P_FORTE),
        "T2": choice(pesos_config(interf)),
        "T3": choice(pesos_contexto(interf)),
        "T4": noul(1 - P_FORTE if corrompido else P_FORTE),
    })
    return {
        "id": f"{inst.id}-T",
        "workflow": "smat_triagem",
        "state": json.dumps(estado, ensure_ascii=False),
        "questions": json.dumps(PERGUNTAS_TRIAGEM, ensure_ascii=False),
        "gold": json.dumps(gold),
        "familia": inst.family,
        "modo": inst.mode,
    }


def linha_julgamento(inst, flags, primeiro):
    estado = {
        "metodo": inst.method,
        "heuristica": heuristica(primeiro),
        "valores": {
            # None quando a chave principal não existe naquela versão (ex.: chave nova no mapa).
            "base": primeiro["B"].get(inst.main_key),
            "left": primeiro["L"].get(inst.main_key),
            "right": primeiro["R"].get(inst.main_key),
            "merge": primeiro["M"].get(inst.main_key),
        },
        "resumo_left": inst.summary_L,
        "resumo_right": inst.summary_R,
    }
    gold = _gold({
        "J1": noul(P_FORTE if not inst.intents_conflict else 1 - P_FORTE),
        "J2": suave(list(PERGUNTAS_JULGAMENTO["J2"]["criteria"]), alvo_j2(flags)),
    })
    return {
        "id": f"{inst.id}-J",
        "workflow": "smat_julgamento",
        "state": json.dumps(estado, ensure_ascii=False),
        "questions": json.dumps(PERGUNTAS_JULGAMENTO, ensure_ascii=False),
        "gold": json.dumps(gold),
        "familia": inst.family,
        "modo": inst.mode,
    }


def linhas_de_instancia(inst):
    flags, primeiro = analisar(inst)
    interf = interferencia(flags)
    linhas = [linha_triagem(inst, interf)]
    if interf:
        linhas.append(linha_julgamento(inst, flags, primeiro))
    return linhas, interf


def gerar_split(n, rng, prefixo, gold_por_estado, gold_julgamento):
    """Gera n instâncias únicas pelo estado de triagem (a identidade do cenário).

    Julgamento não é deduplicado: o estado dele (valores + heurística + resumos) é pouco
    variado e repete entre cenários distintos, então deduplicar apagaria as interferências.
    Em vez disso, repetições de julgamento são só contadas, e contradições (mesmo estado,
    rótulo diferente) são contadas separadamente. Ambas deveriam ser zero para contradições.
    """
    linhas, inst_info, contradicoes = [], [], 0
    for i in range(n):
        inst = gerar_instancia(rng, i, prefixo=prefixo)
        rows, interf = linhas_de_instancia(inst)
        tri = rows[0]
        if tri["state"] in gold_por_estado:
            contradicoes += gold_por_estado[tri["state"]] != tri["gold"]
            continue
        gold_por_estado[tri["state"]] = tri["gold"]
        for row in rows[1:]:
            contradicoes += gold_julgamento.setdefault(row["state"], row["gold"]) != row["gold"]
        linhas.extend(rows)
        inst_info.append((inst.family, inst.mode, interf))
    return linhas, inst_info, contradicoes


def escrever_jsonl(caminho, linhas):
    with open(caminho, "w", encoding="utf-8") as f:
        for row in linhas:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def estatisticas(info):
    por_familia = {}
    for fam, modo, interf in info:
        d = por_familia.setdefault(fam, {"instancias": 0, "interferencia": 0, "modos": Counter()})
        d["instancias"] += 1
        d["interferencia"] += int(interf)
        d["modos"][modo] += 1
    for d in por_familia.values():
        d["modos"] = dict(d["modos"])
        d["taxa_interferencia"] = round(d["interferencia"] / d["instancias"], 3)
    return por_familia


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    gold_por_estado, gold_julgamento = {}, {}
    train, info_tr, contr_tr = gerar_split(args.n_train, random.Random(args.seed), "tr-",
                                           gold_por_estado, gold_julgamento)
    test, info_te, contr_te = gerar_split(args.n_test, random.Random(args.seed + 1), "te-",
                                          gold_por_estado, gold_julgamento)
    estados_jul_tr = {r["state"] for r in train if r["workflow"] == "smat_julgamento"}
    estados_jul_te = {r["state"] for r in test if r["workflow"] == "smat_julgamento"}

    ref = []
    for inst in instancias_referencia():
        rows, _ = linhas_de_instancia(inst)
        ref.extend(rows)

    escrever_jsonl(os.path.join(args.out, "train.jsonl"), train)
    escrever_jsonl(os.path.join(args.out, "test.jsonl"), test)
    escrever_jsonl(os.path.join(args.out, "referencia_eval.jsonl"), ref)

    stats = {
        "seed": args.seed,
        "linhas": {"train": len(train), "test": len(test), "referencia": len(ref)},
        "estados_triagem_unicos": len(gold_por_estado),
        "contradicoes": {"train": contr_tr, "test": contr_te},
        "julgamento_estados_compartilhados_treino_teste": len(estados_jul_tr & estados_jul_te),
        "workflows_train": dict(Counter(r["workflow"] for r in train)),
        "familias_train": estatisticas(info_tr),
        "familias_test": estatisticas(info_te),
        "configs_proxy": CONFIGS,
        "contextos_proxy": CONTEXTOS,
    }
    with open(os.path.join(args.out, "estatisticas.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"train: {len(train)} linhas | test: {len(test)} | referencia: {len(ref)}")
    print(f"gravado em {args.out}")


if __name__ == "__main__":
    main()
