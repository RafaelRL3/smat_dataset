"""Rótulos (gold) a partir da semântica executável das instâncias.

Condições de prompt.md ("Condições operacionais"), verificadas sobre as
amostras de cada instância:

- c0 (especificações): existe x com spec_L(L) e spec_R(R) e não (spec_L(M) e spec_R(M));
- c1 (divergência): existe x com B, L e R não todos iguais (suficiente, não define interferência);
- c2_esq / c2_dir (perda de alteração do pai): existe x com spec_L(L) e não spec_L(M) (idem R);
- c3 (efeito colateral): existe x e chave k com B[k] = L[k] = R[k] e M[k] != B[k].

A condição 4 (efeitos novos de M) não é verificada aqui.
Interferência = c0 ou c2 ou c3. c1 sozinha não é interferência.

Os rótulos de T2 e T3 não são verdade-terreno: são pesos-proxy documentados no README.
"""

from smat_dataset.templates import Instance

CONFIGS = ["ZS-T0.0", "ZS-T0.7", "1S-T0.0", "1S-T0.7"]
CONTEXTOS = ["corpo", "corpo_resumo", "corpo_resumo_atributos", "corpo_resumo_construtores"]
P_FORTE = 0.9   # massa do rótulo "certo" em noul e em choice de julgamento


def analisar(inst: Instance):
    """Devolve (flags, primeiro_disparo). primeiro_disparo é None sem interferência."""
    flags = {"c0": False, "c1": False, "c2_esq": False, "c2_dir": False, "c3": False}
    primeiro = None
    for x in inst.samples:
        b, l, r, m = (inst.run(k, x) for k in ("B", "L", "R", "M"))
        d = {
            "c0": inst.spec_L(l) and inst.spec_R(r) and not (inst.spec_L(m) and inst.spec_R(m)),
            "c1": not (b == l == r),
            "c2_esq": inst.spec_L(l) and not inst.spec_L(m),
            "c2_dir": inst.spec_R(r) and not inst.spec_R(m),
            "c3": any(
                k in b and k in l and k in r and b[k] == l[k] == r[k] and m.get(k) != b[k]
                for k in set(b) | set(m)
            ),
        }
        for nome, v in d.items():
            flags[nome] = flags[nome] or v
        if primeiro is None and (d["c0"] or d["c2_esq"] or d["c2_dir"] or d["c3"]):
            primeiro = {"x": x, "B": b, "L": l, "R": r, "M": m, **d}
    return flags, primeiro


def interferencia(flags: dict) -> bool:
    return flags["c0"] or flags["c2_esq"] or flags["c2_dir"] or flags["c3"]


def heuristica(primeiro: dict) -> str:
    if primeiro["c2_esq"] or primeiro["c2_dir"]:
        return "perda de alteração do pai"
    if primeiro["c3"]:
        return "efeito colateral inesperado"
    return "violação de especificação"


def alvo_j2(flags: dict) -> str:
    if flags["c2_esq"] and flags["c2_dir"]:
        return "ambos"
    if flags["c2_esq"]:
        return "left"
    if flags["c2_dir"]:
        return "right"
    return "nenhum"


def noul(p_true: float) -> dict:
    return {"false": round(1 - p_true, 4), "true": round(p_true, 4)}


def choice(pesos: dict) -> dict:
    total = sum(pesos.values())
    return {k: round(v / total, 4) for k, v in pesos.items()}


def suave(labels: list, alvo: str) -> dict:
    resto = (1 - P_FORTE) / (len(labels) - 1)
    return choice({l: (P_FORTE if l == alvo else resto) for l in labels})


def pesos_config(interf: bool) -> dict:
    # Proxy: o artigo aponta ZS a T0.0 como melhor no conjunto geral; sem outra verdade-terreno.
    if interf:
        return {"ZS-T0.0": 0.40, "ZS-T0.7": 0.15, "1S-T0.0": 0.25, "1S-T0.7": 0.20}
    return {c: 0.25 for c in CONFIGS}


def pesos_contexto(interf: bool) -> dict:
    # Proxy: o artigo diz que prompts com resumo rendem mais (prompt2, 5, 6, 8).
    if interf:
        return {"corpo": 0.10, "corpo_resumo": 0.40,
                "corpo_resumo_atributos": 0.30, "corpo_resumo_construtores": 0.20}
    return {"corpo": 0.30, "corpo_resumo": 0.30,
            "corpo_resumo_atributos": 0.20, "corpo_resumo_construtores": 0.20}
