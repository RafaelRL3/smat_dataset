"""Cenários de referência do prompt.md (E1–E5) e do exemplo getAllEventTypes do PDF.

Ficam num arquivo à parte (referencia_eval.jsonl) e servem de conjunto de
avaliação pequeno e legível. São 6 casos: não substituem medição em código real.
"""

from smat_dataset.templates import make_contador, make_lista, make_mapa, make_texto


def instancias_referencia():
    return [
        # E1: cleanText, Left perde o normalize (interferência, condições 1 e 2)
        make_texto(["the  the  dog"], "perdida", "ref-E1-cleanText"),
        # E2: x/y/z, interferência só pela condição 3
        make_contador(t=1, a=1, b=1, id_="ref-E2-xyz",
                      samples=[(0, 0), (1, 0), (0, 1), (1, 1)]),
        # E3: escrita sobreposta em b, com valor de L igual ao da base (condição 2)
        make_mapa(conflito=True, key="b", v1=0, v2=1, samples=[0], id_="ref-E3-override"),
        # E4: chaves independentes (não é interferência)
        make_mapa(conflito=False, key=None, v1=None, v2=None, samples=[0], id_="ref-E4-chaves"),
        # E5: cleanText com ordem invertida (não é interferência)
        make_texto(["the  the  dog"], "ambas", "ref-E5-cleanText-ordem"),
        # getAllEventTypes: Left perde a adição (interferência, condição 2)
        make_lista(["EVENT_START", "EVENT_END", "EVENT_LOOP"],
                   ("EVENT_TASK_START", "EVENT_SPLIT", "EVENT_JOIN"),
                   "perdida", "ref-getAllEventTypes"),
    ]
