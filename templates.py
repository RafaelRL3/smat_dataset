"""Famílias sintéticas de cenários de merge para o fine-tune da Laya.

Cada instância traz:
- corpos (pseudo-Java) de B, L e R, que entram no estado da triagem;
- resumos de intenção de Left e Right (PT-BR);
- semântica executável: ``run(branch, x)`` devolve a saída observada, como
  um dict de elementos de estado, e ``spec_L``/``spec_R`` dizem se uma saída
  satisfaz a intenção de cada branch.

``M`` é o merge. O modo de cada família decide se a mudança de Left sobrevive
ao merge ("perdida") ou se o merge preserva as duas ("ambas"/"ok").

A semântica é espelhada à mão do código, então os rótulos valem o quanto esse
espelhamento vale. tests/test_gerador.py fixa os exemplos de referência do
prompt.md para pegar divergências.
"""

import re
from dataclasses import dataclass
from typing import Any, Callable

WORDS = ["dog", "cat", "the", "run", "fast", "data", "go", "ok", "list", "item"]
EVENT_POOL = [
    "EVENT_START", "EVENT_END", "EVENT_TASK_START", "EVENT_LOOP", "EVENT_SPLIT",
    "EVENT_JOIN", "EVENT_ERROR", "EVENT_SKIP", "EVENT_WAIT", "EVENT_RETRY",
]
MAP_KEYS = ["chave", "limite", "modo", "fila", "nivel"]
FAMILIAS = ["texto", "contador", "mapa", "lista", "flags"]

_DUP_RE = re.compile(r"\b(\w+)\s+\1\b")


@dataclass
class Instance:
    id: str
    family: str
    mode: str
    method: str
    body: dict            # {"B": str, "L": str, "R": str}
    summary_L: str
    summary_R: str
    intents_conflict: bool
    attributes: str
    constructors: str
    main_key: str
    samples: list
    run: Callable[[str, Any], dict]
    spec_L: Callable[[dict], bool]
    spec_R: Callable[[dict], bool]


# ---------- família texto (cleanText, E1 e E5 do prompt.md) ----------

def _remove_comments(s):
    return re.sub(r"#\S*\s*", "", s)


def _ws(s):
    return re.sub(r" {2,}", " ", s)


def _dup(s):
    prev = None
    while prev != s:
        prev, s = s, _DUP_RE.sub(r"\1", s)
    return s


def gen_texto(rng, forcar=False):
    """Entrada de texto. Com forcar=True, há palavra repetida separada por espaço duplo.

    Esse gatilho é o que faz o modo "perdida" interferir; sem ele o rótulo não
    seria determinado pelo código, que é o único sinal que a triagem vê.
    """
    n = rng.randint(2, 4)
    toks = [rng.choice(WORDS) for _ in range(n)]
    dup_pos = None
    if forcar or rng.random() < 0.6:
        dup_pos = rng.randrange(len(toks)) + 1
        toks.insert(dup_pos, toks[dup_pos - 1])
    out = toks[0]
    for j, t in enumerate(toks[1:], start=1):
        if j == dup_pos:
            sep = "  "
        else:
            sep = "  " if rng.random() < 0.5 else " "
        out += sep + t
    if rng.random() < 0.3:
        out += " #nota"
    return out


def make_texto(samples, mode, id_):
    def run(branch, x):
        if branch == "B":
            s = _remove_comments(x)
        elif branch == "L":
            s = _ws(_remove_comments(x))
        elif branch == "R":
            s = _dup(_remove_comments(x))
        elif mode == "perdida":          # M herda só o efeito de R: L se perde
            s = _dup(_remove_comments(x))
        else:                            # "ambas": M aplica as duas mudanças
            s = _ws(_dup(_remove_comments(x)))
        return {"s": s}

    # No modo "perdida", R reatribui a partir da entrada original (visível no código),
    # sem ver a normalização de L: é isso que faz L se perder no merge.
    r_body = ("return removeDuplicateWords(removeComments(original));" if mode == "perdida"
              else "return removeDuplicateWords(removeComments(s));")
    return Instance(
        id=id_, family="texto", mode=mode,
        method="TextUtils.cleanText(String)",
        body={
            "B": "return removeComments(s);",
            "L": "return removeComments(normalizeWhitespace(s));",
            "R": r_body,
        },
        summary_L="normaliza espaços duplicados",
        summary_R="remove palavras repetidas consecutivas",
        intents_conflict=False,
        attributes="nenhum", constructors="nenhum",
        main_key="s", samples=list(samples), run=run,
        spec_L=lambda o: "  " not in o["s"],
        spec_R=lambda o: not _DUP_RE.search(o["s"]),
    )


# ---------- família contador (x/y/z, E2 do prompt.md) ----------

def make_contador(t, a, b, id_, samples=None):
    def c(x, y):
        return 1 if x + y > t else 0

    def run(branch, x):
        x0, y0 = x
        if branch == "B":
            return {"x": x0, "y": y0, "z": c(x0, y0)}
        if branch == "L":
            return {"x": a, "y": y0, "z": c(a, y0)}
        if branch == "R":
            return {"x": x0, "y": b, "z": c(x0, b)}
        return {"x": a, "y": b, "z": c(a, b)}

    grid = samples or [(0, 0), (1, 0), (0, 1), (1, 1)]
    return Instance(
        id=id_, family="contador", mode=f"t{t}a{a}b{b}",
        method="Calc.compute()",
        body={
            "B": f"x = x0; y = y0; z = (x + y > {t}) ? 1 : 0;",
            "L": f"x = {a}; z = (x + y > {t}) ? 1 : 0;",
            "R": f"y = {b}; z = (x + y > {t}) ? 1 : 0;",
        },
        summary_L=f"define x = {a}",
        summary_R=f"define y = {b}",
        intents_conflict=False,
        attributes="x, y (int)", constructors="nenhum",
        main_key="z", samples=grid, run=run,
        spec_L=lambda o: o["x"] == a,
        spec_R=lambda o: o["y"] == b,
    )


# ---------- família mapa (chaves independentes ou mesma chave) ----------

def make_mapa(conflito, key, v1, v2, samples, id_):
    base_keys = ["a", "b", "c", "d"]

    def run(branch, x):
        base = {k: x for k in base_keys}
        if conflito:
            if branch == "B":
                return base
            if branch == "L":
                return {**base, key: v1}
            if branch == "R":
                return {**base, key: v2}
            return {**base, key: v2}        # M: a escrita de R sobrescreve a de L
        if branch == "B":
            return base
        if branch == "L":
            return {k: v for k, v in base.items() if k not in ("a", "b")}
        if branch == "R":
            return {**base, "e": 1}
        return {**{k: v for k, v in base.items() if k not in ("a", "b")}, "e": 1}

    if conflito:
        body_L, body_R = f'valores.put("{key}", {v1});', f'valores.put("{key}", {v2});'
        sum_L, sum_R = f"grava {key} = {v1}", f"grava {key} = {v2}"
        spec_L = (lambda o: o.get(key) == v1)
        spec_R = (lambda o: o.get(key) == v2)
        main = key
    else:
        body_L = 'valores.remove("a"); valores.remove("b");'
        body_R = 'valores.put("e", 1);'
        sum_L = "remove as chaves a e b"
        sum_R = "adiciona a chave e com valor 1"
        spec_L = (lambda o: "a" not in o and "b" not in o)
        spec_R = (lambda o: o.get("e") == 1)
        main = "e"

    return Instance(
        id=id_, family="mapa", mode="conflito" if conflito else "chaves_independentes",
        method="Config.aplicar(Map)",
        body={"B": 'valores = {"a", "b", "c", "d"};', "L": body_L, "R": body_R},
        summary_L=sum_L, summary_R=sum_R,
        intents_conflict=conflito,
        attributes="valores: Map<String,Integer>", constructors="padrão",
        main_key=main, samples=list(samples), run=run,
        spec_L=spec_L, spec_R=spec_R,
    )


# ---------- família lista (getAllEventTypes, exemplo do PDF) ----------

def make_lista(base, novos, mode, id_):
    new, new1, new2 = novos
    old = base[0]
    keep = [e for e in base if e != old]
    # "perdida": R reatribui a lista inteira a partir do literal (sobrescreve o add de L).
    # "ambas": R edita a lista in-place, preservando o add de L.
    if mode == "perdida":
        r_body = f"eventTypes = [{', '.join(keep + [new1, new2])}];"
    else:
        r_body = f"eventTypes.remove({old}); eventTypes.add({new1}); eventTypes.add({new2});"

    def run(branch, x):
        if branch == "B":
            return {"eventos": list(base)}
        if branch == "L":
            return {"eventos": base + [new]}
        if branch == "R":
            return {"eventos": keep + [new1, new2]}
        if mode == "perdida":
            return {"eventos": keep + [new1, new2]}
        return {"eventos": keep + [new1, new2, new]}

    return Instance(
        id=id_, family="lista", mode=mode,
        method="SlangImpl.getAllEventTypes()",
        body={
            "B": f"eventTypes = [{', '.join(base)}];",
            "L": f"eventTypes.add({new});",
            "R": r_body,
        },
        summary_L=f"adiciona {new} à lista de eventos",
        summary_R=f"troca {old} por {new1} e {new2}",
        intents_conflict=False,
        attributes="eventTypes: List<String>", constructors="padrão",
        main_key="eventos", samples=[0], run=run,
        spec_L=lambda o: new in o["eventos"],
        spec_R=lambda o: new1 in o["eventos"] and new2 in o["eventos"] and old not in o["eventos"],
    )


# ---------- família flags (efeito colateral ou log) ----------

def make_flags(mode, id_):
    def run(branch, x):
        if mode == "log":
            if branch in ("L", "M"):
                return {"log": ["start"]}
            return {"log": []}
        if branch == "B":
            return {"enabled": False, "timeout": 30, "debug": False}
        if branch == "L":
            return {"enabled": True, "timeout": 30, "debug": False}
        if branch == "R":
            return {"enabled": False, "timeout": 60, "debug": False}
        if mode == "efeito":            # M liga debug quando as duas mudanças se somam
            return {"enabled": True, "timeout": 60, "debug": True}
        return {"enabled": True, "timeout": 60, "debug": False}   # "ok"

    if mode == "log":
        body = {"B": "log = new ArrayList<>();", "L": 'log.add("start");',
                "R": "// renomeia variável local"}
        sum_L, sum_R = "registra evento start no log", "renomeia variável local (sem efeito)"
        spec_L, spec_R = (lambda o: "start" in o["log"]), (lambda o: True)
        attrs, main = "log: List<String>", "log"
    else:
        # "efeito": a base já deriva debug de enabled e timeout, que L e R alteram; o merge
        # combina as duas mudanças e liga debug (visível no código, não só no resultado).
        b_debug = "debug = enabled && timeout == 60;" if mode == "efeito" else "debug = false;"
        body = {"B": f"enabled = false; timeout = 30; {b_debug}",
                "L": "enabled = true;", "R": "timeout = 60;"}
        sum_L, sum_R = "habilita o modo de execução", "aumenta timeout para 60"
        spec_L, spec_R = (lambda o: o["enabled"] is True), (lambda o: o["timeout"] == 60)
        attrs, main = "enabled, timeout, debug", "debug"

    return Instance(
        id=id_, family="flags", mode=mode, method="Runner.init()",
        body=body, summary_L=sum_L, summary_R=sum_R,
        intents_conflict=False, attributes=attrs, constructors="padrão",
        main_key=main, samples=[0], run=run, spec_L=spec_L, spec_R=spec_R,
    )


# ---------- sorteio de parâmetros ----------

# Distratores: linhas que não mudam a semântica, para variar a superfície do código.
DISTRATORES = [
    'log.debug("entrando no método");',
    "int tmp = 0;",
    "// ajuste de formatação",
    'metrics.inc("merge");',
    "if (verbose) trace();",
]
NOMES_CLASSE = ["TextUtils", "StrUtils", "Normalizer", "Cleaner", "Pipeline"]


def _ruido(rng, inst):
    """Acrescenta distratores e troca o nome da classe, sem mudar a semântica."""
    for k in ("B", "L", "R"):
        extras = rng.sample(DISTRATORES, rng.randint(0, 2))
        if extras:
            inst.body[k] = " ".join(extras) + " " + inst.body[k]
    _, _, metodo = inst.method.partition(".")
    inst.method = f"{rng.choice(NOMES_CLASSE)}.{metodo}"
    return inst


def gerar_instancia(rng, idx, prefixo="", familia=None):
    """Sorteia uma instância de família aleatória, com parâmetros e ruído aleatórios."""
    return _ruido(rng, _instancia_base(rng, idx, prefixo, familia))


def _instancia_base(rng, idx, prefixo, familia):
    fam = familia or rng.choice(FAMILIAS)
    id_ = f"{prefixo}{fam}-{idx:05d}"
    if fam == "texto":
        samples = [gen_texto(rng, forcar=(i == 0)) for i in range(4)]
        return make_texto(samples, rng.choice(["perdida", "ambas"]), id_)
    if fam == "contador":
        t = rng.choice([1, 2, 3])
        return make_contador(t, rng.randint(1, 3), rng.randint(1, 3), id_)
    if fam == "mapa":
        conflito = rng.random() < 0.5
        key = rng.choice(MAP_KEYS)
        v1, v2 = rng.sample(range(0, 5), 2)
        return make_mapa(conflito, key, v1, v2, [0, 1, 2], id_)
    if fam == "lista":
        pool = rng.sample(EVENT_POOL, 6)
        return make_lista(pool[:3], tuple(pool[3:6]), rng.choice(["perdida", "ambas"]), id_)
    if fam == "flags":
        return make_flags(rng.choice(["efeito", "ok", "log"]), id_)
    raise ValueError(f"família desconhecida: {fam}")
