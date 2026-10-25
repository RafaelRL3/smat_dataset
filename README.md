# Dataset sintético para o fine-tune da Laya (triagem e julgamento SMAT)

Gera cenários de merge sintéticos com rótulos determinísticos, no formato que
`laya.train.items_from_rows` consome (`state`, `questions`, `gold`), para o notebook
`laya/notebooks/laya_finetune_smat_colab.ipynb`.

## Como gerar

Na raiz do projeto:

```bash
./projeto_paulo/bin/python -m smat_dataset.gerar_dataset_sintetico --n-train 3000 --n-test 600 --seed 42
./projeto_paulo/bin/python -m smat_dataset.tests.test_gerador
```

Saída em `smat_dataset/data/`:

| Arquivo | Conteúdo |
|---|---|
| `train.jsonl` | Treino: linhas de triagem (T1–T4) e de julgamento (J1–J2) |
| `test.jsonl` | Teste, com estados que não aparecem no treino |
| `referencia_eval.jsonl` | Os cenários E1–E5 do prompt.md e o exemplo getAllEventTypes do PDF |
| `estatisticas.json` | Contagens por família, taxa de interferência e contradições (deve ser 0) |

Cada linha: `{"id", "workflow", "state", "questions", "gold", "familia", "modo"}`, com
`state`, `questions` e `gold` como strings JSON (o notebook faz `json.loads` em cada uma).

## Como os rótulos são calculados

Os rótulos de T1, J1 e J2 vêm da semântica executável de cada cenário, não de texto escrito
à mão. Cada família (`templates.py`) define `run(branch, x)` para B, L, R e M, e predicados
`spec_L`/`spec_R` sobre a saída. `rotulos.py` verifica as condições de `prompt.md`:

| Condição | Verificação |
|---|---|
| c0 (especificações) | existe x com spec_L(L), spec_R(R) e não (spec_L(M) e spec_R(M)) |
| c1 (divergência) | existe x com B, L e R não todos iguais (suficiente, não define interferência) |
| c2 (perda de alteração do pai) | existe x com spec_L(L) e não spec_L(M), idem para R |
| c3 (efeito colateral) | existe x e chave k com B[k] = L[k] = R[k] e M[k] ≠ B[k] |

**Interferência = c0 ou c2 ou c3.** A condição 4 não é verificada.

| Pergunta | Rótulo | Tipo de rótulo |
|---|---|---|
| T1 (noul) | interferência | determinístico (derivado da semântica) |
| T2 (choice) | pesos por configuração | **proxy** |
| T3 (choice) | pesos por contexto | **proxy** |
| T4 (noul) | resumo de Left consistente com o código | determinístico (resumo trocado em metade dos casos) |
| J1 (noul) | interferência e intenções não excludentes | determinístico |
| J2 (choice) | left / right / ambos / nenhum, pelas condições c2 | determinístico |

Probabilidades não são one-hot: massa 0,9 no rótulo "certo" (`P_FORTE`). Todas somam 1.

### Limitações que importam

- **T2 e T3 são proxy.** O único sinal é a tendência do artigo (ZS a T0.0 é o melhor no geral;
  prompts com resumo rendem mais). Não há verdade-terreno para qual configuração revela uma
  interferência real. As métricas dessas duas perguntas medem aderência ao proxy, não desempenho.
- **A semântica é espelhada à mão.** Cada família foi escrita para reproduzir o comportamento
  do código Java descrito. Os exemplos de referência em `tests/test_gerador.py` fixam esse
  espelhamento para os casos do prompt.md e do PDF.
- **O M é a versão mesclada escolhida pelo modo da família.** Por exemplo, no modo `perdida`
  de `texto`, o merge herda o efeito de R sobre a entrada original e a normalização de L se
  perde. Isso reproduz o E1 do prompt.md, mas é uma escolha de modelagem, não uma regra do SMAT.
- **Diversidade limitada.** Há 5 famílias, parâmetros aleatórios (nomes, chaves, valores,
  contadores de t/a/b), distratores semanticamente neutros e nomes de classe. Mesmo assim o
  espaço é pequeno. Treino e teste são separados pelo estado de **triagem** (nenhum cenário
  aparece nos dois). O estado de **julgamento** é pouco variado (só valores, heurística e
  resumos), então 56 estados se repetem entre treino e teste, sempre com o mesmo rótulo
  (`estatisticas.json`, campo `julgamento_estados_compartilhados_treino_teste`). Deduplicar
  o julgamento apagaria quase todas as interferências, por isso ele não é deduplicado.
- **Os cenários reais do mergedataset não entram** neste gerador. O dataset real tem 13 cenários
  locais com código Java completo e a extração de L/R/M exige parsing do layout `changed-methods`.
  Fica como próximo passo: usar esses 13 como conjunto de avaliação real.

## Relação com a proposta

- A triagem (T1–T4) usa como estado o corpo de B, L e R, os resumos, atributos e construtores.
- O julgamento (J1–J2) usa a heurística disparada, os valores de base/left/right/merge da chave
  principal e os resumos. Só existe para cenários com interferência, porque a proposta diz que a
  Laya só consome interferências já detectadas por execução.
- A proposta não está no formato de treino, mas as perguntas T1–T4 e J1–J2 são as da seção
  "Perguntas que a Laya resolveria".

## Estrutura

```
smat_dataset/
  templates.py              famílias, semântica executável, distratores
  rotulos.py                condições c0–c3, rótulos, pesos proxy
  referencia.py             E1–E5 e getAllEventTypes
  gerar_dataset_sintetico.py  CLI, linhas de triagem/julgamento, split
  tests/test_gerador.py     testes (exemplos, somas, determinismo, sem vazamento, formato)
  data/                     saída gerada
```
