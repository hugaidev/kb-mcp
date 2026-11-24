# kb-mcp

Servidor MCP en Python que expone una base de conocimiento corporativa (políticas y
procedimientos) con RAG, y un harness de evals de retrieval que mide si encuentra el pasaje
correcto.

**Problema.** Los empleados no encuentran la normativa interna: está repartida en PDFs y wikis,
y la búsqueda por palabra clave falla cuando la pregunta no usa el vocabulario del documento
("me robaron la notebook" frente a "pérdida o robo de equipos"). Con `kb-mcp`, cualquier área la
consulta en lenguaje natural desde Claude Desktop o Claude Code, y la respuesta cita
`doc_id` y sección.

> **Demo:** _pendiente — GIF de Claude Desktop respondiendo "¿Puedo ir en business a Madrid?"
> con la cita a POL-002._ (`docs/demo.gif`, a grabar.)

```
Claude (cliente MCP) ──stdio / streamable HTTP──► kb-mcp (MCP SDK) ──► pgvector
                                                                         ▲
                                         docs/*.md ──► kb-ingest (chunk + embed)
```

## Qué demuestra

| | |
|---|---|
| **MCP** | Servidor propio con las tools `search_docs` y `get_doc`, y los resources `kb://docs` (catálogo) y `kb://docs/{doc_id}`. Transportes stdio y streamable HTTP. |
| **RAG** | Ingesta de Markdown con front matter, dos estrategias de chunking, embeddings multilingües locales (fastembed/ONNX, sin API keys) y pgvector con índice HNSW. |
| **Evals** | Dataset dorado de 40 preguntas, recall@1/3/5, MRR@10, doc_hit@5, comparación pareada y gate de regresión. |
| **Docker** | `docker compose up` levanta Postgres+pgvector, re-indexa y sirve por HTTP. |
| **Claude Code** | `CLAUDE.md` con convenciones, un hook que corre los evals antes de cada `git commit` y un subagente revisor (`retrieval-reviewer`). |

## Resultados

Embedder `paraphrase-multilingual-MiniLM-L12-v2` (384 dim), 10 documentos y 40 preguntas.
Un chunk es relevante si cubre ≥ 80 % del span de evidencia dorado, de modo que la relevancia
no depende del chunker. Fuente: [`evals/reports/latest.md`](evals/reports/latest.md).

| Métrica | fixed (700) | fixed-360 | **semantic** | Δ semantic − fixed |
|---|---:|---:|---:|---:|
| recall@1 | 0.475 | 0.475 | **0.575** | +0.100 |
| recall@3 | 0.700 | 0.650 | **0.825** | +0.125 |
| recall@5 | 0.800 | 0.800 | **0.875** | +0.075 |
| MRR@10 | 0.616 | 0.603 | **0.697** | +0.081 |
| doc_hit@5 | 0.950 | 0.950 | **0.975** | +0.025 |
| chunks | 46 | 90 | 74 | |
| tamaño medio (caracteres) | 660 | 344 | 362 | |

**Lectura:**

- **El chunking semántico gana en todas las métricas**: +10 pp en recall@1 y +0.08 en MRR. El
  pasaje correcto sube de posición, y eso es lo que importa cuando el modelo solo lee los
  primeros resultados.
- **No es por el tamaño.** Los chunks semánticos son más chicos, así que agregué `fixed-360`:
  ventanas fijas del mismo tamaño medio. No mejora nada (incluso baja en recall@3). La ganancia
  viene de **dónde** se corta, no de cortar más chico.
- **Doc-level casi saturado, passage-level no.** Ambos encuentran el documento correcto en el
  95–97 % de los casos; la diferencia está en traer el **párrafo** que responde.
- **Honestidad estadística.** Con n=40, cada pregunta vale 2.5 pp. En la comparación pareada el
  semántico mejora el rank en 12 preguntas, empeora en 10 y empata en 18: la mejora neta es real
  pero no uniforme. Hace falta un dataset más grande para afirmarla con confianza.
- **Fallos que quedan** (5 en semántico): preguntas cuyo vocabulario apunta a otro documento
  ("proveedor" en q22/q28 lleva a Compras cuando la respuesta está en Conducta o en Datos). Es
  una limitación del embedder chico, no del chunking: ver [siguientes pasos](#siguientes-pasos).

## Decisiones de chunking

| Decisión | Por qué |
|---|---|
| **Fijo:** ventanas de 700 caracteres con 100 de solapamiento, cortadas en espacios | Línea base estándar. Ignora la estructura del documento. |
| **Semántico:** nunca cruza un encabezado `##` | En normativa, la sección es la unidad de sentido ("Licencias especiales"). Mezclar dos secciones en un chunk diluye su embedding. |
| Dentro de una sección, corta donde la distancia coseno entre oraciones consecutivas supera el percentil 80 del documento | El umbral es relativo a cada documento, así se adapta a textos más homogéneos o más variados. |
| Límites de 200 a 900 caracteres; los restos menores a 200 se fusionan con el chunk anterior | Evita chunks de una línea sin contexto y chunks gigantes. |
| Ítems de lista y filas de tabla son unidades atómicas | Un paso numerado de un procedimiento no se parte. |
| **Ambos** embeben `título > sección` + texto | Con el mismo contexto en los dos, el delta mide solo los límites. |
| Los chunks son spans contiguos (`start`, `end`) del documento | Permite juzgar relevancia por solapamiento con la evidencia, igual para cualquier chunker. |
| Embedder multilingüe local | El corpus está en español; local significa sin costo por consulta, sin API keys y sin que los datos salgan de la red. |

## Uso

Requiere [uv](https://docs.astral.sh/uv/). La primera ejecución descarga el modelo (~220 MB) a
`~/.cache/fastembed`.

```bash
uv sync --all-extras
uv run kb-ingest            # indexa docs/ en .kb_index/ (chunker semántico)
uv run kb-evals             # evals de los tres chunkers → evals/reports/latest.md
uv run kb-evals --gate      # falla si cae por debajo de evals/thresholds.yaml
uv run pytest               # 31 tests, offline (embedder hash)
```

### Con Docker (pgvector)

```bash
docker compose up --build
# MCP por streamable HTTP en http://127.0.0.1:8000/mcp
```

El contenedor re-indexa `docs/` al arrancar. Para cambiar una política: editala y corré
`docker compose restart kb-mcp`. El modelo queda incluido en la imagen, así que el contenedor no
necesita red.

### Desde Claude Code

El repo incluye `.mcp.json`: abrí Claude Code en la carpeta, corré `uv run kb-ingest` una vez y
el servidor `kb` aparece con sus dos tools.

### Desde Claude Desktop

En `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kb": {
      "command": "uv",
      "args": ["run", "--quiet", "--directory", "/ruta/a/kb-mcp", "kb-mcp"],
      "env": { "KB_INDEX_DIR": "/ruta/a/kb-mcp/.kb_index" }
    }
  }
}
```

### Configuración

| Variable | Default | |
|---|---|---|
| `KB_STORE` | `local` | `local` (NumPy + JSON) o `pgvector` |
| `KB_DATABASE_URL` | — | DSN de Postgres si `KB_STORE=pgvector` |
| `KB_INDEX_DIR` | `.kb_index` | Directorio del store local |
| `KB_CHUNKER` | `semantic` | `semantic`, `fixed` o `fixed-<n>` |
| `KB_EMBEDDER` | `fastembed` | `fastembed`, `fastembed:<modelo>` o `hash` (solo tests) |
| `KB_TRANSPORT` | `stdio` | `stdio` o `streamable-http` (`KB_HOST`, `KB_PORT`) |

El servidor se niega a arrancar si el índice se construyó con otro embedder, para no mezclar
espacios vectoriales.

## Tools y resources

| | Entrada | Salida |
|---|---|---|
| `search_docs` | `query`, `k` (1–20, default 5), `doc_type` opcional (`politica` \| `procedimiento`) | `results[]` con `rank`, `doc_id`, `title`, `section`, `chunk_id`, `score`, `text`, `uri` |
| `get_doc` | `doc_id` | Metadatos y cuerpo Markdown; si no existe, `error` y `known_doc_ids` |
| `kb://docs` | — | Catálogo JSON |
| `kb://docs/{doc_id}` | — | Documento en Markdown |

## Evals

- **Dataset dorado** ([`evals/golden.yaml`](evals/golden.yaml)): 40 preguntas como las haría un
  empleado ("Perdí el ticket de un taxi, ¿me lo devuelven igual?"), cada una con su documento y
  una **evidencia literal**. Un test verifica que cada evidencia exista textual en su documento
  y que la pregunta no la copie.
- **Métricas:** recall@k (¿hay un chunk relevante en el top k?), MRR@10 (1/rank del primero
  relevante) y doc_hit@5 (métrica laxa a nivel documento).
- **Reporte determinista** (sin timestamps) en Markdown y JSON, con los fallos de cada chunker,
  para que se pueda comparar con diff en un PR.
- **Gate:** `evals/thresholds.yaml` fija pisos de regresión para el chunker por defecto, apenas
  por debajo del baseline.

## Claude Code

- **`CLAUDE.md`**: comandos, layout, invariantes (chunks contiguos, prefijo compartido, stdout
  limpio en stdio) y reglas (nunca tocar el golden set ni los pisos para pasar un run).
- **Hook `PreToolUse`** ([`.claude/hooks/pre_commit_evals.py`](.claude/hooks/pre_commit_evals.py)):
  detecta `git commit` en cualquier comando Bash, corre `kb-evals --gate` (~45 s) y **bloquea el
  commit** (exit 2) si hay regresión, devolviéndole a Claude el motivo. Tiene tests propios.
- **Subagente `retrieval-reviewer`** ([`.claude/agents/retrieval-reviewer.md`](.claude/agents/retrieval-reviewer.md)):
  revisor de solo lectura que corre tests y evals, compara métricas contra `HEAD`, controla la
  integridad del golden set, la equidad de la comparación y el contrato MCP.
- **Permisos** (`.claude/settings.json`): pide confirmación para editar `golden.yaml` y
  `thresholds.yaml`, y para hacer commit o push.

## Estructura

```
docs/     corpus: 6 políticas + 4 procedimientos de "Nortia S.A." (sintéticos)
ingest/   loader, chunking (fixed / semantic), embeddings, stores (local / pgvector), CLI
server/   KnowledgeBase + servidor MCP (tools y resources)
evals/    golden.yaml, harness.py, thresholds.yaml, reports/
tests/    31 tests: chunking, store, harness, protocolo MCP en memoria, hook
.claude/  settings.json, hooks/, agents/
```

## Siguientes pasos

1. **Búsqueda híbrida (BM25 + vectores) o reranker cross-encoder.** Los 5 fallos que quedan son
   de vocabulario. Por ejemplo, "me robaron el celular de la empresa" trae primero un chunk de
   *desvinculación* (que menciona "celular corporativo") y en segundo lugar el de *pérdida o
   robo*.
2. **Barrido de hiperparámetros**: percentil de corte (70/80/90) y `max_chars`, reportando el
   intervalo de confianza por bootstrap en lugar de un único número.
3. **Embedder más grande** (`multilingual-e5-large`) para medir cuánto del techo es del modelo.
4. **Golden set de 150+ preguntas**, con preguntas sin respuesta en la base para medir
   abstención.
5. **Evals de punta a punta**: además de retrieval, si Claude responde bien y cita bien usando
   las tools.

## Limitaciones

- El corpus y la empresa son sintéticos. Los números valen para este corpus y no se extrapolan.
- Los evals usan el store local (búsqueda exacta). pgvector usa HNSW, que es aproximado; con 74
  chunks el resultado es prácticamente idéntico, pero no se midió por separado.
- `docker compose` no se probó en la máquina de desarrollo (no tiene Docker). Los transportes
  stdio y HTTP sí se probaron de punta a punta.

Licencia MIT.
