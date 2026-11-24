# Retrieval eval report

- Embedder: `fastembed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Questions: 40 · relevance: chunk covers >= 80% of the gold evidence span

| Metric | fixed | fixed-360 | semantic | Δ semantic − fixed |
|---|---:|---:|---:|---:|
| recall@1 | 0.475 | 0.475 | 0.575 | +0.100 |
| recall@3 | 0.700 | 0.650 | 0.825 | +0.125 |
| recall@5 | 0.800 | 0.800 | 0.875 | +0.075 |
| mrr@10 | 0.616 | 0.603 | 0.697 | +0.081 |
| doc_hit@5 | 0.950 | 0.950 | 0.975 | +0.025 |
| chunks | 46 | 90 | 74 | |
| avg chunk chars | 660 | 344 | 362 | |

Paired by question (rank of first relevant chunk): semantic better in 12, worse in 10, tied in 18.

## Misses (no relevant chunk in top 5)

**fixed** (8)

- `q19` Me caso el mes que viene, ¿tengo días libres? → gold `POL-004`, first relevant rank 6, top-3 docs ['POL-004', 'POL-004', 'POL-001']
- `q20` ¿Tengo días para hacer trámites personales sin explicar para qué? → gold `POL-004`, first relevant rank 10, top-3 docs ['POL-006', 'POL-004', 'POL-004']
- `q22` Mi hermano es dueño de una empresa que nos vende insumos, ¿tengo que avisar? → gold `POL-005`, first relevant rank —, top-3 docs ['PRC-004', 'PRC-004', 'PRC-001']
- `q27` ¿Mi jefe puede ver el certificado médico que presenté? → gold `POL-006`, first relevant rank 7, top-3 docs ['POL-004', 'PRC-002', 'PRC-002']
- `q28` ¿Podemos pasarle la base de clientes a un proveedor de marketing? → gold `POL-006`, first relevant rank 7, top-3 docs ['PRC-004', 'PRC-004', 'PRC-004']
- `q29` Perdí el ticket de un taxi, ¿me lo devuelven igual? → gold `PRC-001`, first relevant rank —, top-3 docs ['POL-002', 'PRC-002', 'POL-002']
- `q34` ¿Con cuánta anticipación hay que avisar que entra alguien nuevo al equipo? → gold `PRC-002`, first relevant rank —, top-3 docs ['PRC-002', 'PRC-003', 'PRC-002']
- `q36` ¿Por cuánto tiempo puede tener cuenta un consultor externo? → gold `PRC-002`, first relevant rank 6, top-3 docs ['POL-001', 'POL-001', 'POL-006']

**fixed-360** (8)

- `q19` Me caso el mes que viene, ¿tengo días libres? → gold `POL-004`, first relevant rank 9, top-3 docs ['POL-004', 'POL-004', 'POL-004']
- `q22` Mi hermano es dueño de una empresa que nos vende insumos, ¿tengo que avisar? → gold `POL-005`, first relevant rank —, top-3 docs ['PRC-004', 'POL-005', 'PRC-004']
- `q23` ¿Cómo denuncio algo sin que sepan que fui yo? → gold `POL-005`, first relevant rank —, top-3 docs ['PRC-003', 'POL-005', 'PRC-003']
- `q24` ¿Está permitido darle algo a un empleado municipal para que agilice una habilitación? → gold `POL-005`, first relevant rank 7, top-3 docs ['PRC-001', 'POL-001', 'POL-001']
- `q28` ¿Podemos pasarle la base de clientes a un proveedor de marketing? → gold `POL-006`, first relevant rank —, top-3 docs ['PRC-004', 'PRC-004', 'PRC-004']
- `q29` Perdí el ticket de un taxi, ¿me lo devuelven igual? → gold `PRC-001`, first relevant rank —, top-3 docs ['POL-002', 'POL-002', 'PRC-001']
- `q31` ¿Quién tiene que aprobar una rendición de 3.000 dólares? → gold `PRC-001`, first relevant rank 7, top-3 docs ['PRC-001', 'PRC-001', 'PRC-004']
- `q40` ¿Puedo dividir una compra grande en varias órdenes chicas para que la apruebe mi gerente? → gold `PRC-004`, first relevant rank 7, top-3 docs ['PRC-004', 'PRC-004', 'PRC-004']

**semantic** (5)

- `q22` Mi hermano es dueño de una empresa que nos vende insumos, ¿tengo que avisar? → gold `POL-005`, first relevant rank —, top-3 docs ['PRC-004', 'PRC-001', 'PRC-004']
- `q27` ¿Mi jefe puede ver el certificado médico que presenté? → gold `POL-006`, first relevant rank —, top-3 docs ['POL-004', 'PRC-002', 'PRC-001']
- `q28` ¿Podemos pasarle la base de clientes a un proveedor de marketing? → gold `POL-006`, first relevant rank —, top-3 docs ['PRC-004', 'PRC-004', 'PRC-004']
- `q29` Perdí el ticket de un taxi, ¿me lo devuelven igual? → gold `PRC-001`, first relevant rank —, top-3 docs ['POL-002', 'PRC-002', 'PRC-001']
- `q36` ¿Por cuánto tiempo puede tener cuenta un consultor externo? → gold `PRC-002`, first relevant rank 8, top-3 docs ['POL-004', 'POL-001', 'POL-001']

