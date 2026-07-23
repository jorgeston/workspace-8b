# Procedencia de los datos — P1

Copias locales de los artefactos numéricos del run confirmatorio v2
(google/gemma-4-e4b-it, protocolo A1-MATH v1.0), para que P1 sea
autocontenido sin depender de que `v2_A1MATH/` no se mueva. Todos son
derivados pequeños (≤ 14 KB); el tensor Jacobiano crudo se excluye por
tamaño y se referencia abajo.

## Incluidos (copiados en este directorio)

| Archivo | Origen | Contenido |
|---|---|---|
| `report_v2.md` | `v2_A1MATH/confirmatorio/report_v2.md` | Reporte confirmatorio: onset τ̂₁=L15, τ̂₂=L23, Q=0.1109, controles §10, comparación v1.1 |
| `bootstrap_v2.json` | `v2_A1MATH/confirmatorio/bootstrap_v2.json` | Bootstrap B=500 (3 bloques): intervalo de estabilidad [9,16], distribución bimodal, veredicto sellado |
| `bootstrap_4blocks.json` | `v2_A1MATH/confirmatorio/bootstrap_4blocks.json` | Bootstrap exploratorio del diagnóstico post-hoc de 4 bloques (τ₁, τ₂, τ₃) |
| `cka_matrix_v2.npy` | `v2_A1MATH/confirmatorio/cka_matrix_v2.npy` | Matriz CKA capa×capa (42×42, float), base de la segmentación y las figuras |
| `curves_v2.npz` | `v2_A1MATH/confirmatorio/curves_v2.npz` | Curvas agregadas (coste Q(τ), corroborativas §7) usadas en las figuras |

## Excluido (referenciado, no copiado)

| Archivo | Tamaño | Motivo de exclusión | Cómo verificarlo |
|---|---|---|---|
| `J_bar_v2.pt` | 1.10 GB | Tensor Jacobiano promedio crudo (J̄, todas las capas × vocabulario admitido). Es el insumo pesado equivalente a un banco de embeddings — CKA/curvas/bootstrap ya están derivados de él y son suficientes para reproducir las cifras del paper. | `sha256sum` archivado en `J_bar_v2.pt.sha256` de este mismo directorio. Ruta original: `jlens/v2_A1MATH/confirmatorio/J_bar_v2.pt` (no versionado, disponible bajo pedido). |

## Documentos de protocolo (no copiados — viven en `jlens/protocolo/` y son compartidos por P1–P7)

- `protocolo/preregistro_A1_jlens_L21.md` — pre-registro congelado v1.1 (SHA-256 en el changelog), define P-LSGOT/P-EXTRAP y las ventanas L19-23 / L13-17 citadas en el paper.
- `protocolo/propuesta_adenda_A1_MATH_v1.0.md` — adenda con el estimador *current-and-future* (ec. 1-17), criterio de segmentación y regla de decisión sellada.
- `protocolo/preregistro_A1_CHANGELOG.md` — cronología, hashes, y la regla dura: nunca comparar cifras v1.1 ↔ v2 sin leer este archivo primero.
- `sintesis/INSIGHTS_post_correccion_A1MATH.md` §1, §1-bis, §5 — síntesis narrativa de la que nace el claim de P1.

No se duplican aquí porque cambian con el resto de la campaña (P2–P7 los citan igual) y duplicarlos crearía versiones divergentes.

## Reproducibilidad

Corpus: WikiText-103 test, seed 42, 256 tokens/doc, 200 docs para ajustar J̄ +
100 docs de evaluación sin solape. 200/200 documentos válidos (0% inválidos).
Vocabulario admitido: 255 893/262 144 tokens, SHA-256
`6bd2b7d9fa90b02c0f6ba919d37c4c9515ed2bc4704b1f00c0927feb960289e3`.
