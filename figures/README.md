# Figuras — P1

Todas generadas por `generate_figures.py` a partir de `../data/` (sin
depender del tensor Jacobiano excluido). Regenerar con:

```bash
cd P1_anatomia_cuatro_bloques/figures
python3 generate_figures.py
```

| Archivo | Contenido | Fuente | Referenciada en |
|---|---|---|---|
| `fig1_cka_heatmap.png` | Mapa de calor CKA 42×42 con cortes de 3 y 4 bloques | `cka_matrix_v2.npy` | §4 de ambos papers |
| `fig2_cost_landscape.png` | Curva Q(τ₁) reconstruida, mínimos casi degenerados L9/L15, ventanas P-EXTRAP/P-LSGOT | `cka_matrix_v2.npy` (recomputado) | §4 de ambos papers |
| `fig3_bootstrap_onset_3block.png` | Histograma bootstrap bimodal del onset (3 bloques), intervalo de estabilidad y veredicto sellado | `bootstrap_v2.json` | §3 de ambos papers |
| `fig4_bootstrap_4block.png` | Histogramas bootstrap de τ₁, τ₂, τ₃ (4 bloques, exploratorio) | `bootstrap_4blocks.json` | §4 de ambos papers |
| `fig5_corroborative_diagnostics.png` | Dimensión efectiva, curtosis, precisión top-k, autocorrelación por capa | `curves_v2.npz` | §5 de ambos papers |

## Nota sobre `fig2_cost_landscape.png`

Q(τ₁) no está guardado en ningún artefacto — se reconstruye directamente
desde la matriz CKA con la fórmula de costo documentada en el changelog
(`c(a,b) = 2/(n−1)·ΣD_ij`, `D = 1−CKA`), normalizando por el número de
capas (42). Esta reconstrucción reproduce los valores oficiales de
`report_v2.md` con diferencias de redondeo (Q=0.1109 en L15, Q≈0.1153 en
L9, Q=0.0993 para el corte óptimo de 4 bloques con fronteras exactas en
L9/L15/L23) — la coincidencia sirve como validación de que la fórmula
recuperada es la misma que usó el pipeline sellado, aunque el script de
segmentación original no forma parte de los artefactos copiados en
`data/`.

## Paleta y estilo

Colores tomados de la paleta por defecto de la skill `dataviz` (modo
claro): azul secuencial `#2a78d6`→`#0d366b` para magnitud (heatmap,
histogramas), orden categórico fijo (azul/verde/amarillo/naranja) para
series comparadas en un mismo eje, sin ejes duales, tinta de texto
separada del color de las series.
