# P1 — La anatomía del workspace se conserva y se refina a escala 8B: cuatro costuras CKA en Gemma-4-E4B-it

Paquete autocontenido de la primera publicación de la campaña J-lens.
Companion por capas de LSGOT_v3, replicando y extendiendo el estudio *J-lens*
de Anthropic (*Verbalizable Representations Form a Global Workspace in
Language Models*, Transformer Circuits Thread, jul. 2026) en
google/gemma-4-e4b-it.

**Estado:** preprint de primera etapa (v2.1), no revisado por pares.

**Estado de la evidencia:** run confirmatorio completo (200/200 documentos
válidos, 0% inválidos) bajo regla de decisión sellada, con veredicto formal
de localización *inconcluyente* — reportado tal cual —, más un diagnóstico
estructural post-hoc, explícitamente exploratorio, que explica la
no-localización: cuatro costuras CKA jerarquizadas en L9, L15, L23 y L40,
dominadas por la salida del workspace (L23). Los experimentos pendientes
antes de la versión para revisión de pares están registrados en §7 de ambos
papers.

**Claim central (uno solo):** la fracción de profundidad del onset (~36%)
se conserva respecto a Sonnet 4.5, y el análisis de estabilidad revela que
la segmentación de tres bloques está mal especificada para este modelo — la
anatomía real tiene cuatro costuras con jerarquía medible, lo que explica
mecánicamente el veredicto inconcluyente del bootstrap sellado.


## Contenido

| Archivo | Descripción |
|---|---|
| `paper_EN.md` | Preprint completo en inglés (venue: arXiv cs.LG / serie EXIS) |
| `paper_ES.md` | Versión en español (adaptación completa, no traducción literal) |
| `data/` | Artefactos numéricos del run confirmatorio v2 + `seam_analysis.json` (ver `data/DATA_SOURCES.md`) |
| `figures/` | 6 figuras (PNG) + scripts de regeneración determinista |

## Datos

`data/` contiene los derivados pequeños (reporte confirmatorio, bootstraps
de 3 y 4 bloques, matriz CKA, curvas, y `seam_analysis.json` con las
segmentaciones óptimas k=3…6, los pesos marginales de costura y el
diagnóstico de la región L40–42). El tensor Jacobiano crudo (`J_bar_v2.pt`,
1.1 GB) se excluye por tamaño y se referencia por checksum — ver
`data/DATA_SOURCES.md` para la procedencia exacta de cada cifra citada.

## Figuras y reproducibilidad

- `figures/generate_figures.py` regenera las figuras 1–5 (heatmap CKA,
  paisaje de costo, bootstraps, diagnósticos corroborativos) desde `data/`.
- `figures/seam_analysis.py` regenera la figura 6 (curva Q(k) + jerarquía
  de costuras) y todas las cifras de §4.3: segmentaciones óptimas por
  enumeración exacta para k=3…6, pesos marginales ΔQ por costura, y el
  diagnóstico intra/cruzada de L40–42. Valida además la reconstrucción del
  costo contra los valores publicados del pipeline sellado (Q=0.1109 /
  0.1153 / 0.0993, coincidencia a redondeo).

Ninguno de los dos scripts depende del tensor excluido.

## Pendientes antes de la versión para revisión de pares

Registrados en §7 de ambos papers:

- [ ] Bootstrap sellado del modelo de cinco bloques (criterio de
      localización por costura individual; predicción registrada
      {L9, L15, L23, L40}).
- [ ] Distribución bootstrap conjunta (τ₁…τ₄) para demostrar las anatomías
      alternantes A/B (hoy solo inferidas de marginales).
- [ ] Criterio de selección de k pre-registrado (estabilidad de fronteras
      primario; Q penalizado secundario).
- [ ] Diagnóstico dedicado de la región L40–42 (costura de emisión).
- [ ] Panel cross-arquitectura (Qwen2.5-7B cerrado; DeepSeek-R1-Distill-7B
      en curso).
- [ ] Réplica en un segundo corpus.
- [ ] Revisar la cita de Gurnee, Sofroniew, Lindsey et al. (2026) contra el
      formato del venue elegido antes de someter.
