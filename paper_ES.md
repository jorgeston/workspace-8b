# La anatomía por capas del workspace se conserva y se refina a escala 8B: onset en 35.7% de profundidad y cuatro costuras CKA en Gemma-4-E4B-it

**Autores:** Jorge Castillo Sepúlveda, Marco Torres Yévenes, Juan Carlos Lanas
**Afiliaciones:** AXIS Dynamics Chile · https://axisdynamics.cl · contacto@axisdynamics.cl
**Estado:** preprint de primera etapa (v2.1) — no revisado por pares. Venue objetivo: arXiv cs.LG / serie de preprints Axisdynamics.
**Estado de la evidencia:** run confirmatorio completo (200/200 documentos válidos) bajo regla de decisión sellada, cuyo veredicto formal de localización es *inconcluyente*; más un diagnóstico estructural post-hoc, explícitamente exploratorio, que explica esa no-localización y define el pre-registro de seguimiento (§7).

---

## Resumen

El estudio *J-lens* de Anthropic (Gurnee, Sofroniew, Lindsey et al., 2026) reporta que las representaciones verbalizables de Claude Sonnet 4.5 cristalizan en un workspace global a ~37% de profundidad relativa, y deja abierta (§9.1) la pregunta de si esa fracción se generaliza a otras escalas y regímenes de entrenamiento. Replicamos el estimador J-lens completo (*current-and-future*, targets residuales pre-norma, CKA lineal centrada sobre vocabulario admitido) en google/gemma-4-e4b-it (42 capas, escala 8B, post-entrenamiento mm-RLHF) bajo un protocolo pre-registrado con regla de decisión bootstrap sellada. Tres resultados. (1) **La fracción de profundidad se conserva:** la estimación puntual del onset cae en L15 = 35.7%, a menos de dos puntos del ~37% de Sonnet 4.5, con la transición workspace→motor en L23. (2) **El veredicto sellado de localización es inconcluyente:** el intervalo de estabilidad al 95% es [L9, L16] (ancho 7 > umbral 4) y la distribución bootstrap es bimodal (59% en L15, 14% en L9). (3) **La no-localización tiene una explicación estructural** que reportamos como diagnóstico exploratorio: la red no tiene dos fronteras sino **cuatro costuras CKA jerarquizadas — L9, L15, L23 y L40 —**, y forzar una segmentación de tres bloques obliga al estimador a elegir entre anatomías alternantes, que es exactamente lo que el bootstrap registra. La jerarquía por peso marginal (ΔQ al eliminar cada costura: L23 +0.034 ≫ L15 +0.016 > L9 +0.012 > L40 +0.009) identifica la *salida* del workspace (L23), no su entrada, como la transición dominante de la red. El segmento final L40–42 no constituye un bloque coherente (CKA intra 0.558 < cruzada 0.670) y se interpreta como costura de emisión, no como banda funcional. Puesto que el costo Q decrece monótonamente con el número de bloques sin codo definido, ningún k queda establecido aquí; el pre-registro de seguimiento (§7) reemplaza la pregunta "¿cuántos bloques?" por la pregunta correcta: "¿qué costuras se localizan individualmente bajo bootstrap?", con predicción registrada {L9, L15, L23, L40}.

---

## 1. Introducción

Gurnee, Sofroniew, Lindsey et al. (Anthropic, *Verbalizable Representations Form a Global Workspace in Language Models*, Transformer Circuits Thread, julio 2026) identifican en Claude Sonnet 4.5 una banda de capas donde el contenido de los estados ocultos se vuelve ampliamente disponible para verbalización —un "workspace global" en el sentido de las ciencias cognitivas—, con onset a ~37% de profundidad relativa. Su §9.1 deja abierta la pregunta que este trabajo aborda: ¿esa fracción de profundidad es una propiedad que se conserva a otra escala y bajo otro régimen de post-entrenamiento, o es un artefacto de la arquitectura y receta de Sonnet?

Respondemos con una réplica directa a escala 8B y, en el camino, encontramos algo que la pregunta original no anticipaba: a esta escala, la anatomía por capas del modelo es más rica que la partición sensorial/workspace/motor que la metodología original presupone, y esa riqueza es medible con los mismos artefactos que produce la réplica.

Una nota metodológica que los replicadores deben conocer: nuestro primer run (protocolo v1.1) transcribió mal el estimador, computando solo el término posición-a-posición ∂z[t]/∂h[t] en lugar de sumar el cotangente sobre todas las posiciones futuras válidas t′≥t. Un lente miope al futuro mide la maquinaria local de predicción del siguiente token, no la disponibilidad global para verbalización, y subestima sistemáticamente la profundidad del onset (≈L10 en vez de L15). El detalle completo vive en el changelog del pre-registro (referencia 4); en el cuerpo de este paper solo usamos el estimador corregido, y en §4 mostramos que la discrepancia v1.1↔v2 tiene además una lectura estructural.

## 2. Método

**Modelo.** google/gemma-4-e4b-it, 42 capas transformer (λ = 1…42, sin fila de embeddings en el índice de capa), post-entrenamiento mm-RLHF.

**Estimador.** J-lens *current-and-future* (adenda ec. 1-2): el cotangente se inyecta en cada posición objetivo válida del documento, no solo en la posición bajo análisis, y se agrega en dos etapas —entre posiciones dentro del documento, luego entre documentos con igual peso—. Target residual pre-norma (adenda ec. 3-4): z es la salida del último bloque y el readout es `r_λ(h) = softcap(W_U · N(J̄_λ h) + b_U)`, con la RMSNorm aplicada dentro del readout.

**CKA y segmentación.** CKA lineal centrada sobre el vocabulario admitido (255 893 de 262 144 tokens; hash publicado en §8) vía la forma cerrada `G = W_c^T W_c`. La segmentación en bloques contiguos minimiza `c(a,b) = 2/(m−1)·Σ_{i<j} D_ij` sobre distancias CKA `D = 1 − CKA`, con `Q = Σ costos de bloque / 42`, tamaños mínimos de bloque 3 (5 para el bloque workspace en el corte sellado de 3 bloques), resuelto por enumeración exacta con desempate determinista.

**Corpus.** WikiText-103, split de test, seed 42, documentos de 256 tokens: 200 para ajustar J̄ + 100 de evaluación sin solape. 200/200 documentos válidos; ~144 s/documento.

**Incertidumbre.** Bootstrap por documento, B = 500, seed 42. El intervalo de estabilidad al 95% es el intervalo entero contiguo más corto que contiene ≥95% de las réplicas (no un IC frecuentista clásico). El onset se declara **localizado** solo si ese intervalo tiene ancho ≤4 capas; contraste con bootstrap m-out-of-n (m = 69).

**Ventanas de decisión pre-registradas** (sin cambios respecto al texto congelado): **P-LSGOT** predice onset en L19–23 (45–55% de profundidad), derivada de análisis por capas en documentos internos de la serie LSGOT, referenciados por hash en el pre-registro (referencias 2–4). Nota de procedencia: el paper LSGOT publicado (arXiv:2607.09842) analiza exclusivamente trayectorias de última capa y no contiene la predicción de L21; esa hipótesis vive únicamente en el material interno congelado, y su análisis por capas queda comprometido para el trabajo companion de esa serie; **P-EXTRAP** predice onset en L13–17 (31–40%, extrapolando el ~37% de Sonnet). Ventanas y umbral de localización se fijaron antes de computar ningún Jacobiano sobre este modelo con el estimador corregido.

## 3. Resultado confirmatorio

La estimación puntual es **τ̂₁ = L15** (15/42 = 35.7% de profundidad relativa), en el centro de la ventana P-EXTRAP y a menos de dos puntos porcentuales del ~37% de Sonnet 4.5. La transición workspace→motor es **τ̂₂ = L23**. Calidad del corte: Q = 0.1109, R₃ = 0.480, G_min = 0.1379. El onset es idéntico bajo vocabulario admitido y completo.

La regla sellada exige, además de una estimación puntual dentro de ventana, que el onset esté *localizado*. No lo está: el intervalo de estabilidad al 95% es **[L9, L16]** (ancho 7 > umbral 4; m-out-of-n da [L9, L17] consistente), y la distribución bootstrap es **bimodal** — 59% de las 500 réplicas en L15 (84% en la banda L14–17), con un modo secundario del 14% en L9. El veredicto formal del run, tal como fue pre-registrado y sin reformulación, es:

> **Inconcluyente: onset no localizado.**

![Distribución bootstrap del onset, bimodal, con el intervalo de estabilidad al 95% sellado y el veredicto](figures/fig3_bootstrap_onset_3block.png)

*Figura 1. Distribución bootstrap del onset de 3 bloques (B=500). 59% de las réplicas en L15, modo secundario de 14% en L9; el intervalo [9,16] excede el umbral de localización.*

Sobre las hipótesis en competencia: P-LSGOT queda refutada en su forma fuerte — L21 no es la *entrada* al workspace (retomamos su reinterpretación en §4.4). P-EXTRAP recibe la estimación puntual en el centro exacto de su ventana, pero no puede marcarse "confirmada" bajo la regla sellada. Este veredicto se reporta aquí una sola vez; el resto del paper analiza qué estructura lo produce.

## 4. Diagnóstico estructural (exploratorio): cuatro costuras, no dos fronteras

*Todo lo que sigue en esta sección es post-hoc y exploratorio: no formó parte del análisis confirmatorio sellado y motiva el pre-registro de seguimiento de §7, no un resultado confirmado. Cada cifra es reproducible desde los artefactos de `data/` con `figures/seam_analysis.py`.*

### 4.1 Dos mínimos casi degenerados

El paisaje de costo de tres bloques tiene dos mínimos casi degenerados: τ₁ = 15 (Q = 0.1109) y τ₁ = 9 (Q = 0.1153), separados por solo 3.9%. La segmentación forzada a tres bloques elige entre dos fronteras candidatas reales según qué documentos entran en cada remuestreo — no converge sobre ruido alrededor de una sola.

![Paisaje de costo reconstruido Q(τ1) para 3 bloques, mostrando los dos mínimos casi degenerados en L9 y L15 respecto a las ventanas pre-registradas](figures/fig2_cost_landscape.png)

*Figura 2. Costo de segmentación en función del onset candidato. La estimación puntual (L15) cae dentro de P-EXTRAP; L21, centro de P-LSGOT, queda pasados ambos candidatos.*

### 4.2 El bootstrap alterna entre dos anatomías completas

Relajar la segmentación a cuatro bloques ubica las fronteras en **L9, L15 y L23** (Q = 0.0993). Sin embargo — y este punto corrige una lectura que sostuvimos en un borrador anterior de este mismo paper — el bootstrap de esa segmentación **no es más estable** que el de tres bloques: los intervalos al 95% de τ₁, τ₂ y τ₃ tienen anchos 8, 12 y 17 respectivamente (`data/bootstrap_4blocks.json`). Pasar de tres a cuatro bloques no "resuelve" la inestabilidad.

Lo que las distribuciones marginales muestran, en cambio, es un patrón de alternancia: τ₁ concentra masa en L9 (43%) y L15 (29%); τ₂ en L23 (39%) y L15 (36%); τ₃ en L23 (61%) y L40 (39%). Ese patrón es consistente con réplicas que oscilan entre **dos anatomías completas de cuatro bloques**:

- Anatomía A: cortes en (L9, L15, L23)
- Anatomía B: cortes en (L15, L23, L40)

Con los artefactos publicados solo disponemos de marginales, por lo que la alternancia se infiere y no se demuestra; el pre-registro de seguimiento almacena la distribución conjunta precisamente para cerrar este punto (§7).

![Distribuciones bootstrap post-hoc de cuatro bloques para τ1, τ2, τ3 (exploratorio)](figures/fig4_bootstrap_4block.png)

*Figura 3. Estabilidad bootstrap de las tres fronteras de 4 bloques. Las bimodalidades de τ₂ y τ₃ son la firma de las anatomías alternantes A/B.*

### 4.3 La segmentación de cinco bloques unifica ambas anatomías

La verificación directa: la segmentación óptima de **cinco** bloques (enumeración exacta) ubica sus cuatro costuras exactamente en **L9, L15, L23 y L40** (Q = 0.0905) — la unión de las anatomías A y B. Las dos estimaciones de frontera única previamente discrepantes (≈L10 del estimador miope v1.1; L15 del estimador corregido) y las dos anatomías alternantes del bootstrap son proyecciones parciales de esta única estructura.

El peso de cada costura, medido como el aumento de Q al eliminarla del corte de cinco bloques, establece una jerarquía clara:

| Costura | ΔQ al eliminarla | Lectura |
|---|---|---|
| **L23** | **+0.0337** | salida del workspace — la transición dominante de la red |
| L15 | +0.0160 | entrada del workspace (35.7% de profundidad) |
| L9 | +0.0116 | fin de la banda sensorial temprana |
| L40 | +0.0087 | inicio de la región de emisión |

![Curva Q(k) sin codo y jerarquía de costuras por peso marginal](figures/fig6_seam_hierarchy.png)

*Figura 4. Izquierda: Q decrece monótonamente con k (0.1109 → 0.0993 → 0.0905 → 0.0833 para k=3…6) sin codo definido — el número de bloques no es identificable por costo. Derecha: jerarquía de costuras por peso marginal; L23 domina.*

Dos advertencias que el propio análisis impone. Primera: como Q decrece monótonamente en k por construcción y las mejoras relativas no muestran codo (10.5%, 8.9%, 8.0%), **ningún valor de k queda establecido por este análisis** — ni cuatro ni cinco bloques. La pregunta identificable no es "¿cuántos bloques?" sino "¿qué costuras se localizan individualmente?". Segunda: el segmento final **L40–42 no es un bloque coherente** — su CKA intra media es 0.558, *inferior* a su similitud cruzada con el bloque motor (0.670). L40 marca el punto donde las capas finales comienzan a divergir individualmente hacia el unembedding: una **costura de emisión**, un fenómeno de readout, no una quinta banda funcional homogénea.

![Mapa de calor de similitud CKA a lo largo de las 42 capas, con los cortes de 3 y 4 bloques superpuestos](figures/fig1_cka_heatmap.png)

*Figura 5. Similitud CKA capa×capa. Las cuatro costuras {L9, L15, L23, L40} son caídas genuinas de similitud visibles en la matriz, no artefactos de una elección de segmentación.*


## 5. Controles de robustez

- **Vocabulario:** onset idéntico (L15) bajo vocabulario admitido (255 893 tokens) y completo (262 144).
- **Precisión:** onset invariante bajo fp16.
- **Validez de CKA:** la forma cerrada coincide con el cómputo explícito sobre un subconjunto de 20K tokens con diferencia de 1e-5.
- **Causalidad:** el gradiente del estimador respecto a posiciones no causales (t′ < t) es exactamente 0.00e+00.
- **Diagnósticos corroborativos** (adenda §7, 500 posiciones): curtosis mediana en [0.18, 9.82]; precisión top-{1,5,10,25} del readout contra el argmax de última capa = 1.000 en todo k; dimensionalidad efectiva d.99/d en [0.007, 0.973]; autocorrelación media del log-ratio −1.713.

![Cuatro diagnósticos corroborativos por capa: dimensión efectiva, curtosis, precisión top-k del readout, autocorrelación log-ratio](figures/fig5_corroborative_diagnostics.png)

*Figura 6. Diagnósticos corroborativos por capa, con L15 y L23 marcadas. Dimensión efectiva y precisión top-k suben marcadamente cerca de L23 — consistente con fidelidad de readout próxima a la salida—; curtosis y autocorrelación muestran extremos locales cerca de L21, dentro de la banda workspace.*

## 6. Discusión

Leído en sentido estricto, el resultado confirmatorio responde la pregunta abierta del §9.1 del paper de origen: a escala 8B y bajo un régimen de post-entrenamiento distinto, el onset del workspace cae en 35.7% de profundidad — la fracción no es obviamente un artefacto de una sola familia de modelos, aunque una réplica única no la establece como constante.

Leído con el cuidado que la regla sellada impone, la contribución más informativa de este trabajo no es la coincidencia numérica sino el diagnóstico estructural: la operacionalización de tres bloques que heredamos de la metodología original está **mal especificada arquitectónicamente** para este modelo. La red tiene al menos cuatro costuras CKA jerarquizadas, y un procedimiento autorizado a reportar solo dos oscila entre proyecciones parciales de esa estructura — que es exactamente lo que el veredicto inconcluyente registró. La no-localización no fue una medición fallida; fue el pre-registro detectando un supuesto estructural falso.

El hallazgo jerárquico merece subrayarse: la costura dominante de la red (L23, ΔQ tres veces el de L40 y el doble del de L15) no es la entrada del workspace sino su salida. Si esto se sostiene cross-arquitectura, sugiere que el evento computacional más marcado en la organización por capas de estos modelos es la transición desde disponibilidad global hacia compromiso motor/de emisión — una hipótesis con eco directo en la señal LSGOT previa (§4.4) y contrastable con el panel en curso.

Evitamos deliberadamente el vocabulario de sistemas dinámicos ("atractor", "cristalización") al describir esta anatomía: son propiedades de una matriz de similitud y un objetivo de segmentación, no afirmaciones sobre la dinámica interna del modelo.

**Perspectiva.** Un run paralelo en Qwen2.5-7B-Instruct (28 capas, solo SFT) muestra el perfil opuesto: un onset único y nítido en L8 con intervalo al 95% de ancho cero (492/500 réplicas idénticas). Que la *nitidez* de las costuras —no solo su ubicación— pueda ser diagnóstica del régimen de post-entrenamiento es materia del paper complementario, una vez que DeepSeek-R1-Distill-7B complete el panel.

## 7. Experimentos pendientes antes de la versión para revisión de pares

Este preprint es la primera etapa de la serie. Antes de someter a revisión de pares, quedan registrados como pendientes:

1. **Bootstrap sellado del modelo de cinco bloques** (B=500, seed nuevo pre-registrado), con criterio de localización *por costura individual* (ancho ≤4 por frontera) en lugar de por número de bloques. Predicción registrada: las cuatro costuras {L9, L15, L23, L40} se localizan individualmente cuando el modelo de segmentación las admite todas.
2. **Distribución bootstrap conjunta** (τ₁, τ₂, τ₃, τ₄), no solo marginales — necesaria para demostrar (y no solo inferir) las anatomías alternantes A/B de §4.2.
3. **Criterio de selección de modelo pre-registrado** para k (estabilidad de fronteras como criterio primario; Q penalizado como secundario), cerrado antes de computar sobre datos nuevos.
4. **Diagnóstico dedicado de la región L40–42** (CKA intra vs cruzada por réplica) para sostener o descartar la lectura de "costura de emisión".
5. **Panel cross-arquitectura** (Qwen2.5-7B-Instruct cerrado; DeepSeek-R1-Distill-7B en curso) para la hipótesis nitidez-de-costura ↔ régimen de post-entrenamiento.
6. **Réplica de corpus** (un segundo corpus además de WikiText-103) para descartar dependencia de dominio en la posición de las costuras.

## 8. Limitaciones y reproducibilidad

- Una única arquitectura confirmada (Gemma-4-E4B-it); la conservación de la fracción de profundidad no debe generalizarse sin el panel de §7.
- Todo §4 es exploratorio y post-hoc; ninguna de las cuatro costuras está confirmada bajo regla sellada hasta completar §7.1–7.3.
- La alternancia de anatomías A/B (§4.2) se infiere de marginales; su demostración requiere la distribución conjunta (§7.2).
- Protocolo completo, adenda del estimador (ec. 1-26) y changelog de desviaciones (con hashes SHA-256) como material suplementario; ver `data/DATA_SOURCES.md` para la procedencia de cada cifra. El tensor Jacobiano crudo (1.1 GB) se referencia por checksum.
- Corpus, seeds, hash de vocabulario (`6bd2b7d9…0289e3`) y parámetros del bootstrap en §2; la ruta de análisis es determinista dados esos valores. Las cifras de §4.3 se regeneran con `figures/seam_analysis.py` a partir de `data/cka_matrix_v2.npy`.

## 9. Conclusión

A escala 8B y bajo un régimen de post-entrenamiento distinto, el onset del workspace de Gemma-4-E4B-it cae en 35.7% de profundidad relativa, conservando de cerca la fracción reportada por Anthropic para Sonnet 4.5. El veredicto sellado de localización —inconcluyente— no es un resultado nulo sino el síntoma medible de un supuesto estructural falso: la red no tiene dos fronteras sino cuatro costuras jerarquizadas {L9, L15, L23, L40}, dominadas por la salida del workspace en L23. La pregunta bien planteada para la siguiente etapa no es cuántos bloques tiene el modelo, sino qué costuras sobreviven a un criterio de localización individual pre-registrado — y ese pre-registro queda definido aquí.

---

## Referencias

1. Gurnee, W.\*, Sofroniew, N.\*, Pearce, A., Piotrowski, M., Kauvar, I., Chen, R., Soligo, A., Bogdan, P., Ong, E., Wang, R., Thompson, T. B., Abrahams, D., Kantamneni, S., Ameisen, E., Batson, J., Lindsey, J.\*† (2026). *Verbalizable Representations Form a Global Workspace in Language Models*. Transformer Circuits Thread, Anthropic. https://transformer-circuits.pub/2026/workspace/index.html (6 de julio de 2026).
2. Pre-registro interno `preregistro_A1_jlens_L21.md` (v1.1, congelado; SHA-256 `af754c73c8b92b6a2eb4d3bd0b87c091579971adc698364de04278e5dfca813f`).
3. Adenda interna `propuesta_adenda_A1_MATH_v1.0.md` (SHA-256 `038204e637d3e160547700314e0e45a9a99624af8660b953f509ef195f1e9001`).
4. Changelog interno `preregistro_A1_CHANGELOG.md` — registro completo de desviaciones con fechas y hashes.

**Disponibilidad de datos y código:** los artefactos derivados (matriz CKA, distribuciones bootstrap, curvas, reporte confirmatorio, `seam_analysis.json`) acompañan este preprint en `data/`; `figures/generate_figures.py` y `figures/seam_analysis.py` regeneran todas las figuras y cifras de §4.3 deterministicamente. Ver `data/DATA_SOURCES.md` para procedencia y el checksum del tensor Jacobiano excluido.
