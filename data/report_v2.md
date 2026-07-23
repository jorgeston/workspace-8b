# Confirmatorio V2 (A1-MATH v1.0) — google/gemma-4-e4b-it

## Estimador
- Current-and-future (ec. 1-2), z pre-norma, todas las posiciones I_p, 200 docs válidos
  (inválidos 0, tasa 0.00%), 144 s/doc.

## Controles §10
- hook-mapping N(z_pre)≡h_postnorm: OK | causalidad: ver log fase fit
- CKA explícita vs fórmula (subset 20K): OK
- Invariancia FP16 del onset: OK
- Vocabulario admitido: 255893/262144, SHA-256 6bd2b7d9fa90b02c0f6ba919d37c4c9515ed2bc4704b1f00c0927feb960289e3

## ENDPOINT PRIMARIO (adenda ec. 15-17)
- **onset τ̂₁ = L15** | τ̂₂ (workspace→motor) = L23
- Q = 0.1109 | R₃ = 0.480 | G_min = 0.1379
- Robustez full-vocab: onset L15 (cortes 15,23)
- Ventanas: P-LSGOT {19..23} | P-EXTRAP {13..17} | decisión requiere el
  intervalo de estabilidad de bootstrap_v2.py (B=500) — PENDIENTE.

## Corroborativas (adenda §7, 500 posiciones)
- Curtosis (mediana): rango [0.18, 9.82]
- Top-1/5/10/25 acc (vs y* del modelo) última capa: 1.000/1.000/1.000/1.000
- Dim. efectiva d.99/d: [0.007, 0.973]
- Autocorrelación log-ratio media: -1.713

## Comparación con v1.1 (posición-a-posición)
- v1.1: onset=10 (índice de tensor con embeddings), cortes (10,24), IC [10,12].
- v2 usa etiquetas λ=1..42 sin fila de embeddings — comparar con cuidado.

## Intervalo de estabilidad (adenda §8) y decisión (§9)
- onset puntual = L15; intervalo 95% = [L9, L16] (ancho 7, NO localizado)
- m-out-of-n (m=69): [L9, L17]
- **DECISIÓN: INCONCLUYENTE: onset no localizado**
