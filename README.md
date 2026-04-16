# YouTube Trending Durability — Predictive Modeling MX vs US

> **¿Qué señales del día 1 en trending predicen cuántos días permanecerá un video?**  
> Análisis comparativo México vs Estados Unidos • LightGBM + SHAP • 3.7 años de datos

---

## Problema de Negocio

El ranking de trending de YouTube es uno de los mecanismos de distribución más poderosos de la plataforma. Para creadores y marcas, la pregunta crítica no es solo *"¿llegará mi video a trending?"* sino **"¿cuánto tiempo permanecerá ahí?"** — ya que la durabilidad determina el alcance total y el retorno sobre la inversión en contenido.

Este proyecto construye un modelo predictivo que, usando únicamente las señales disponibles en el **primer día de aparición en trending**, estima cuántos días consecutivos permanecerá el video en el ranking. Se analiza en paralelo el comportamiento de los mercados de **México y Estados Unidos** para identificar si los factores de durabilidad son universales o culturalmente específicos.

---

## Dataset

| Atributo | Detalle |
|---|---|
| Fuente | [Kaggle — YouTube Trending Video Dataset](https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset) |
| Cobertura | México (MX) y Estados Unidos (US) |
| Rango temporal | 2020-08-12 → 2024-04-15 (3.7 años) |
| Registros brutos | MX: 268,598 · US: 268,787 (total: ~537k) |
| Registros modelo | MX: 32,442 · US: 47,142 (uno por video) |
| Licencia | CC0 Public Domain |

---

## Pipeline del Proyecto

```
NB01 — Auditoría     →  NB02 — Feature Engineering  →  NB03 — EDA Comparativa
       ↓                          ↓                              ↓
Detectar problemas         Construir target               Patrones MX vs US
       ↓                  (days_in_trending)                     ↓
NB04 — Modelado      →  NB05 — Conclusiones & Recomendaciones
   LightGBM + SHAP
```

---

## Construcción del Target: `days_in_trending`

**Decisión técnica clave:** El target no es simplemente el conteo total de días — es el **máximo streak consecutivo** de apariciones en trending, **excluyendo días sin captura de datos** (21 gaps en MX, 20 en US). Esta distinción evita inflar artificialmente la durabilidad de videos que cruzaron un gap de captura.

```python
# Para cada video, encontrar la racha consecutiva más larga
# respetando los días sin captura del dataset
days_in_trending = max_consecutive_streak(
    trending_dates,
    exclude=gap_dates
)
```

---

## Hallazgos Clave

### 1. MX Videos Stay ~60% Longer in Trending

| Métrica | MX | US |
|---|---|---|
| Mediana | **8 días** | **5 días** |
| P75 | 9d | 6d |
| P95 | 13d | 8d |
| Videos únicos | 32,442 | 47,142 |

**Explicación:** Menor densidad de creadores en MX genera menos competencia por los ~200 slots diarios de trending, permitiendo mayor durabilidad por video.

### 2. El Engagement del Día 1 es el Predictor más Fuerte

Los features con mayor importancia SHAP en ambos mercados:

| Feature | MX (SHAP) | US (SHAP) |
|---|---|---|
| `log_likes` (día 1) | 🥇 | 🥇 |
| `log_views` (día 1) | 🥈 | 🥈 |
| `days_to_trending` | 🥉 | 🥉 |
| `log_comments` | 4 | 4 |
| `title_word_count` | 5 | 5 |

### 3. Velocidad de Viralización → Mayor Durabilidad

Videos que llegan a trending el **mismo día de publicación** tienen durabilidad significativamente mayor. La correlación negativa entre `days_to_trending` y `days_in_trending` es más fuerte en US (r = -0.184) que en MX (r = -0.059).

| Región | % videos viralizados en ≤1 día |
|---|---|
| MX | 64.2% |
| US | 61.3% |

### 4. Las Categorías con Mayor Durabilidad son Distintas por Mercado

| Ranking | MX | US |
|---|---|---|
| Mayor durabilidad | Music / Entertainment | Music / Entertainment |
| Menor durabilidad | News & Politics | Gaming / News |

### 5. Diferencia Estructural en el Mix de Contenido

| Categoría | MX | US | Diferencia |
|---|---|---|---|
| Music | 22.1% | 15.6% | **MX +6.5pp** |
| Entertainment | 23.7% | 19.5% | **MX +4.2pp** |
| Gaming | 14.8% | 20.0% | **US +5.2pp** |
| People & Blogs | 12.2% | 8.6% | **MX +3.7pp** |

---

## Resultados del Modelo

| Modelo | R² (log) MX | MAE (días) MX | R² (log) US | MAE (días) US |
|---|---|---|---|---|
| Baseline (mediana) | -0.039 | 2.19d | -0.020 | 1.31d |
| Ridge Regression | N/A* | — | 0.106 | 1.22d |
| LightGBM (región) | 0.279 | 1.88d | 0.260 | 1.13d |
| **LightGBM (combinado)** | **0.436** | **1.68d** | **0.411** | **1.02d** |

> *Ridge falló para MX debido a inestabilidad numérica causada por outliers extremos en `like_rate` y `comment_rate`. Se documentó como hallazgo técnico sobre la importancia del preprocessing robusto.

**Mejor modelo final (combinado, evaluación independiente):**
- R² = 0.390 · MAE = 1.43 días · RMSE = 2.05 días

**Hallazgo sorprendente:** El modelo combinado (MX+US) supera a los modelos por región en ~56% de R². Tener más datos de entrenamiento importa más que la homogeneidad del mercado.

---

## Features Utilizadas (24 total)

| Grupo | Features |
|---|---|
| Engagement (día 1) | `log_views`, `log_likes`, `log_comments`, `like_rate`, `comment_rate` |
| Temporales | `days_to_trending`, `publish_hour`, `publish_dayofweek`, `publish_month` |
| Título | `title_length`, `title_word_count`, `title_has_caps`, `title_has_excl`, `title_has_question`, `title_has_number` |
| SEO | `tag_count`, `has_tags` |
| Descripción | `has_description`, `desc_length`, `desc_has_url`, `desc_has_hashtag` |
| Flags | `comments_disabled`, `ratings_disabled` |
| Categórica | `category_name` (encoded) |
| Región | `is_mx` (solo modelo combinado) |

---

## Decisiones Técnicas

| Decisión | Justificación |
|---|---|
| Target = log1p(days_in_trending) | Distribución fuertemente sesgada a la derecha → log transforma a distribución más normal |
| Una fila por video (primer día) | Simula el escenario de predicción real: usar solo señales disponibles el día 1 |
| Excluir `dislikes` | 99%+ de valores = 0 post Nov-2021 → discontinuidad temporal severa |
| Streak consecutiva vs conteo total | Evita inflar target con días sin captura (gap robustness) |
| Modelo combinado MX+US | Más datos > homogeneidad; flag `is_mx` permite aprender efectos de región |
| LightGBM sobre Ridge/RF | Maneja outliers mejor, captura interacciones no lineales, eficiente en tabular data |

---

## Limitaciones

- **Datos observacionales:** Solo vemos videos que YA llegaron a trending. No podemos predecir si un video ajeno al dataset habría trended.
- **No hay channel-level features:** Historial del canal (frecuencia en trending, suscriptores) sería altamente predictivo pero no está en el dataset.
- **Algoritmo es caja negra:** Los features son proxies del comportamiento del algoritmo, no inputs directos.
- **R² moderado (~0.39):** Esperado en datos de comportamiento humano con muchos factores externos no capturables.

---

## Recomendaciones Accionables

### Para Creadores en MX
1. Maximizar engagement en las primeras 24h (promoción cruzada, push notifications)
2. Publicar entre 18–23h UTC (12–17h hora México)
3. Contenido de Music y Entertainment tiene mayor durabilidad en MX
4. Títulos concisos (< 60 caracteres) asociados con mayor durabilidad

### Para Creadores en US
1. Day 1 engagement es predictor aún más fuerte en US que en MX
2. Gaming tiene alta competencia → diferenciarse por calidad de engagement (like_rate)
3. US trending es más competitivo (47k videos únicos vs 32k en MX)

### Para Marcas y Plataformas
1. Priorizar canales con engagement consistente día 1, no solo viralidad ocasional
2. En MX: ventana de 8 días promedio permite campañas de amplificación más largas
3. Integración musical genera el contenido más duradero en trending

---

## Próximos Pasos

| Horizonte | Extensión |
|---|---|
| Corto plazo | Agregar features de historial del canal (freq. trending, avg durabilidad) |
| Corto plazo | NLP semántico en títulos con sentence-transformers multilingüe |
| Mediano plazo | Validación temporal (train 2020–22 / test 2023–24) |
| Mediano plazo | Expandir a BR e IN para análisis más amplio |
| Avanzado | Clasificación binaria: ¿llegará a 7+ días en trending? |
| Avanzado | Computer vision sobre thumbnails para features visuales |

---

## Estructura del Proyecto

```
nuevo_proyecto/
├── data/
│   ├── raw/
│   │   ├── MX_youtube_trending_data.csv
│   │   ├── US_youtube_trending_data.csv
│   │   ├── MX_category_id.json
│   │   └── US_category_id.json
│   └── processed/
│       ├── mx_clean.parquet
│       ├── us_clean.parquet
│       ├── mx_model.parquet
│       ├── us_model.parquet
│       └── combined_model.parquet
├── notebooks/
│   ├── NB01_data_understanding_audit.ipynb
│   ├── NB02_feature_engineering.ipynb
│   ├── NB03_eda_comparative.ipynb
│   ├── NB04_modeling.ipynb
│   └── NB05_conclusions.ipynb
├── images/
│   └── fig_01 ... fig_19 (visualizaciones exportadas)
├── docs/
│   ├── PORTFOLIO_PROJECT.md
│   ├── diccionario_variables_youtube_trending.txt
│   └── memory/
│       ├── MEMORY.md
│       └── project_youtube_trending.md
└── README.md
```

---

*Dataset: CC0 Public Domain — Kaggle / rsrishav*  
*Análisis realizado con Python 3.13 · pandas · LightGBM · SHAP · matplotlib · seaborn*
