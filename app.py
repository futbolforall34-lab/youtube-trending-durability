from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="YouTube Trending Durability",
    page_icon="data:image/svg+xml,<svg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 24 24%27><path fill=%27%23ff2d55%27 d=%27M23.5 6.2a3.2 3.2 0 0 0-2.3-2.3C19 3.3 12 3.3 12 3.3s-7 0-9.2.6A3.2 3.2 0 0 0 .5 6.2 33.5 33.5 0 0 0 0 12a33.5 33.5 0 0 0 .5 5.8 3.2 3.2 0 0 0 2.3 2.3c2.2.6 9.2.6 9.2.6s7 0 9.2-.6a3.2 3.2 0 0 0 2.3-2.3A33.5 33.5 0 0 0 24 12a33.5 33.5 0 0 0-.5-5.8ZM9.6 15.7V8.3L16 12l-6.4 3.7Z%27/></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
IMAGE_DIR = ROOT / "images"

YT_RED = "#ff2d55"
YT_CORAL = "#ff6b4a"
BG = "#050608"
SURFACE = "#111317"
SURFACE_ALT = "#171b22"
TEXT = "#f6f7fb"
TEXT_MUTED = "#a7afc3"


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATA_DIR / "combined_model.parquet").copy()
    clean = pd.concat(
        [
            pd.read_parquet(DATA_DIR / "mx_clean.parquet"),
            pd.read_parquet(DATA_DIR / "us_clean.parquet"),
        ],
        ignore_index=True,
    )

    for frame in (df, clean):
        frame["trending_date"] = pd.to_datetime(frame["trending_date"])
        frame["publishedAt"] = pd.to_datetime(frame["publishedAt"], utc=True, errors="coerce")

    df["publish_date"] = df["publishedAt"].dt.tz_convert(None).dt.date
    df["engagement_score"] = (
        0.45 * df["log_views"] + 0.35 * df["log_likes"] + 0.20 * df["log_comments"]
    )
    df["durability_band"] = pd.cut(
        df["days_in_trending"],
        bins=[0, 3, 7, 10, np.inf],
        labels=["Flash", "Solid", "Extended", "Elite"],
        include_lowest=True,
    )
    clean["year_month"] = clean["trending_date"].dt.to_period("M").dt.to_timestamp()
    return df, clean


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Instrument+Serif:ital@0;1&display=swap');

            :root {{
                --bg: {BG};
                --surface: {SURFACE};
                --surface-alt: {SURFACE_ALT};
                --text: {TEXT};
                --muted: {TEXT_MUTED};
                --line: rgba(255,255,255,.08);
                --red: {YT_RED};
                --coral: {YT_CORAL};
                --glow: rgba(255,45,85,.18);
                --shadow: 0 20px 70px rgba(0,0,0,.32);
                --ease: cubic-bezier(0.16, 1, 0.3, 1);
            }}

            html, body, [class*="css"]  {{
                font-family: 'Space Grotesk', sans-serif;
            }}

            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(255,45,85,.20), transparent 30%),
                    radial-gradient(circle at 85% 10%, rgba(255,107,74,.13), transparent 22%),
                    linear-gradient(180deg, #050608 0%, #090b10 45%, #06070a 100%);
                color: var(--text);
            }}

            .main .block-container {{
                padding-top: 1.4rem;
                padding-bottom: 4rem;
                max-width: 1320px;
            }}

            section[data-testid="stSidebar"] {{
                background: rgba(9, 11, 16, .92);
                border-right: 1px solid var(--line);
            }}

            .stMarkdown, .stMetric, .stSelectbox, .stMultiSelect, .stSlider, .stTabs, .stRadio {{
                color: var(--text);
            }}

            h1, h2, h3 {{
                letter-spacing: -0.04em;
                color: var(--text);
            }}

            .hero-shell {{
                position: relative;
                overflow: hidden;
                background: linear-gradient(135deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
                border: 1px solid rgba(255,255,255,.08);
                border-radius: 28px;
                padding: 2.4rem 2.4rem 2rem;
                box-shadow: var(--shadow);
                animation: fadeUp .9s var(--ease) both;
            }}

            .hero-shell::before {{
                content: '';
                position: absolute;
                inset: -30% auto auto 65%;
                width: 340px;
                height: 340px;
                border-radius: 999px;
                background: radial-gradient(circle, rgba(255,45,85,.32), transparent 70%);
                filter: blur(12px);
            }}

            .eyebrow {{
                display: inline-flex;
                align-items: center;
                gap: .55rem;
                padding: .45rem .8rem;
                border: 1px solid rgba(255,255,255,.12);
                border-radius: 999px;
                background: rgba(255,255,255,.04);
                color: #ffd7df;
                text-transform: uppercase;
                letter-spacing: .14em;
                font-size: .72rem;
            }}

            .hero-title {{
                font-size: clamp(3.2rem, 7vw, 6.5rem);
                line-height: .92;
                margin: 1rem 0 .6rem;
                max-width: 11ch;
            }}

            .hero-title em {{
                font-family: 'Instrument Serif', serif;
                font-style: italic;
                font-weight: 400;
                color: #ffe8ec;
            }}

            .hero-copy {{
                max-width: 860px;
                font-size: 1.03rem;
                line-height: 1.75;
                color: var(--muted);
            }}

            .hero-grid {{
                display: grid;
                grid-template-columns: 1.35fr .95fr;
                gap: 1rem;
                margin-top: 1.6rem;
            }}

            .glass-card {{
                background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.025));
                border: 1px solid rgba(255,255,255,.08);
                border-radius: 24px;
                padding: 1.15rem 1.2rem;
                box-shadow: inset 0 1px 0 rgba(255,255,255,.03);
            }}

            .impact-stat {{
                font-size: 2.3rem;
                font-weight: 700;
                letter-spacing: -0.05em;
                margin-bottom: .15rem;
            }}

            .muted {{ color: var(--muted); }}

            .section-title {{
                margin: 2.5rem 0 1rem;
                font-size: 2rem;
            }}

            .section-title em {{
                font-family: 'Instrument Serif', serif;
                font-style: italic;
                font-weight: 400;
                color: #ffc8d3;
            }}

            .kpi-card {{
                padding: 1.1rem 1.15rem;
                border-radius: 22px;
                background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
                border: 1px solid rgba(255,255,255,.07);
                min-height: 140px;
                box-shadow: var(--shadow);
                animation: fadeUp .85s var(--ease) both;
            }}

            .kpi-label {{
                text-transform: uppercase;
                font-size: .72rem;
                letter-spacing: .12em;
                color: #ffb9c8;
                margin-bottom: .8rem;
            }}

            .kpi-value {{
                font-size: 2.3rem;
                line-height: 1;
                letter-spacing: -0.05em;
                font-weight: 700;
            }}

            .kpi-sub {{
                margin-top: .55rem;
                color: var(--muted);
                font-size: .95rem;
                line-height: 1.55;
            }}

            .insight-ribbon {{
                margin: 1rem 0 2rem;
                padding: 1rem 1.1rem;
                border-left: 3px solid var(--red);
                background: rgba(255,255,255,.03);
                border-radius: 0 18px 18px 0;
                color: #f7dbe2;
            }}

            .stPlotlyChart, [data-testid="stImage"] {{
                border: 1px solid rgba(255,255,255,.08);
                border-radius: 22px;
                background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
                padding: .55rem;
                box-shadow: var(--shadow);
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: .6rem;
                background: rgba(255,255,255,.03);
                padding: .35rem;
                border-radius: 16px;
            }}

            .stTabs [data-baseweb="tab"] {{
                background: transparent;
                border-radius: 12px;
                color: var(--muted);
                height: 42px;
                padding: 0 1rem;
            }}

            .stTabs [aria-selected="true"] {{
                background: rgba(255,255,255,.08);
                color: var(--text);
            }}

            .stAlert {{
                background: rgba(255,255,255,.04);
                border: 1px solid rgba(255,255,255,.08);
                color: var(--text);
            }}

            .footer-note {{
                margin-top: 2rem;
                padding-top: 1rem;
                border-top: 1px solid rgba(255,255,255,.08);
                color: var(--muted);
                font-size: .92rem;
            }}

            ::selection {{
                background: rgba(255, 45, 85, .92);
                color: white;
            }}

            @keyframes fadeUp {{
                from {{ opacity: 0; transform: translateY(22px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}

            @media (max-width: 980px) {{
                .hero-grid {{ grid-template-columns: 1fr; }}
                .hero-shell {{ padding: 1.6rem 1.3rem; }}
                .hero-title {{ font-size: clamp(2.7rem, 14vw, 4.5rem); }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def theme_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=16, r=16, t=54, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Space Grotesk, sans-serif"),
        title=dict(font=dict(size=18)),
        legend=dict(bgcolor="rgba(0,0,0,0)", title=None, orientation="h", y=1.1),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.08)", zeroline=False),
    )
    return fig


def metric_card(label: str, value: str, sub: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compact_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


df, clean = load_data()
inject_css()

years = sorted(df["publish_year"].dropna().astype(int).unique().tolist())
categories = sorted(df["category_name"].dropna().unique().tolist())

with st.sidebar:
    st.markdown("## Control Room")
    regions = st.multiselect("Mercado", ["MX", "US"], default=["MX", "US"])
    year_range = st.slider(
        "Año de publicación",
        min_value=min(years),
        max_value=max(years),
        value=(min(years), max(years)),
    )
    selected_categories = st.multiselect(
        "Categorías", categories, default=categories[:6]
    )
    min_days = st.slider("Duración mínima en trending", 1, 15, 1)
    show_model_images = st.toggle("Mostrar artefactos del modelo", value=True)

if not regions:
    st.warning("Selecciona al menos un mercado para construir el dashboard.")
    st.stop()

filtered = df[
    df["region"].isin(regions)
    & df["publish_year"].between(year_range[0], year_range[1])
    & df["days_in_trending"].ge(min_days)
]

if selected_categories:
    filtered = filtered[filtered["category_name"].isin(selected_categories)]

clean_filtered = clean[clean["region"].isin(regions)].copy()

if filtered.empty:
    st.warning("Los filtros actuales no dejan datos para visualizar.")
    st.stop()

videos = len(filtered)
channels = filtered["channelTitle"].nunique()
median_days = filtered["days_in_trending"].median()
elite_share = (filtered["days_in_trending"] >= 10).mean()
avg_views = filtered["view_count"].median()
viral_share = (filtered["days_to_trending"] <= 1).mean()
top_region = filtered.groupby("region")["days_in_trending"].median().sort_values(ascending=False).index[0]

st.markdown(
    f"""
    <section class="hero-shell">
        <div class="eyebrow">YouTube Intelligence / Streamlit Experience</div>
        <div class="hero-title">How long does a video <em>own</em> trending?</div>
        <div class="hero-copy">
            Dashboard editorial construido sobre el proyecto de modelado MX vs US. Resume el pulso de 79.6k videos únicos,
            transforma las señales del primer día en narrativa visual y traduce hallazgos técnicos en una lectura ejecutiva más cercana a una landing premium que a un reporte básico.
        </div>
        <div class="hero-grid">
            <div class="glass-card">
                <div class="muted">Insight dominante</div>
                <div class="impact-stat">{top_region} retiene mejor el momentum</div>
                <div class="muted">En la selección actual, la mediana de permanencia es de <strong>{median_days:.1f} días</strong> y <strong>{viral_share:.0%}</strong> de los videos aterriza en trending en 24h o menos.</div>
            </div>
            <div class="glass-card">
                <div class="muted">Arquitectura del proyecto</div>
                <div class="impact-stat">5 notebooks / 19 visuales</div>
                <div class="muted">Feature engineering, comparación de mercados, LightGBM y SHAP integrados ahora en una experiencia interactiva con lenguaje visual inspirado en YouTube.</div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h2 class="section-title">Executive <em>signal</em></h2>', unsafe_allow_html=True)
cols = st.columns(5)
with cols[0]:
    metric_card("Videos modelados", f"{videos:,.0f}", "Observaciones únicas tras deduplicación por video y primer día de trending.")
with cols[1]:
    metric_card("Canales", f"{channels:,.0f}", "Universo activo de creadores y publishers dentro del recorte elegido.")
with cols[2]:
    metric_card("Mediana de durabilidad", f"{median_days:.1f}d", "Tiempo típico que un video resiste dentro del ranking de trending.")
with cols[3]:
    metric_card("Viralización en 24h", f"{viral_share:.0%}", "Share de videos que alcanzan trending el mismo día o al siguiente.")
with cols[4]:
    metric_card("Vista mediana día 1", compact_number(avg_views), "Escala de consumo en el primer día de señal algorítmica.")

st.markdown(
    f'<div class="insight-ribbon">La capa narrativa del dashboard se apoya en tres ideas del proyecto: México retiene más tiempo, el engagement del día 1 domina la predicción y la velocidad de entrada a trending funciona como multiplicador de durabilidad.</div>',
    unsafe_allow_html=True,
)

tab_overview, tab_comp, tab_model = st.tabs(["Pulse", "MX vs US", "Model & Explainability"])

with tab_overview:
    monthly = (
        clean_filtered.groupby(["year_month", "region"]).size().reset_index(name="records")
    )
    fig_monthly = px.area(
        monthly,
        x="year_month",
        y="records",
        color="region",
        line_group="region",
        color_discrete_map={"MX": YT_RED, "US": "#7c8cff"},
        title="Volumen de registros trending en el tiempo",
    )
    fig_monthly.update_traces(mode="lines", line=dict(width=2), opacity=0.78)
    theme_fig(fig_monthly, 390)

    cat_mix = (
        filtered.groupby(["region", "category_name"]).size().reset_index(name="videos")
    )
    cat_mix["share"] = cat_mix.groupby("region")["videos"].transform(lambda s: s / s.sum())
    cat_mix = cat_mix.sort_values(["region", "share"], ascending=[True, False]).groupby("region").head(8)
    fig_cat = px.bar(
        cat_mix,
        x="share",
        y="category_name",
        color="region",
        barmode="group",
        orientation="h",
        color_discrete_map={"MX": YT_RED, "US": "#7c8cff"},
        title="Mix de categorías líder por mercado",
        text=cat_mix["share"].map(lambda x: f"{x:.0%}"),
    )
    fig_cat.update_layout(xaxis_tickformat=".0%")
    theme_fig(fig_cat, 390)

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.plotly_chart(fig_monthly, use_container_width=True)
    with col_b:
        st.plotly_chart(fig_cat, use_container_width=True)

    scatter_sample = filtered.sample(min(4000, len(filtered)), random_state=7)
    fig_scatter = px.scatter(
        scatter_sample,
        x="log_views",
        y="days_in_trending",
        color="region",
        size="like_rate",
        hover_data=["channelTitle", "category_name", "title"],
        color_discrete_map={"MX": YT_RED, "US": "#7c8cff"},
        title="Engagement inicial vs durabilidad",
        opacity=0.68,
    )
    fig_scatter.update_traces(marker=dict(line=dict(width=0)))
    theme_fig(fig_scatter, 420)

    timing = (
        filtered.groupby(["publish_dayofweek", "publish_hour"]) ["days_in_trending"]
        .median()
        .reset_index()
    )
    heatmap = timing.pivot(index="publish_dayofweek", columns="publish_hour", values="days_in_trending")
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig_heat = go.Figure(
        data=go.Heatmap(
            z=heatmap.values,
            x=heatmap.columns,
            y=day_labels[: len(heatmap.index)],
            colorscale=[(0, "#171b22"), (0.35, "#5f2741"), (0.7, "#ff2d55"), (1, "#ffd166")],
            hovertemplate="Hour %{x}:00<br>%{y}<br>Median days %{z:.1f}<extra></extra>",
        )
    )
    fig_heat.update_layout(title="Cuándo se publica el contenido más durable")
    theme_fig(fig_heat, 420)

    col_c, col_d = st.columns([1.18, 1])
    with col_c:
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col_d:
        st.plotly_chart(fig_heat, use_container_width=True)

with tab_comp:
    region_summary = (
        filtered.groupby("region")
        .agg(
            videos=("video_id", "count"),
            median_days=("days_in_trending", "median"),
            p75=("days_in_trending", lambda s: s.quantile(0.75)),
            p95=("days_in_trending", lambda s: s.quantile(0.95)),
            viral_share=("days_to_trending", lambda s: (s <= 1).mean()),
            elite_share=("days_in_trending", lambda s: (s >= 10).mean()),
            median_views=("view_count", "median"),
        )
        .reset_index()
    )

    duel = st.columns(len(region_summary))
    for col, row in zip(duel, region_summary.itertuples(index=False), strict=False):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{row.region}</div>
                    <div class="kpi-value">{row.median_days:.1f}d</div>
                    <div class="kpi-sub">P75: {row.p75:.1f}d · P95: {row.p95:.1f}d<br>Viralización ≤24h: {row.viral_share:.0%}<br>Videos elite (10+d): {row.elite_share:.0%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    fig_box = px.violin(
        filtered,
        x="region",
        y="days_in_trending",
        color="region",
        box=True,
        points=False,
        color_discrete_map={"MX": YT_RED, "US": "#7c8cff"},
        title="Distribución de durabilidad por mercado",
    )
    theme_fig(fig_box, 400)

    leaders = (
        filtered.groupby(["region", "channelTitle"])
        .agg(videos=("video_id", "count"), median_days=("days_in_trending", "median"))
        .reset_index()
    )
    leaders = leaders[leaders["videos"] >= 12]
    leaders = leaders.sort_values(["region", "median_days", "videos"], ascending=[True, False, False]).groupby("region").head(7)
    fig_channels = px.bar(
        leaders,
        x="median_days",
        y="channelTitle",
        color="region",
        orientation="h",
        color_discrete_map={"MX": YT_RED, "US": "#7c8cff"},
        title="Canales con mejor historial de permanencia",
        hover_data=["videos"],
    )
    theme_fig(fig_channels, 400)

    col_e, col_f = st.columns(2)
    with col_e:
        st.plotly_chart(fig_box, use_container_width=True)
    with col_f:
        st.plotly_chart(fig_channels, use_container_width=True)

    cat_perf = (
        filtered.groupby(["region", "category_name"])
        .agg(median_days=("days_in_trending", "median"), videos=("video_id", "count"))
        .reset_index()
    )
    cat_perf = cat_perf[cat_perf["videos"] >= 120]
    fig_cat_perf = px.scatter(
        cat_perf,
        x="videos",
        y="median_days",
        color="region",
        size="videos",
        text="category_name",
        color_discrete_map={"MX": YT_RED, "US": "#7c8cff"},
        title="Escala de categoría vs durabilidad mediana",
    )
    fig_cat_perf.update_traces(textposition="top center")
    theme_fig(fig_cat_perf, 430)
    st.plotly_chart(fig_cat_perf, use_container_width=True)

with tab_model:
    model_cols = st.columns(3)
    with model_cols[0]:
        metric_card("Modelo final", "LightGBM", "Modelo combinado MX + US con evaluación independiente sobre ambos mercados.")
    with model_cols[1]:
        metric_card("Performance", "R² 0.39", "Señal fuerte para comportamiento humano ruidoso; suficiente para generar recomendaciones tácticas.")
    with model_cols[2]:
        metric_card("Error medio", "1.43d", "Desviación típica del pronóstico final en días de permanencia en trending.")

    feature_rank = (
        filtered[["log_likes", "log_views", "days_to_trending", "log_comments", "title_word_count"]]
        .corrwith(filtered["days_in_trending"])
        .abs()
        .sort_values(ascending=False)
        .reset_index()
    )
    feature_rank.columns = ["feature", "signal_strength"]
    fig_signal = px.bar(
        feature_rank,
        x="signal_strength",
        y="feature",
        orientation="h",
        title="Señal estadística alineada con los hallazgos SHAP",
        color="signal_strength",
        color_continuous_scale=["#52304d", YT_RED, "#ffd166"],
    )
    fig_signal.update_layout(coloraxis_showscale=False)
    theme_fig(fig_signal, 340)
    st.plotly_chart(fig_signal, use_container_width=True)

    if show_model_images:
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(str(IMAGE_DIR / "fig_16_shap_importance.png"), caption="Importancia SHAP del modelo final")
            st.image(str(IMAGE_DIR / "fig_19_model_comparison.png"), caption="Comparativa de modelos")
        with img_col2:
            st.image(str(IMAGE_DIR / "fig_17_shap_beeswarm.png"), caption="Dirección del efecto por feature")
            st.image(str(IMAGE_DIR / "fig_18_shap_dependence.png"), caption="Dependencias clave entre señales")
    else:
        st.info("Activa 'Mostrar artefactos del modelo' en la barra lateral para ver las visualizaciones exportadas del análisis original.")

st.markdown(
    """
    <div class="footer-note">
        Fuente: Kaggle / rsrishav · Cobertura 2020-08-12 a 2024-04-15 · 537k+ registros brutos convertidos en una capa ejecutiva de storytelling.
    </div>
    """,
    unsafe_allow_html=True,
)
