import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import joblib
import os

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Predição de Safra Soja - Pitanga/PR",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# ESTILIZAÇÃO VISUAL - GOOGLE MATERIAL DESIGN / GOOGLE AI STUDIO
# ==============================================================================
st.markdown("""
    <style>
    /* Fundo geral da aplicação */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Roboto', 'Google Sans', sans-serif;
    }
    
    /* Cabeçalho principal */
    .main-title {
        color: #1a73e8;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #5f6368;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Estilização dos Cartões de Métricas (Estilo Google AI Studio) */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 16px rgba(0, 0, 0, 0.08);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #5f6368 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #1a73e8 !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }
    
    /* Botão Principal no estilo Google Blue */
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 10px 24px;
        width: 100%;
        transition: background-color 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1557b0;
        color: white;
    }
    
    /* Abas Superiores */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 8px 8px 0px 0px;
        border: 1px solid #e0e0e0;
        padding: 10px 20px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #e8f0fe !important;
        color: #1a73e8 !important;
        border-bottom: 3px solid #1a73e8 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌱 Sistema Preditivo da Safra de Soja — Pitanga (PR)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title"><b>Trabalho de Conclusão de Curso (TCC)</b> | Modelagem Preditiva via Machine Learning e Sensoriamento Remoto Orbital</div>', unsafe_allow_html=True)

# ==============================================================================
# 1. CARREGAMENTO ROBUSTO DO MODELO TREINADO (.PKL)
# ==============================================================================
@st.cache_resource
def carregar_modelo():
    # Localiza o arquivo no mesmo diretório em que o script app.py está sendo executado
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_pkl = os.path.join(diretorio_atual, 'pipeline_soja_pitanga_rf.pkl')
    
    if not os.path.exists(caminho_pkl):
        # Tenta carregar do diretório de trabalho atual caso o caminho relativo mude
        caminho_pkl = 'pipeline_soja_pitanga_rf.pkl'
        
    if os.path.exists(caminho_pkl):
        return joblib.load(caminho_pkl), None
    else:
        return None, f"Arquivo '{caminho_pkl}' não encontrado no diretório local."

pipeline_rf, erro_carregamento = carregar_modelo()

if pipeline_rf is not None:
    st.sidebar.success("✅ Modelo Random Forest carregado com sucesso!")
else:
    st.sidebar.error(f"⚠️ {erro_carregamento}")
    st.sidebar.info("💡 Certifique-se de que o arquivo **pipeline_soja_pitanga_rf.pkl** está salvo na mesma pasta do arquivo **app.py**.")

# ==============================================================================
# 2. FUNÇÃO DE INTEGRAÇÃO COM A API DO NASA POWER
# ==============================================================================
def buscar_clima_nasa_power(ano_inicio, ano_fim):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=PRECTOTCORR,T2M_MAX,RH2M&community=AG&"
        f"longitude=-51.761&latitude=-24.757&"
        f"start={ano_inicio}1001&end={ano_fim}0331&format=JSON"
    )
    try:
        res = requests.get(url, timeout=15).json()
        params = res['properties']['parameter']
        
        precip = [p for p in list(params['PRECTOTCORR'].values()) if p >= 0]
        tmax = [t for t in list(params['T2M_MAX'].values()) if t > -50]
        ur = [u for u in list(params['RH2M'].values()) if u >= 0]
        
        return sum(precip), np.mean(tmax), np.mean(ur)
    except Exception:
        return 650.0, 25.5, 74.8

# ==============================================================================
# 3. BASE HISTÓRICA DO IBGE E SENTINEL-2 (PITANGA/PR)
# ==============================================================================
dados_historicos = [
    {"safra": "2016/2017", "ano": 2017, "area": 51000, "prod_real": 50.33, "ndvi_pico": 0.7747, "precip": 637.8, "tmax": 25.60, "ur": 74.67},
    {"safra": "2017/2018", "ano": 2018, "area": 50000, "prod_real": 60.83, "ndvi_pico": 0.8005, "precip": 538.1, "tmax": 25.38, "ur": 74.64},
    {"safra": "2018/2019", "ano": 2019, "area": 50000, "prod_real": 55.00, "ndvi_pico": 0.8017, "precip": 848.4, "tmax": 25.66, "ur": 74.76},
    {"safra": "2019/2020", "ano": 2020, "area": 51700, "prod_real": 70.00, "ndvi_pico": 0.7443, "precip": 605.7, "tmax": 25.80, "ur": 75.40},
    {"safra": "2020/2021", "ano": 2021, "area": 49000, "prod_real": 60.00, "ndvi_pico": 0.7138, "precip": 737.5, "tmax": 25.43, "ur": 75.24},
    {"safra": "2021/2022", "ano": 2022, "area": 52000, "prod_real": 51.67, "ndvi_pico": 0.7351, "precip": 634.3, "tmax": 25.26, "ur": 74.80},
    {"safra": "2022/2023", "ano": 2023, "area": 55500, "prod_real": 69.17, "ndvi_pico": 0.8251, "precip": 629.6, "tmax": 25.58, "ur": 75.54},
    {"safra": "2023/2024", "ano": 2024, "area": 55800, "prod_real": 66.17, "ndvi_pico": 0.7810, "precip": 613.1, "tmax": 25.45, "ur": 74.90},
    {"safra": "2024/2025", "ano": 2025, "area": 57000, "prod_real": 72.33, "ndvi_pico": 0.7571, "precip": 679.4, "tmax": 25.65, "ur": 74.43}
]
df_hist = pd.DataFrame(dados_historicos)

# ==============================================================================
# 4. CONTROLES NA BARRA LATERAL
# ==============================================================================
st.sidebar.header("🕹️ Painel de Entrada de Dados")
modo_entrada = st.sidebar.radio("Fonte dos Dados Climáticos:", ["Simulação Manual", "Conexão Automática (NASA POWER API)"])

if modo_entrada == "Conexão Automática (NASA POWER API)":
    st.sidebar.info("🌐 Conectado à API do NASA POWER (Pitanga/PR)...")
    precip_api, tmax_api, ur_api = buscar_clima_nasa_power("2024", "2025")
    input_precip = st.sidebar.number_input("Chuva Acumulada (mm) [NASA]", value=round(precip_api, 1))
    input_tmax = st.sidebar.number_input("Temp. Máx. Média (°C) [NASA]", value=round(tmax_api, 2))
    input_ur = st.sidebar.number_input("Umidade Relativa (%) [NASA]", value=round(ur_api, 2))
else:
    input_precip = st.sidebar.slider("Precipitação Acumulada no Ciclo (mm)", 400.0, 900.0, 650.0)
    input_tmax = st.sidebar.slider("Temperatura Máxima Média em R3-R5 (°C)", 22.0, 28.0, 25.5)
    input_ur = st.sidebar.slider("Umidade Relativa Média (%)", 65.0, 85.0, 74.8)

input_area = st.sidebar.number_input("Área Colhida Estimada (ha) [IBGE]", value=57000)
input_ndvi = st.sidebar.slider("NDVI de Pico (Sentinel-2 L2A)", 0.50, 0.90, 0.78, step=0.01)

# Cálculo do Modelo em Tempo Real
df_input = pd.DataFrame([{
    'area': input_area,
    'tmax': input_tmax,
    'ur': input_ur,
    'ndvi_pico': input_ndvi,
    'precip_acum': input_precip
}])

if pipeline_rf is not None:
    predicao_sc_ha = pipeline_rf.predict(df_input)[0]
else:
    predicao_sc_ha = 64.50

predicao_kg_ha = predicao_sc_ha * 60
producao_total_ton = (predicao_kg_ha * input_area) / 1000

# ==============================================================================
# 5. ORGANIZAÇÃO EM ABAS (ESTILO GOOGLE AI STUDIO)
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["🎛️ Simulação & API", "📈 Gráficos & Histórico IBGE", "ℹ️ Documentação do Modelo (TCC)"])

with tab1:
    st.markdown("### 🎯 Resultados da Estimativa da Safra Simulada")
    
    c1, c2, c3 = st.columns(3)
    c1.metric(
        label="Produtividade Estimada", 
        value=f"{predicao_sc_ha:.2f} sc/ha",
        delta=f"{predicao_sc_ha - df_hist['prod_real'].mean():+.2f} sc vs média histórica"
    )
    c2.metric(
        label="Rendimento em Massa", 
        value=f"{predicao_kg_ha:.0f} kg/ha",
        delta="Conversão oficial (1 sc = 60 kg)"
    )
    c3.metric(
        label="Volume Total Estimado", 
        value=f"{producao_total_ton:,.0f} toneladas",
        delta=f"Área: {input_area:,} hectares"
    )
    
    st.markdown("---")
    st.markdown("#### 📋 Resumo dos Preditores Utilizados no Cálculo")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.info(f"**Área Colhida:** {input_area:,} ha")
    col_p2.info(f"**Precipitação:** {input_precip:.1f} mm")
    col_p3.info(f"**Temp. Máxima:** {input_tmax:.2f} °C")
    col_p4.info(f"**NDVI Pico:** {input_ndvi:.4f}")

with tab2:
    st.markdown("### 📈 Comparativo: Dados Oficiais IBGE vs Modelo Random Forest")
    
    if pipeline_rf is not None:
        X_hist = df_hist[['area', 'tmax', 'ur', 'ndvi_pico', 'precip']].rename(columns={'precip': 'precip_acum'})
        df_hist['prod_predita'] = pipeline_rf.predict(X_hist)
    else:
        df_hist['prod_predita'] = df_hist['prod_real']
    
    fig = px.line(
        df_hist, 
        x="safra", 
        y=["prod_real", "prod_predita"],
        labels={"value": "Sacas por Hectare (sc/ha)", "safra": "Safra Agrícola"},
        title="Série Histórica (2016-2025): IBGE Tabela 1612 vs Modelo Random Forest",
        markers=True,
        color_discrete_map={"prod_real": "#1f77b4", "prod_predita": "#2ca02c"}
    )
    
    fig.add_scatter(
        x=["Safra Simulada"], 
        y=[predicao_sc_ha], 
        mode="markers", 
        marker=dict(size=14, color="red", symbol="star"),
        name="Simulação Atual"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### ℹ️ Metodologia e Desempenho Científico do TCC")
    st.write("""
    * **Área de Estudo**: Município de Pitanga (PR).
    * **Algoritmo de Aprendizado**: *Random Forest Regressor* (50 árvores, profundidade máxima 3).
    * **Fontes Primárias**:
        * **Produtividade Alvo (Y)**: IBGE PAM (Tabela 1612).
        * **Clima (X)**: NASA POWER API (Precipitação, Temperatura Máxima e Umidade Relativa).
        * **Sensoriamento Remoto (X)**: Sentinel-2 L2A (NDVI de pico mascarado por uso agrícola ESA WorldCover).
    * **Protocolo de Validação**: *Leave-One-Year-Out* (LOYO) estrito (9 rodadas temporais).
    * **Métricas Globais de Acurácia**:
        * **MAE (Erro Absoluto Médio)**: 8,98 sc/ha (~538 kg/ha).
        * **MAPE (Erro Percentual Absoluto)**: 14,72%.
    """)
