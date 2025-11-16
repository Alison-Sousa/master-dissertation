import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import altair as alt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import re
import io
import base64
from datetime import datetime

st.set_page_config(
    page_title="Painel de Análise de Conflitos",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def parse_peak_size(size_str):
    size_str = str(size_str).strip().lower().replace(',', '').replace('>', '').replace('+', '')
    
    if not size_str:
        return np.nan

    value = np.nan
    try:
        if 'million' in size_str:
            number_part = re.search(r"[\d\.]+", size_str)
            if number_part:
                value = float(number_part.group(0)) * 1_000_000
        elif 'thousand' in size_str:
            number_part = re.search(r"[\d\.]+", size_str)
            if number_part:
                value = float(number_part.group(0)) * 1_000
        else:
            number_part = re.search(r"[\d\.]+", size_str)
            if number_part:
                value = float(number_part.group(0))
    except (ValueError, TypeError):
        return np.nan
        
    return value

@st.cache_data
def carregar_e_preparar_dados(uploaded_file):
    
    df = pd.read_excel(uploaded_file, dtype={'Start Date': str})
    
    df['Clean Peak Size'] = df['Peak Size'].apply(parse_peak_size)
    df['Log Peak Size'] = np.log1p(df['Clean Peak Size']) 
    
    df['Start Date'] = df['Start Date'].fillna('N/A').astype(str)
    
    df['Analysis Text'] = df['Triggers'].fillna('') + " " + df['Motivations'].fillna('')
    df['Analysis Text'] = df['Analysis Text'].str.strip()
    df.loc[df['Analysis Text'] == '', 'Analysis Text'] = 'No description'
    
    return df

@st.cache_resource
def carregar_modelo_ia():
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    return model, tokenizer

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_cluster_labels(df, n_clusters):
    try:
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(df['Analysis Text'])
        
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out(), index=df.index)
        
        df_with_tfidf = pd.concat([df, df_tfidf], axis=1)
        
        label_map = {}
        for i in range(n_clusters):
            cluster_df = df_with_tfidf[df_with_tfidf['IA Cluster'] == i]
            
            if cluster_df.empty:
                label_map[i] = f"{i}: N/A"
                continue

            cluster_tfidf_mean = cluster_df[tfidf.get_feature_names_out()].mean().sort_values(ascending=False)
            
            top_words = cluster_tfidf_mean.head(3).index.tolist()
            
            label_map[i] = f"{i}: {', '.join(top_words)}"
            
        return label_map
    except Exception:
        return {i: str(i) for i in range(n_clusters)}

@st.cache_data
def rodar_analise_ia(_df, _model_tokenizer_tuple, n_clusters=6):
    _model, _tokenizer = _model_tokenizer_tuple
    
    textos = _df['Analysis Text'].tolist()
    
    encoded_input = _tokenizer(textos, padding=True, truncation=True, return_tensors='pt', max_length=256)

    with torch.no_grad():
        model_output = _model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    embeddings_np = embeddings.numpy()
    
    if len(embeddings_np) < n_clusters:
        n_clusters = len(embeddings_np)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    _df['IA Cluster'] = kmeans.fit_predict(embeddings_np) 
    
    cluster_label_map = get_cluster_labels(_df, n_clusters)
    _df['IA Cluster Label'] = _df['IA Cluster'].map(cluster_label_map)
    
    if len(embeddings_np) > 1:
        tsne_perplexity = min(30, max(1, len(embeddings_np) - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity, n_iter=300)
        tsne_results = tsne.fit_transform(embeddings_np) 
        _df['IA Vis (x)'] = tsne_results[:, 0]
        _df['IA Vis (y)'] = tsne_results[:, 1]
    else:
        _df['IA Vis (x)'] = 0
        _df['IA Vis (y)'] = 0
        
    return _df

def get_table_download_link_excel(df, filename="dados.xlsx", link_text="Baixar dados (Excel)"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Dados')
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{link_text}</a>'

st.title("🌍 Painel de Análise de Conflitos Globais")
st.markdown("Uma plataforma interativa para análise geoespacial e de IA sobre tensões globais.")

st.sidebar.header("Controle de Dados e Filtros")

uploaded_file = st.sidebar.file_uploader(
    "Faça upload do seu arquivo de dados (gpt.xlsx)",
    type=["xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Por favor, faça o upload do arquivo `gpt.xlsx` na barra lateral para começar a análise.")
    st.stop()

try:
    with st.spinner("Carregando e limpando dados..."):
        df_base = carregar_e_preparar_dados(uploaded_file) 
    
    with st.spinner("Carregando modelo de IA (Hugging Face / PyTorch)..."):
        model_tuple = carregar_modelo_ia() 

    with st.spinner("Rodando análise de IA (Clustering, TF-IDF e T-SNE)..."):
        df_final = rodar_analise_ia(df_base.copy(), model_tuple, n_clusters=6) 
except pd.errors.ParserError:
    st.error("Erro ao ler o arquivo. Parece que o formato do Excel é inválido.")
    st.stop()
except Exception as e:
    st.error(f"Ocorreu um erro durante a inicialização: {e}")
    st.stop()

st.sidebar.header("Filtros Interativos")

dates_text = sorted(df_final['Start Date'].unique())
dates_selecionadas = st.sidebar.multiselect(
    "Filtrar por Data",
    options=dates_text,
    default=dates_text
)
    
paises = sorted(df_final['Country'].unique())
paises_selecionados = st.sidebar.multiselect(
    "Filtrar por País",
    options=paises,
    default=paises
)

clusters_ia = sorted(df_final['IA Cluster Label'].unique())
clusters_selecionados = st.sidebar.multiselect(
    "Filtrar por Cluster de IA",
    options=clusters_ia,
    default=clusters_ia
)

df_filtrado = df_final[
    (df_final['Start Date'].isin(dates_selecionadas)) &
    (df_final['Country'].isin(paises_selecionados)) &
    (df_final['IA Cluster Label'].isin(clusters_selecionados))
]

if df_filtrado.empty:
    st.warning("Nenhum dado selecionado com base nos filtros. Ajuste-os para ver as análises.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard Principal", 
    "Análise de Risco (IA)", 
    "Mapa de Risco Global", 
    "Explorador de Dados"
])

with tab1:
    st.header("Dashboard Principal de Conflitos")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Contagem de Conflitos por País")
        chart_bar_pais = alt.Chart(df_filtrado).mark_bar(color="#1f77b4").encode(
            x=alt.X('count()', title='Contagem de Conflitos'),
            y=alt.Y('Country', title='País', sort='-x'),
            tooltip=['Country', 'count()']
        ).properties(
            height=400
        ).interactive()
        st.altair_chart(chart_bar_pais, use_container_width=True)

    with col2:
        st.subheader("Visualização dos Clusters de IA (t-SNE)")
        st.markdown("Como a IA agrupou os conflitos. Pontos próximos têm textos semelhantes.")
        
        cluster_domain = sorted(df_filtrado['IA Cluster Label'].unique())
        cluster_range = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        chart_tsne = alt.Chart(df_filtrado).mark_circle().encode(
            x=alt.X('IA Vis (x)', title='Dimensão t-SNE 1', scale=alt.Scale(zero=False)),
            y=alt.Y('IA Vis (y)', title='Dimensão t-SNE 2', scale=alt.Scale(zero=False)),
            color=alt.Color('IA Cluster Label', title="Cluster de IA", 
                            scale=alt.Scale(domain=cluster_domain, range=cluster_range[:len(cluster_domain)])),
            tooltip=['Protest Name', 'Country', 'Analysis Text', 'IA Cluster Label']
        ).properties(
            height=400
        ).interactive()
        st.altair_chart(chart_tsne, use_container_width=True)

with tab2:
    st.header("Análise de Risco para Negócios")
    st.markdown("Análise dos locais com maior instabilidade com base no tamanho médio dos conflitos (`Clean Peak Size`).")
    
    df_risco = df_filtrado.dropna(subset=['Clean Peak Size'])

    if not df_risco.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 15 Países por Risco (Tamanho Médio)")
            df_risco_pais = df_risco.groupby('Country', as_index=False)['Clean Peak Size'].mean().sort_values(by='Clean Peak Size', ascending=False)
            
            chart_risco_pais = alt.Chart(df_risco_pais.head(15)).mark_bar(color="#d62728").encode(
                x=alt.X('Clean Peak Size', title='Tamanho Médio do Protesto'),
                y=alt.Y('Country', title='País', sort='-x'),
                tooltip=['Country', 'Clean Peak Size']
            ).properties(
                height=400
            ).interactive()
            st.altair_chart(chart_risco_pais, use_container_width=True)

        with col2:
            st.subheader("Top 15 Cidades por Risco (Tamanho Médio)")
            df_risco_cidade = df_risco[df_risco['Capital city'] != 'X'].groupby('Capital city', as_index=False)['Clean Peak Size'].mean().sort_values(by='Clean Peak Size', ascending=False)
            
            chart_risco_cidade = alt.Chart(df_risco_cidade.head(15)).mark_bar(color="#ff7f0e").encode(
                x=alt.X('Clean Peak Size', title='Tamanho Médio do Protesto'),
                y=alt.Y('Capital city', title='Cidade', sort='-x'),
                tooltip=['Capital city', 'Clean Peak Size']
            ).properties(
                height=400
            ).interactive()
            st.altair_chart(chart_risco_cidade, use_container_width=True)
    else:
        st.warning("Nenhum dado de 'Tamanho' válido para calcular a análise de risco.")

with tab3:
    st.header("Mapa de Risco Global (por País)")
    st.markdown("Nível de risco com base no tamanho médio dos protestos (com filtros aplicados). Países em cinza não possuem dados.")

    df_risco_pais_mapa = df_filtrado.dropna(subset=['Clean Peak Size']).groupby('Country', as_index=False)['Clean Peak Size'].mean()

    if not df_risco_pais_mapa.empty:
        fig_mapa_pais = go.Figure(go.Choropleth(
            locations=df_risco_pais_mapa['Country'],
            locationmode='country names',
            z=df_risco_pais_mapa['Clean Peak Size'],
            colorscale='Reds',
            colorbar_title='Tamanho Médio (Risco)',
            autocolorscale=False,
            reversescale=False,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            text=df_risco_pais_mapa.apply(lambda row: f"<b>{row['Country']}</b><br>Risco (Tamanho Médio): {row['Clean Peak Size']:,.0f}", axis=1),
            hoverinfo='text'
        ))

        fig_mapa_pais.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='equirectangular'
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            height=500
        )
        st.plotly_chart(fig_mapa_pais, use_container_width=True)
    else:
        st.warning("Nenhum dado de 'Tamanho' válido para calcular o mapa de risco.")


with tab4:
    st.header("Explorador de Dados e Downloads (Excel)")
    st.markdown("Navegue pelos dados filtrados e faça o download das tabelas de análise como arquivos `.xlsx`.")
    
    st.subheader("Dados Filtrados (Tabela Completa)")
    df_display = df_filtrado.copy()
    df_display['Peak Size'] = df_display['Peak Size'].astype(str)
    st.dataframe(df_display)
    st.markdown(get_table_download_link_excel(df_filtrado, "dados_filtrados.xlsx", "Baixar Tabela Filtrada (Excel)"), unsafe_allow_html=True)
    
    st.subheader("Resumo da Análise de Risco")
    
    try:
        df_risco_pais_dl = df_filtrado.dropna(subset=['Clean Peak Size']).groupby('Country', as_index=False)['Clean Peak Size'].mean().sort_values(by='Clean Peak Size', ascending=False)
        df_risco_cidade_dl = df_filtrado.dropna(subset=['Clean Peak Size']).groupby('Capital city', as_index=False)['Clean Peak Size'].mean().sort_values(by='Clean Peak Size', ascending=False)
        
        st.markdown("Tabelas de risco por país e cidade, ordenadas por tamanho médio de protesto.")
        st.markdown(get_table_download_link_excel(df_risco_pais_dl, "analise_risco_pais.xlsx", "Baixar Risco por País (Excel)"), unsafe_allow_html=True)
        st.markdown(get_table_download_link_excel(df_risco_cidade_dl, "analise_risco_cidade.xlsx", "Baixar Risco por Cidade (Excel)"), unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Não foi possível gerar as tabelas de risco: {e}")

    st.subheader("Resumo dos Clusters de IA")
    try:
        df_cluster_summary = df_final.groupby('IA Cluster Label').agg(
            Contagem=('Protest Name', 'count'),
            Exemplos_Textos=('Analysis Text', lambda x: ' | '.join(x.unique()[:2]))
        ).reset_index()
        
        st.dataframe(df_cluster_summary)
        st.markdown(get_table_download_link_excel(df_cluster_summary, "resumo_clusters_ia.xlsx", "Baixar Resumo dos Clusters (Excel)"), unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Não foi possível gerar o resumo dos clusters: {e}")