"""
Aplicação Streamlit para Detecção de Botnets IoT
Utiliza o dataset N-BaIoT para treinar e avaliar modelos de classificação
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from io import StringIO

from utils import (
    load_dataset, preprocess_data, train_random_forest,
    train_xgboost, train_model,
    evaluate_model, get_feature_importance, count_csv_files,
    find_suitable_target_columns, get_available_devices
)


def make_arrow_compatible(df):
    """
    Converte DataFrame para ser compatível com PyArrow/Streamlit
    Converte tipos problemáticos para strings ou tipos compatíveis
    """
    df_copy = df.copy()
    
    # Converte tipos object que podem causar problemas
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            # Tenta converter para string se possível
            try:
                df_copy[col] = df_copy[col].astype(str)
            except:
                pass
    
    return df_copy

# Configuração da página
st.set_page_config(
    page_title="Detecção de Botnets IoT",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar a aparência
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar - Navegação
st.sidebar.title("📋 Navegação")
page = st.sidebar.radio(
    "Selecione uma página:",
    ["🏠 Dashboard", "📤 Upload & Pré-processamento", "🤖 Treinamento", "📈 Resultados"]
)

# Título principal (só mostra se não for dashboard)
if page != "🏠 Dashboard":
    st.markdown('<h1 class="main-header">🛡️ N-BaIoT Intrusion Detection Lab</h1>', unsafe_allow_html=True)
    st.markdown("---")

# Inicialização de variáveis de sessão
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'dataset_path' not in st.session_state:
    st.session_state.dataset_path = None
if 'auto_download_attempted' not in st.session_state:
    st.session_state.auto_download_attempted = False

# Função para fazer download do dataset
def download_dataset():
    """Faz o download do dataset do Kaggle"""
    try:
        import kagglehub
        path = kagglehub.dataset_download("mkashifn/nbaiot-dataset")
        st.session_state.dataset_path = path
        return path, None
    except Exception as e:
        return None, str(e)

# Download automático do dataset na primeira execução
# Mostra um banner no topo da página se o dataset ainda não foi baixado
if not st.session_state.dataset_path and not st.session_state.auto_download_attempted:
    st.session_state.auto_download_attempted = True
    
    # Container destacado para o download
    with st.container():
        st.markdown("---")
        st.markdown("### 🔄 Download Automático do Dataset")
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        status_placeholder.info("🔄 Iniciando download do dataset N-BaIoT do Kaggle... Isso pode levar alguns minutos na primeira vez.")
        
        progress_placeholder.progress(0, text="Conectando ao Kaggle...")
        path, error = download_dataset()
        
        if path:
            progress_placeholder.progress(100, text="Download concluído!")
            status_placeholder.success(f"✅ **Dataset baixado com sucesso!**\n\n📁 Localização: `{path}`\n\n💡 Agora você pode carregar os dados na página 'Exploração de Dados'")
            progress_placeholder.empty()
            time.sleep(1)
        else:
            progress_placeholder.empty()
            status_placeholder.error(f"❌ **Erro ao baixar o dataset automaticamente**\n\nErro: `{error}`")
            st.warning("💡 **Soluções possíveis:**")
            st.markdown("""
            - Certifique-se de que suas credenciais do Kaggle estão configuradas (veja `kaggle_setup.md`)
            - Verifique sua conexão com a internet
            - Você pode tentar baixar manualmente na página 'Exploração de Dados'
            """)
        
        st.markdown("---")

# Página: Dashboard
if page == "🏠 Dashboard":
    # Header do Dashboard
    st.markdown('<h1 class="main-header">🛡️ N-BaIoT Intrusion Detection Lab</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Cards principais
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 2rem; border-radius: 10px; text-align: center; border: 2px solid #1f77b4;'>
            <h2 style='color: #1f77b4; margin-bottom: 1rem;'>📤</h2>
            <h3 style='color: #1f77b4; margin-bottom: 1rem;'>Pré-processamento</h3>
            <p style='color: #666;'>Carregar dataset, visualizar, limpar e preparar dados</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Ir para Pré-processamento", key="btn_preprocess", use_container_width=True):
            st.session_state.page_redirect = "📤 Upload & Pré-processamento"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 2rem; border-radius: 10px; text-align: center; border: 2px solid #1f77b4;'>
            <h2 style='color: #1f77b4; margin-bottom: 1rem;'>🤖</h2>
            <h3 style='color: #1f77b4; margin-bottom: 1rem;'>Treinamento</h3>
            <p style='color: #666;'>Treinar modelos de ML e ajustar hiperparâmetros</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Ir para Treinamento", key="btn_train", use_container_width=True):
            st.session_state.page_redirect = "🤖 Treinamento"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 2rem; border-radius: 10px; text-align: center; border: 2px solid #1f77b4;'>
            <h2 style='color: #1f77b4; margin-bottom: 1rem;'>📈</h2>
            <h3 style='color: #1f77b4; margin-bottom: 1rem;'>Resultados</h3>
            <p style='color: #666;'>Visualizar métricas e análises detalhadas</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Ver Resultados", key="btn_results", use_container_width=True):
            st.session_state.page_redirect = "📈 Resultados"
            st.rerun()
    
    st.markdown("---")
    
    # Indicadores/Estatísticas
    st.subheader("📊 Indicadores do Sistema")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        if st.session_state.dataset_loaded and st.session_state.data is not None:
            n_samples = len(st.session_state.data)
            st.metric("📦 Amostras Carregadas", f"{n_samples:,}")
        else:
            st.metric("📦 Amostras Carregadas", "0")
    
    with col_stat2:
        if st.session_state.dataset_loaded and st.session_state.data is not None and 'device' in st.session_state.data.columns:
            n_devices = st.session_state.data['device'].nunique()
            st.metric("📱 Dispositivos", n_devices)
        else:
            st.metric("📱 Dispositivos", "0")
    
    with col_stat3:
        if st.session_state.model_trained:
            st.metric("✅ Modelos Treinados", "1")
        else:
            st.metric("✅ Modelos Treinados", "0")
    
    with col_stat4:
        if st.session_state.dataset_path:
            st.metric("💾 Dataset", "Disponível")
        else:
            st.metric("💾 Dataset", "Não baixado")
    
    st.markdown("---")
    
    # Status e informações
    st.subheader("ℹ️ Status do Sistema")
    
    if st.session_state.dataset_path:
        st.success(f"✅ **Dataset baixado!** Localização: `{st.session_state.dataset_path}`")
    else:
        st.warning("⚠️ Dataset ainda não foi baixado. Vá para 'Pré-processamento' para baixar.")
    
    if st.session_state.dataset_loaded:
        st.success("✅ **Dados carregados na memória!**")
    else:
        st.info("ℹ️ Dados ainda não foram carregados. Vá para 'Pré-processamento' para carregar.")
    
    if st.session_state.model_trained:
        st.success("✅ **Modelo treinado!** Vá para 'Resultados' para ver métricas.")
    else:
        st.info("ℹ️ Nenhum modelo treinado ainda. Vá para 'Treinamento' para treinar um modelo.")
    
    # Redirecionamento se necessário
    if hasattr(st.session_state, 'page_redirect'):
        page = st.session_state.page_redirect
        del st.session_state.page_redirect

# Página: Upload & Pré-processamento
elif page == "📤 Upload & Pré-processamento":
    st.header("📤 Upload & Pré-processamento")
    
    # Seção de carregamento
    st.subheader("📤 Carregamento")
    
    # Opções de carregamento
    with st.expander("⚙️ Opções de Carregamento", expanded=True):
        st.markdown("### 📋 Configurações de Carregamento")
        
        # Descobre dispositivos disponíveis
        available_devices = []
        device_names_preview = {}
        if st.session_state.dataset_path:
            try:
                available_devices, device_names_preview = get_available_devices(st.session_state.dataset_path)
            except:
                pass
        
        if available_devices:
            st.info(f"📱 **Dispositivos disponíveis no dataset:** {len(available_devices)}")
            
            # Mostra lista de dispositivos disponíveis
            device_list = []
            for dev_num in available_devices:
                dev_name = device_names_preview.get(dev_num, f"Device {dev_num}")
                device_list.append(f"{dev_name} (Device {dev_num})")
            
            st.markdown("**Dispositivos encontrados:**")
            for dev_info in device_list:
                st.markdown(f"- {dev_info}")
        
        # Seleção de dispositivos
        if available_devices:
            st.markdown("### 📱 Seleção de Dispositivos")
            st.markdown("""
            **Como funciona:**
            - Cada dispositivo tem múltiplos arquivos CSV (benign + diferentes tipos de ataque)
            - Ao selecionar um dispositivo, TODOS os seus arquivos serão carregados
            - Isso garante que você tenha dados completos de cada dispositivo
            """)
            
            # Multi-select de dispositivos
            device_options = []
            for dev_num in available_devices:
                dev_name = device_names_preview.get(dev_num, f"Device {dev_num}")
                device_options.append(f"{dev_name} (Device {dev_num})")
            
            selected_devices_display = st.multiselect(
                "Selecione os Dispositivos para Carregar:",
                device_options,
                default=device_options[:1] if device_options else [],  # Seleciona o primeiro por padrão
                help="Selecione um ou mais dispositivos. Cada dispositivo terá TODOS os seus arquivos CSV carregados (benign + todos os tipos de ataque)."
            )
            
            # Extrai números dos dispositivos selecionados
            selected_devices = []
            for display_name in selected_devices_display:
                # Extrai o número do dispositivo do nome (último número entre parênteses)
                import re
                match = re.search(r'Device (\d+)', display_name)
                if match:
                    selected_devices.append(int(match.group(1)))
            
            if not selected_devices:
                st.warning("⚠️ Selecione pelo menos um dispositivo para carregar.")
        else:
            selected_devices = None  # Carrega todos se não conseguir detectar
            st.info("ℹ️ Não foi possível detectar dispositivos. Carregando todos os arquivos disponíveis.")
        
        sample_size = st.number_input(
            "Amostra por arquivo (opcional, deixe 0 para carregar tudo)",
            min_value=0,
            max_value=1000000,
            value=0,
            step=10000,
            help="Se o dataset for muito grande, você pode carregar apenas uma amostra de cada arquivo para economizar memória. 0 = carregar tudo. Recomendado: 0 (tudo) ou 50000-100000 para testes rápidos."
        )
        if sample_size == 0:
            sample_size = None
        
        # Estimativa de memória (se dispositivos selecionados)
        if selected_devices and available_devices:
            # Estima quantos arquivos serão carregados (cada dispositivo tem ~11 arquivos)
            estimated_files = len(selected_devices) * 11  # Aproximação: cada dispositivo tem ~11 arquivos
            if sample_size:
                estimated_rows = estimated_files * sample_size
                estimated_mb = (estimated_rows * 50) / (1024 * 1024)
                st.info(f"💾 **Estimativa:** ~{estimated_rows:,} linhas, ~{estimated_mb:.1f} MB de memória")
            else:
                st.info(f"💾 **Nota:** Carregando dados de {len(selected_devices)} dispositivo(s). Cada dispositivo tem múltiplos arquivos CSV (benign + ataques).")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Verifica se pode carregar
        can_load = True
        if available_devices:
            if not selected_devices or len(selected_devices) == 0:
                can_load = False
                st.warning("⚠️ Selecione pelo menos um dispositivo para carregar.")
        
        if st.button("🔄 Carregar Dataset do Kaggle", type="primary", width='stretch', disabled=not can_load):
            progress_container = st.container()
            with progress_container:
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                try:
                    if selected_devices:
                        devices_str = ', '.join([device_names_preview.get(d, f"Device {d}") for d in selected_devices])
                        status_placeholder.info(f"🔄 Iniciando carregamento de {len(selected_devices)} dispositivo(s): {devices_str}...")
                    else:
                        status_placeholder.info("🔄 Iniciando carregamento de todos os dispositivos disponíveis...")
                    progress_bar.progress(10)
                    
                    # Se o dataset já foi baixado, usa o caminho salvo
                    if st.session_state.dataset_path:
                        status_placeholder.info("📂 Usando dataset já baixado...")
                        progress_bar.progress(20)
                        # Carrega usando o caminho já baixado
                        df, dataset_path, device_names = load_dataset(
                            devices_to_load=selected_devices if selected_devices else None,
                            sample_size=sample_size,
                            dataset_path=st.session_state.dataset_path
                        )
                    else:
                        status_placeholder.info("📥 Fazendo download do dataset...")
                        progress_bar.progress(20)
                        # Faz download e carrega
                        df, dataset_path, device_names = load_dataset(
                            devices_to_load=selected_devices if selected_devices else None,
                            sample_size=sample_size
                        )
                        st.session_state.dataset_path = dataset_path
                    
                    # Salva os nomes dos dispositivos
                    st.session_state.device_names = device_names
                    
                    progress_bar.progress(80)
                    status_placeholder.info("✅ Processando dados...")
                    
                    st.session_state.data = df
                    st.session_state.dataset_loaded = True
                    
                    progress_bar.progress(100)
                    status_placeholder.empty()
                    progress_bar.empty()
                    
                    # Mostra informações detalhadas
                    st.success(f"✅ **Dataset carregado com sucesso!**")
                    
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.metric("Total de Linhas", f"{len(df):,}")
                    with col_info2:
                        st.metric("Total de Colunas", len(df.columns))
                    with col_info3:
                        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
                        st.metric("Memória Usada", f"{memory_mb:.2f} MB")
                    
                    st.info(f"📁 **Localização:** `{dataset_path}`")
                    
                    if selected_devices and len(selected_devices) < len(available_devices):
                        st.info(f"💡 **Dica:** Você carregou {len(selected_devices)} dispositivo(s). Para mais dados, selecione mais dispositivos nas opções acima.")
                    
                except MemoryError as e:
                    progress_bar.empty()
                    status_placeholder.error("❌ **Erro de Memória**")
                    st.error("O dataset é muito grande para a memória disponível.")
                    st.warning("**Soluções:**")
                    st.markdown("""
                    - Reduza o número de arquivos (tente 1-3 arquivos)
                    - Use uma amostra menor (ex: 50000 linhas por arquivo)
                    - Feche outros aplicativos para liberar memória
                    """)
                except Exception as e:
                    progress_bar.empty()
                    status_placeholder.error("❌ **Erro ao carregar dataset**")
                    st.error(f"Erro: `{str(e)}`")
                    st.info("💡 Certifique-se de que suas credenciais do Kaggle estão configuradas corretamente.")
                    st.info("💡 **Dicas:**")
                    st.markdown("""
                    - Tente reduzir o número de arquivos
                    - Use uma amostra menor (ex: 50000 linhas)
                    - Verifique sua conexão com a internet
                    """)
    
    # Mostra status do download automático
    if st.session_state.dataset_path and not st.session_state.dataset_loaded:
        st.info(f"📥 Dataset já foi baixado em: {st.session_state.dataset_path}")
        st.info("💡 Clique no botão acima para carregar os dados na memória.")
    
    # Seção de visualização
    if st.session_state.dataset_loaded and st.session_state.data is not None:
        df = st.session_state.data
        
        st.subheader("2. Informações do Dataset")
        
        # Estatísticas básicas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Registros", f"{len(df):,}")
        with col2:
            st.metric("Total de Features", len(df.columns))
        with col3:
            st.metric("Valores Nulos", df.isnull().sum().sum())
        with col4:
            st.metric("Memória Usada", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Visualização dos dados
        st.subheader("3. Visualização dos Dados")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Primeiras Linhas", "📊 Estatísticas", "📈 Distribuições", "🔍 Informações"])
        
        with tab1:
            try:
                df_display = make_arrow_compatible(df.head(100))
                st.dataframe(df_display, width='stretch')
            except Exception as e:
                st.warning(f"Erro ao exibir dataframe: {str(e)}")
                st.text("Tentando exibir como texto...")
                st.text(str(df.head(100)))
        
        with tab2:
            try:
                desc_df = df.describe()
                # Converte todos os valores para float64 explícito
                desc_df = desc_df.astype('float64')
                st.dataframe(desc_df, width='stretch')
            except Exception as e:
                st.warning(f"Erro ao exibir estatísticas: {str(e)}")
                # Fallback: converte para string
                desc_df = df.describe()
                for col in desc_df.columns:
                    desc_df[col] = desc_df[col].astype(str)
                st.dataframe(desc_df, width='stretch')
        
        with tab3:
            # Seleciona colunas numéricas para visualização
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Selecione uma coluna para visualizar:", numeric_cols[:20])
                if selected_col:
                    fig = px.histogram(df, x=selected_col, nbins=50, title=f"Distribuição de {selected_col}")
                    st.plotly_chart(fig, width='stretch')
            else:
                st.warning("Nenhuma coluna numérica encontrada para visualização.")
        
        with tab4:
            st.text("Informações do DataFrame:")
            try:
                buffer = StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
            except Exception as e:
                st.warning(f"Erro ao exibir informações: {str(e)}")
                st.text(f"Shape: {df.shape}")
                st.text(f"Colunas: {list(df.columns)}")
            
            st.text("\nTipos de Dados:")
            try:
                # Converte tipos para string para evitar problemas com PyArrow
                dtypes_df = df.dtypes.to_frame(name="Tipo")
                dtypes_df['Tipo'] = dtypes_df['Tipo'].astype(str)
                # Reseta o índice para garantir compatibilidade
                dtypes_df = dtypes_df.reset_index()
                dtypes_df.columns = ['Coluna', 'Tipo']
                st.dataframe(dtypes_df, width='stretch')
            except Exception as e:
                st.warning(f"Erro ao exibir tipos: {str(e)}")
                # Fallback: exibe como texto
                for col in df.columns:
                    st.text(f"{col}: {str(df[col].dtype)}")
        
        # Detecção de coluna target
        st.subheader("4. Configuração para Treinamento")
        
        # Informações sobre o dataset N-BaIoT
        if 'label' in df.columns:
            unique_labels = df['label'].unique()
            label_counts = df['label'].value_counts()
            
            st.info("📋 **Sobre o Dataset N-BaIoT:**")
            st.markdown("""
            O dataset N-BaIoT contém tráfego de rede de dispositivos IoT:
            - **Benign**: Tráfego normal (sem ataque)
            - **Mirai**: Ataques do botnet Mirai (scan, ack, syn, udp, udpplain)
            - **Gafgyt (BASHLITE)**: Ataques do botnet Gafgyt (udp, junk, scan, tcp, combo)
            
            Os labels são extraídos automaticamente do nome do arquivo.
            """)
            
            st.markdown("**Labels encontrados no dataset:**")
            label_info_df = pd.DataFrame({
                'Label': label_counts.index,
                'Amostras': label_counts.values,
                '%': (label_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(make_arrow_compatible(label_info_df), width='stretch')
        
        # Informações sobre dispositivos
        if 'device' in df.columns:
            unique_devices = sorted(df['device'].unique())
            device_counts = df['device'].value_counts().sort_index()
            device_names = getattr(st.session_state, 'device_names', {})
            
            st.markdown("**📱 Dispositivos encontrados no dataset:**")
            
            # Cria DataFrame com nomes dos dispositivos
            device_display_names = []
            for d in device_counts.index:
                device_name = device_names.get(d, f"Device {d}")
                device_display_names.append(f"{device_name} (Device {d})")
            
            device_info_df = pd.DataFrame({
                'Dispositivo': device_display_names,
                'Amostras': device_counts.values,
                '%': (device_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(make_arrow_compatible(device_info_df), width='stretch')
            
            # Opção de treinar por dispositivo
            st.markdown("---")
            st.subheader("🎯 Modo de Treinamento")
            
            # Se houver múltiplos dispositivos, permite escolher
            if len(unique_devices) > 1:
                train_mode = st.radio(
                    "Escolha a estratégia de treinamento:",
                    ["Treinar por Dispositivo (Recomendado)", "Treinar com Todos os Dispositivos"],
                    help="Treinar por dispositivo melhora a sensibilidade. Treinar com todos combina os dados de todos os dispositivos.",
                    index=0
                )
                
                train_by_device = (train_mode == "Treinar por Dispositivo (Recomendado)")
            else:
                train_by_device = True
                st.info(f"ℹ️ Apenas 1 dispositivo encontrado. Treinando modelo específico para este dispositivo.")
            
            if train_by_device:
                # Cria lista de opções com nomes dos dispositivos
                device_options = []
                for d in unique_devices:
                    device_name = device_names.get(d, f"Device {d}")
                    device_options.append(f"{device_name} (Device {d})")
                
                selected_device_display = st.selectbox(
                    "Selecione o Dispositivo para Treinar:",
                    device_options,
                    format_func=lambda x: f"{x} - {device_counts[unique_devices[device_options.index(x)]]:,} amostras",
                    help="Cada dispositivo terá seu próprio modelo treinado apenas com seus dados."
                )
                
                # Extrai o número do dispositivo da seleção
                selected_device = unique_devices[device_options.index(selected_device_display)]
                st.session_state.selected_device = selected_device
                st.session_state.train_by_device = True
                
                # Mostra informações sobre o dispositivo selecionado
                device_df = df[df['device'] == selected_device]
                device_labels = device_df['label'].value_counts()
                device_name = device_names.get(selected_device, f"Device {selected_device}")
                
                st.success(f"✅ **{device_name} (Device {selected_device}) selecionado:** {len(device_df):,} amostras")
                st.markdown(f"**Distribuição de labels no {device_name}:**")
                device_label_df = pd.DataFrame({
                    'Label': device_labels.index,
                    'Amostras': device_labels.values,
                    '%': (device_labels.values / len(device_df) * 100).round(2)
                })
                st.dataframe(make_arrow_compatible(device_label_df), width='stretch')
            else:
                st.session_state.train_by_device = False
                st.session_state.selected_device = None
                
                # Mostra informações sobre todos os dispositivos combinados
                total_samples = len(df)
                all_labels = df['label'].value_counts()
                
                st.success(f"✅ **Treinando com todos os {len(unique_devices)} dispositivos combinados:** {total_samples:,} amostras")
                st.markdown("**Distribuição de labels (todos os dispositivos):**")
                all_labels_df = pd.DataFrame({
                    'Label': all_labels.index,
                    'Amostras': all_labels.values,
                    '%': (all_labels.values / total_samples * 100).round(2)
                })
                st.dataframe(make_arrow_compatible(all_labels_df), width='stretch')
                
                st.info("💡 **Vantagem:** Mais dados para treinar, mas pode ter menor sensibilidade por dispositivo.")
        
        # Encontra colunas adequadas para classificação
        suitable_cols = find_suitable_target_columns(df)
        suitable_targets = [col['column'] for col in suitable_cols if col['is_suitable']]
        recommended_targets = [col['column'] for col in suitable_cols if col['is_suitable'] and col['has_keyword']]
        
        # Mostra informações sobre colunas adequadas
        if recommended_targets:
            st.success(f"✅ **Colunas recomendadas encontradas:** {', '.join(recommended_targets[:5])}")
            default_target = recommended_targets[0]
        elif suitable_targets:
            st.info(f"💡 **Colunas adequadas encontradas:** {', '.join(suitable_targets[:5])}")
            default_target = suitable_targets[0]
        else:
            st.warning("⚠️ **Atenção:** Nenhuma coluna claramente adequada para classificação foi encontrada.")
            st.info("💡 Você pode selecionar manualmente uma coluna, mas verifique se ela tem poucos valores únicos (classes).")
            # Tenta encontrar por keywords mesmo que não seja "suitable"
            possible_targets = [col['column'] for col in suitable_cols if col['has_keyword']]
            if possible_targets:
                default_target = possible_targets[0]
            else:
                default_target = df.columns[-1]
        
        # Tabela com informações das colunas
        with st.expander("📊 Ver todas as colunas e adequação para classificação", expanded=False):
            cols_info_df = pd.DataFrame(suitable_cols)
            cols_info_df['Status'] = cols_info_df.apply(
                lambda x: '✅ Recomendada' if x['is_suitable'] and x['has_keyword'] 
                else '✅ Adequada' if x['is_suitable'] 
                else '⚠️ Muitos valores únicos' if x['unique_count'] > 50 
                else '❌ Poucos valores únicos',
                axis=1
            )
            display_df = cols_info_df[['column', 'unique_count', 'percentage', 'Status']].copy()
            display_df.columns = ['Coluna', 'Valores Únicos', '% Únicos', 'Status']
            # Garante que todos os valores numéricos sejam compatíveis
            display_df['Valores Únicos'] = display_df['Valores Únicos'].astype('int64')
            display_df['% Únicos'] = display_df['% Únicos'].astype('float64')
            st.dataframe(display_df, width='stretch')
        
        # Selectbox com todas as colunas, mas destacando as adequadas
        all_columns = df.columns.tolist()
        target_column = st.selectbox(
            "Selecione a coluna target (classe):",
            all_columns,
            index=all_columns.index(default_target) if default_target in all_columns else 0,
            help="Selecione uma coluna com poucos valores únicos (classes categóricas). Colunas recomendadas aparecem primeiro na lista acima."
        )
        
        # Mostra distribuição da classe target
        if target_column:
            unique_count = df[target_column].nunique()
            total_count = len(df[target_column].dropna())
            percentage = (unique_count / total_count * 100) if total_count > 0 else 0
            
            # Validação visual
            if unique_count > max(50, total_count * 0.5):
                st.error(f"⚠️ **Atenção:** A coluna '{target_column}' tem {unique_count} valores únicos ({percentage:.2f}% dos dados). Isso parece ser uma variável contínua (regressão), não classificação.")
                st.warning("Por favor, selecione uma coluna diferente com poucos valores únicos (classes categóricas).")
            elif unique_count < 2:
                st.error(f"⚠️ **Atenção:** A coluna '{target_column}' tem menos de 2 valores únicos. Não é possível fazer classificação.")
            else:
                st.success(f"✅ Coluna adequada para classificação: {unique_count} classes distintas")
            
            st.write(f"**Distribuição da classe '{target_column}':**")
            class_dist = df[target_column].value_counts()
            
            # Limita a exibição se houver muitas classes
            if len(class_dist) > 20:
                st.info(f"Mostrando apenas as 20 classes mais frequentes (total: {len(class_dist)} classes)")
                class_dist_display = class_dist.head(20)
            else:
                class_dist_display = class_dist
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.bar(x=class_dist_display.index.astype(str), y=class_dist_display.values,
                           labels={'x': target_column, 'y': 'Frequência'},
                           title=f"Distribuição das Classes ({len(class_dist)} classes)")
                st.plotly_chart(fig, width='stretch')
            with col2:
                freq_df = class_dist_display.to_frame(name="Frequência")
                freq_df['Frequência'] = freq_df['Frequência'].astype('int64')
                st.dataframe(freq_df, width='stretch')
        
        st.session_state.target_column = target_column
        
        # Botão para ir para treinamento
        st.markdown("---")
        if st.button("➡️ Ir para Treinamento", type="primary", use_container_width=True):
            st.session_state.page_redirect = "🤖 Treinamento"
            st.rerun()

# Página: Treinamento
elif page == "🤖 Treinamento":
    st.header("🤖 Treinamento dos Modelos")
    
    if not st.session_state.dataset_loaded or st.session_state.data is None:
        st.warning("⚠️ Por favor, carregue o dataset primeiro na página 'Upload & Pré-processamento'")
        if st.button("📤 Ir para Pré-processamento"):
            st.session_state.page_redirect = "📤 Upload & Pré-processamento"
            st.rerun()
    else:
        df = st.session_state.data
        
        # ========== SEÇÃO DE TREINAMENTO ==========
        st.subheader("🎯 Escolher Tipo de Modelo")
        
        # Seleção do algoritmo
        algorithm = st.radio(
            "Selecione o Algoritmo de Machine Learning:",
            ["Random Forest", "XGBoost"],
            help="Escolha o algoritmo que deseja usar para classificação",
            horizontal=True
        )
        
        st.session_state.selected_algorithm = algorithm.lower().replace(" ", "_")
        
        # Configuração de divisão de dados
        st.subheader("⚙️ Configurações do Modelo")
        test_size = st.slider(
            "Proporção de Dados para Teste",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Proporção do dataset que será usado para teste.",
            key="test_size_training"
        )
        
        # Hiperparâmetros específicos por algoritmo
        st.subheader("⚙️ Hiperparâmetros")
        
        # Informações sobre hiperparâmetros recomendados
        with st.expander("💡 Valores Recomendados para N-BaIoT", expanded=False):
            st.markdown("""
            **Baseado em pesquisas e melhores práticas para o dataset N-BaIoT:**
            
            **Random Forest:**
            - **n_estimators**: 50-200 (valores menores reduzem overfitting)
            - **max_depth**: 10-20 (profundidade moderada)
            - **min_samples_split**: 2-5
            - **min_samples_leaf**: 1-2
            
            **XGBoost:**
            - **n_estimators**: 50-200
            - **max_depth**: 3-8 (valores menores são mais conservadores)
            - **learning_rate**: 0.01-0.1 (valores menores = menos overfitting)
            - **subsample**: 0.7-0.9 (reduz overfitting)
            - **colsample_bytree**: 0.7-0.9
            
            **💡 Dica:** Se você obteve 0.93 com parâmetros menores, isso indica que valores mais conservadores 
            estão funcionando melhor para evitar overfitting. Continue experimentando com valores menores!
            """)
        
        if algorithm == "Random Forest":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Parâmetros do Random Forest")
                
                n_estimators = st.slider(
                    "Número de Estimadores (árvores)",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Número de árvores na floresta. Valores menores (50-150) são recomendados para evitar overfitting no N-BaIoT."
                )
                
                max_depth = st.slider(
                    "Profundidade Máxima",
                    min_value=1,
                    max_value=50,
                    value=15,
                    step=1,
                    help="Profundidade máxima das árvores. Valores entre 10-20 são recomendados. Valores menores (10-15) reduzem overfitting."
                )
                
                if max_depth == 50:
                    max_depth = None  # Sem limite de profundidade
            
            with col2:
                st.markdown("#### Parâmetros de Divisão")
                
                min_samples_split = st.slider(
                    "Mínimo de Amostras para Split",
                    min_value=2,
                    max_value=20,
                    value=5,
                    step=1,
                    help="Número mínimo de amostras necessárias para dividir um nó interno. Valores maiores (3-5) reduzem overfitting."
                )
                
                min_samples_leaf = st.slider(
                    "Mínimo de Amostras por Folha",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1,
                    help="Número mínimo de amostras necessárias em uma folha. Valores maiores (2-4) reduzem overfitting."
                )
                
                criterion = st.selectbox(
                    "Critério de Divisão",
                    ["gini", "entropy"],
                    help="Função para medir a qualidade de uma divisão. 'gini' para impureza de Gini, 'entropy' para ganho de informação."
                )
            
            hyperparams = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'criterion': criterion
            }
            
        elif algorithm == "XGBoost":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Parâmetros do XGBoost")
                
                n_estimators = st.slider(
                    "Número de Estimadores (árvores)",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Número de árvores no modelo. Valores menores (50-150) são recomendados para evitar overfitting."
                )
                
                max_depth = st.slider(
                    "Profundidade Máxima",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1,
                    help="Profundidade máxima das árvores. Valores entre 3-8 são recomendados. Valores menores reduzem overfitting."
                )
                
                learning_rate = st.slider(
                    "Taxa de Aprendizado (Learning Rate)",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.05,
                    step=0.01,
                    help="Taxa de aprendizado. Valores menores (0.01-0.1) são mais conservadores e reduzem overfitting."
                )
            
            with col2:
                st.markdown("#### Parâmetros de Regularização")
                
                subsample = st.slider(
                    "Subsample",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.8,
                    step=0.1,
                    help="Proporção de amostras usadas para treinar cada árvore. Valores entre 0.7-0.9 reduzem overfitting."
                )
                
                colsample_bytree = st.slider(
                    "Colsample by Tree",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.8,
                    step=0.1,
                    help="Proporção de features usadas para cada árvore. Valores entre 0.7-0.9 são recomendados."
                )
                
                min_child_weight = st.slider(
                    "Mínimo Child Weight",
                    min_value=1,
                    max_value=10,
                    value=1,
                    step=1,
                    help="Peso mínimo necessário em uma folha."
                )
            
            hyperparams = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'min_child_weight': min_child_weight
            }
        
        # Estratégia de treino
        st.subheader("🧪 Estratégia de Treino")
        
        # Configuração de divisão de dados (já definida acima, apenas mostra o valor)
        st.info(f"Proporção de dados para teste: {test_size*100:.0f}%")
        
        # Validação da coluna target
        st.subheader("🎯 Validação da Coluna Target")
        
        target_col = getattr(st.session_state, 'target_column', None)
        if target_col is None:
            # Usa a função de detecção melhorada
            suitable_cols = find_suitable_target_columns(df)
            recommended = [col['column'] for col in suitable_cols if col['is_suitable'] and col['has_keyword']]
            if recommended:
                target_col = recommended[0]
            else:
                suitable = [col['column'] for col in suitable_cols if col['is_suitable']]
                target_col = suitable[0] if suitable else None
        
        # Se não encontrou uma coluna adequada, mostra aviso mas não para
        if target_col is None:
            st.error("⚠️ **Nenhuma coluna target adequada foi selecionada!**")
            st.warning("Por favor, selecione uma coluna adequada para classificação na página 'Upload & Pré-processamento'.")
            if st.button("📤 Ir para Pré-processamento", key="btn_go_preprocess"):
                st.session_state.page_redirect = "📤 Upload & Pré-processamento"
                st.rerun()
            target_col_valid = False
        elif target_col not in df.columns:
            st.error(f"⚠️ **Coluna '{target_col}' não encontrada no dataset!**")
            target_col_valid = False
        else:
            target_col_valid = True
        
        if target_col_valid and target_col and target_col in df.columns:
            unique_count = df[target_col].nunique()
            total_count = len(df[target_col].dropna())
            
            col_val1, col_val2, col_val3 = st.columns(3)
            with col_val1:
                st.metric("Coluna Target", target_col)
            with col_val2:
                st.metric("Valores Únicos", unique_count)
            with col_val3:
                percentage = (unique_count / total_count * 100) if total_count > 0 else 0
                st.metric("% Únicos", f"{percentage:.2f}%")
            
            # Validação
            if unique_count > max(50, total_count * 0.5):
                st.error("❌ **ERRO:** Esta coluna NÃO é adequada para classificação!")
                st.warning(f"""
                **Problema:** A coluna '{target_col}' tem {unique_count} valores únicos ({percentage:.1f}% dos dados).
                Isso indica uma variável contínua (regressão), não classificação.
                
                **Solução:** 
                1. Vá para a página 'Upload & Pré-processamento' e selecione uma coluna adequada
                2. Selecione uma coluna marcada como "✅ Recomendada" ou "✅ Adequada"
                3. Colunas com poucos valores únicos (idealmente < 50) são adequadas para classificação
                """)
                st.info("💡 **Dica:** O dataset N-BaIoT geralmente tem uma coluna 'label' criada automaticamente do nome do arquivo. Procure por essa coluna!")
                target_col_valid = False
            elif unique_count < 2:
                st.error("⚠️ **Atenção:** Esta coluna tem menos de 2 valores únicos. Não é possível fazer classificação.")
                target_col_valid = False
            else:
                st.success(f"✅ Coluna adequada para classificação ({unique_count} classes)")
                # Mostra uma prévia das classes
                if unique_count <= 20:
                    class_preview = df[target_col].value_counts().head(10)
                    st.info(f"**Classes encontradas:** {', '.join([str(x) for x in class_preview.index[:10]])}")
                target_col_valid = True
        else:
            target_col_valid = False
        
        # Botão de treinamento (sempre visível)
        st.markdown("---")
        st.subheader("🚀 Treinar Modelo")
        
        # Verifica se pode treinar
        can_train = target_col_valid and target_col is not None and target_col in df.columns
        
        if not can_train:
            st.warning("⚠️ Por favor, configure uma coluna target válida antes de treinar.")
            if st.button("📤 Ir para Pré-processamento", key="btn_go_preprocess2"):
                st.session_state.page_redirect = "📤 Upload & Pré-processamento"
                st.rerun()
        else:
            if st.button("🚀 Treinar Modelo", type="primary", use_container_width=True, key="btn_train_model"):
                with st.spinner("Pré-processando dados e treinando modelo... Isso pode levar alguns minutos."):
                    try:
                        # Pré-processamento
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Pré-processando dados...")
                        progress_bar.progress(20)
                        
                        # Filtra por dispositivo se o modo "treinar por dispositivo" estiver ativado
                        train_by_device = getattr(st.session_state, 'train_by_device', False)
                        selected_device = getattr(st.session_state, 'selected_device', None)
                        
                        df_to_use = df.copy()
                        if train_by_device and selected_device is not None and 'device' in df.columns:
                            df_to_use = df[df['device'] == selected_device].copy()
                            status_text.text(f"Filtrando dados do Device {selected_device}...")
                            st.info(f"📱 **Treinando modelo apenas para Device {selected_device}** ({len(df_to_use):,} amostras)")
                            progress_bar.progress(25)
                        
                        target_col = getattr(st.session_state, 'target_column', None)
                        if target_col is None:
                            # Usa a função de detecção melhorada
                            suitable_cols = find_suitable_target_columns(df_to_use)
                            recommended = [col['column'] for col in suitable_cols if col['is_suitable'] and col['has_keyword']]
                            if recommended:
                                target_col = recommended[0]
                            else:
                                suitable = [col['column'] for col in suitable_cols if col['is_suitable']]
                                target_col = suitable[0] if suitable else df_to_use.columns[-1]
                        
                        # Validação prévia antes de processar
                        if target_col and target_col in df_to_use.columns:
                            unique_count = df_to_use[target_col].nunique()
                            total_count = len(df_to_use[target_col].dropna())
                            
                            if unique_count > max(50, total_count * 0.5):
                                raise ValueError(
                                    f"A coluna '{target_col}' selecionada tem {unique_count} valores únicos ({unique_count/total_count*100:.1f}% dos dados). "
                                    f"Isso é uma variável contínua (regressão), não classificação.\n\n"
                                    f"Por favor, volte para a página 'Exploração de Dados' e selecione uma coluna adequada para classificação:\n"
                                    f"- Colunas com poucos valores únicos (idealmente < 50)\n"
                                    f"- Colunas categóricas (strings) ou inteiros discretos\n"
                                    f"- Exemplos: 'label', 'class', ou outras colunas com poucos valores distintos"
                                )
                        
                        X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_data(
                            df_to_use, target_column=target_col, test_size=test_size
                        )
                        
                        # Salva informação do dispositivo usado
                        if train_by_device and selected_device is not None:
                            st.session_state.trained_device = selected_device
                        
                        # Mostra informações sobre o pré-processamento
                        with st.expander("📊 Informações do Pré-processamento", expanded=False):
                            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                            with col_info1:
                                st.metric("Features", len(X_train.columns))
                            with col_info2:
                                st.metric("Treino", f"{len(X_train):,}")
                            with col_info3:
                                st.metric("Teste", f"{len(X_test):,}")
                            with col_info4:
                                n_classes = len(np.unique(y_train))
                                st.metric("Classes", n_classes)
                            
                            # Distribuição das classes
                            st.markdown("**Distribuição das Classes no Treino:**")
                            train_class_dist = pd.Series(y_train).value_counts().sort_index()
                            st.dataframe(train_class_dist.to_frame(name="Amostras"), width='stretch')
                            
                            # Avisos sobre possíveis problemas
                            if len(X_train) < 100:
                                st.warning("⚠️ Dataset de treino muito pequeno (< 100 amostras). Métricas podem não ser confiáveis.")
                            
                            if n_classes < 3:
                                st.info(f"ℹ️ Problema com {n_classes} classe(s). Poucas classes podem facilitar a classificação.")
                            
                            train_min = train_class_dist.min()
                            train_max = train_class_dist.max()
                            if train_min / train_max < 0.1:
                                st.warning("⚠️ Dataset muito desbalanceado! A classe menor tem menos de 10% das amostras da classe maior.")
                        
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.scaler = scaler
                        st.session_state.label_encoder = label_encoder
                        
                        algorithm_name = st.session_state.selected_algorithm
                        algorithm_display = algorithm  # Nome para exibição
                        status_text.text(f"Treinando modelo {algorithm_display}...")
                        progress_bar.progress(50)
                        
                        # Treinamento com algoritmo selecionado
                        model = train_model(algorithm_name, X_train, y_train, **hyperparams)
                        
                        # Salva informações do algoritmo usado
                        st.session_state.algorithm_display = algorithm_display
                        
                        st.session_state.model = model
                        
                        status_text.text("Avaliando modelo...")
                        progress_bar.progress(80)
                        
                        # Avaliação (inclui métricas de treino para detectar overfitting)
                        results = evaluate_model(model, X_test, y_test, label_encoder, X_train, y_train)
                        st.session_state.results = results
                        st.session_state.model_trained = True
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Modelo treinado com sucesso!")
                        
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success("✅ Modelo treinado e avaliado com sucesso!")
                        st.balloons()
                        
                        # Opção de salvar modelo
                        st.markdown("---")
                        st.subheader("💾 Salvar Modelo")
                        
                        col_save1, col_save2 = st.columns(2)
                        with col_save1:
                            device_names_save = getattr(st.session_state, 'device_names', {})
                            train_by_device_save = getattr(st.session_state, 'train_by_device', False)
                            selected_device_save = getattr(st.session_state, 'selected_device', None)
                            
                            if train_by_device_save and selected_device_save is not None:
                                device_name_save = device_names_save.get(selected_device_save, f"Device{selected_device_save}")
                                default_name = f"{algorithm_display}_{device_name_save.replace(' ', '_')}"
                            else:
                                default_name = f"{algorithm_display}_all_devices"
                            model_name = st.text_input("Nome do modelo (opcional):", value=default_name, key="model_name_input")
                        with col_save2:
                            if st.button("💾 Salvar Modelo", use_container_width=True):
                                try:
                                    import joblib
                                    import os
                                    
                                    # Cria diretório de modelos se não existir
                                    models_dir = "saved_models"
                                    os.makedirs(models_dir, exist_ok=True)
                                    
                                    # Salva o modelo
                                    model_path = os.path.join(models_dir, f"{model_name}.pkl")
                                    joblib.dump({
                                        'model': model,
                                        'scaler': scaler,
                                        'label_encoder': label_encoder,
                                        'algorithm': algorithm_display,
                                        'hyperparams': hyperparams,
                                        'device': selected_device if train_by_device and selected_device else None
                                    }, model_path)
                                    
                                    st.success(f"✅ Modelo salvo em: `{model_path}`")
                                    
                                    # Botão de download
                                    with open(model_path, 'rb') as f:
                                        st.download_button(
                                            label="📥 Download do Modelo",
                                            data=f.read(),
                                            file_name=f"{model_name}.pkl",
                                            mime="application/octet-stream"
                                        )
                                except Exception as e:
                                    st.error(f"❌ Erro ao salvar modelo: {str(e)}")
                    
                    except ValueError as e:
                        error_msg = str(e)
                        if "least populated class" in error_msg or "minimum number of groups" in error_msg:
                            st.error("❌ **Erro: Classes com poucas amostras**")
                            st.warning("""
                            O dataset tem classes com menos de 2 amostras, o que impede a divisão estratificada.
                            
                            **Soluções:**
                            - Tente aumentar o número de arquivos carregados
                            - Aumente o tamanho da amostra por arquivo
                            - O código tentará automaticamente usar divisão sem estratificação
                            """)
                            st.info("💡 Tente novamente com mais dados ou verifique a distribuição das classes na página 'Upload & Pré-processamento'")
                        elif "regressão" in error_msg.lower() or "regression" in error_msg.lower() or "valores únicos" in error_msg.lower():
                            st.error("❌ **Erro: Variável Target Incorreta**")
                            st.warning("""
                            A coluna selecionada como target parece ser uma variável contínua (regressão), 
                            mas estamos usando um modelo de classificação que requer valores categóricos.
                            
                            **O que fazer:**
                            - Vá para a página 'Upload & Pré-processamento'
                            - Verifique a distribuição da coluna target
                            - Selecione uma coluna com valores categóricos (poucos valores únicos)
                            - Exemplos: 'label', 'class', 'attack', 'type', etc.
                            """)
                            st.info(f"💡 **Detalhes:** {error_msg}")
                        elif "apenas" in error_msg.lower() and "classe" in error_msg.lower():
                            st.error("❌ **Erro: Poucas Classes**")
                            st.warning("""
                            Após o pré-processamento, restaram menos de 2 classes no dataset.
                            
                            **Soluções:**
                            - Aumente o número de arquivos carregados
                            - Verifique se a coluna target está correta
                            - Aumente o tamanho da amostra
                            """)
                        else:
                            st.error(f"❌ Erro ao treinar modelo: {error_msg}")
                        st.exception(e)
                    except Exception as e:
                        st.error(f"❌ Erro ao treinar modelo: {str(e)}")
                        st.exception(e)
        
        # Mostra informações do modelo se já foi treinado
        if st.session_state.model_trained:
            st.subheader("10. Informações do Modelo Treinado")
            
            model = st.session_state.model
            algorithm_display = getattr(st.session_state, 'algorithm_display', 'Modelo')
            
            st.success(f"✅ **Modelo {algorithm_display} treinado com sucesso!**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Algoritmo", algorithm_display)
                if hasattr(model, 'n_estimators'):
                    st.metric("Número de Estimadores", model.n_estimators)
                if hasattr(model, 'max_depth'):
                    st.metric("Profundidade Máxima", str(model.max_depth) if model.max_depth else "Sem limite")
            
            with col2:
                if hasattr(model, 'criterion'):
                    st.metric("Critério", model.criterion)
                if hasattr(model, 'learning_rate'):
                    st.metric("Learning Rate", model.learning_rate)
            
            st.info("💡 Vá para a página 'Resultados' para ver métricas detalhadas e visualizações.")

# Página: Resultados
elif page == "📈 Resultados":
    st.header("📈 Resultados e Avaliação")
    
    if not st.session_state.model_trained or st.session_state.results is None:
        st.warning("⚠️ Por favor, treine o modelo primeiro na página 'Dados e Treinamento'")
    else:
        results = st.session_state.results
        algorithm_display = getattr(st.session_state, 'algorithm_display', 'Modelo')
        
        # Informações do modelo
        st.subheader("1. Informações do Modelo")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Algoritmo Utilizado", algorithm_display)
            # Mostra dispositivo se foi treinado por dispositivo
            trained_device = getattr(st.session_state, 'trained_device', None)
            device_names = getattr(st.session_state, 'device_names', {})
            train_by_device = getattr(st.session_state, 'train_by_device', False)
            
            if train_by_device and trained_device is not None:
                device_name = device_names.get(trained_device, f"Device {trained_device}")
                st.info(f"📱 Modelo treinado para **{device_name} (Device {trained_device})** (modelo por dispositivo)")
            elif not train_by_device:
                st.info(f"📱 Modelo treinado com **todos os dispositivos combinados**")
        with col_info2:
            model = st.session_state.model
            if hasattr(model, 'n_estimators'):
                st.metric("Número de Estimadores", model.n_estimators)
        
        # Métricas principais
        st.subheader("2. Métricas de Desempenho")
        
        # Verifica se há métricas de treino para comparação
        has_train_metrics = results.get('train_metrics') is not None
        
        if has_train_metrics:
            train_metrics = results['train_metrics']
            
            st.markdown("### 📊 Comparação: Treino vs Teste")
            st.markdown("**A diferença entre treino e teste indica overfitting:**")
            st.markdown("- **Diferença < 2%**: Modelo generaliza bem ✅")
            st.markdown("- **Diferença 2-5%**: Leve overfitting ⚠️")
            st.markdown("- **Diferença > 5%**: Overfitting significativo ❌")
            
            # Calcula diferenças
            acc_diff = train_metrics['accuracy'] - results['accuracy']
            f1_diff = train_metrics['f1_score'] - results['f1_score']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                delta_acc = f"{acc_diff:.4f}"
                delta_color = "normal" if abs(acc_diff) < 0.02 else "inverse" if acc_diff > 0.05 else "off"
                st.metric("Acurácia (Teste)", f"{results['accuracy']:.4f}", 
                         delta=f"Treino: {train_metrics['accuracy']:.4f} ({delta_acc})",
                         delta_color=delta_color)
            with col2:
                delta_prec = train_metrics['precision'] - results['precision']
                st.metric("Precisão (Teste)", f"{results['precision']:.4f}",
                         delta=f"Treino: {train_metrics['precision']:.4f} ({delta_prec:+.4f})")
            with col3:
                delta_rec = train_metrics['recall'] - results['recall']
                st.metric("Recall (Teste)", f"{results['recall']:.4f}",
                         delta=f"Treino: {train_metrics['recall']:.4f} ({delta_rec:+.4f})")
            with col4:
                delta_color_f1 = "normal" if abs(f1_diff) < 0.02 else "inverse" if f1_diff > 0.05 else "off"
                st.metric("F1-Score (Teste)", f"{results['f1_score']:.4f}",
                         delta=f"Treino: {train_metrics['f1_score']:.4f} ({f1_diff:+.4f})",
                         delta_color=delta_color_f1)
            
            # Aviso sobre overfitting
            if abs(acc_diff) > 0.05:
                st.error(f"❌ **Overfitting Detectado!**")
                st.warning(f"""
                **Diferença de acurácia entre treino e teste: {abs(acc_diff)*100:.2f}%**
                
                O modelo está performando muito melhor no treino do que no teste, indicando overfitting.
                
                **Soluções:**
                - Reduza ainda mais a complexidade do modelo (menos árvores, menor profundidade)
                - Aumente `min_samples_split` e `min_samples_leaf` (Random Forest)
                - Aumente regularização (subsample, colsample_bytree para XGBoost)
                - Reduza `learning_rate` e aumente `n_estimators` (XGBoost)
                """)
            elif abs(acc_diff) > 0.02:
                st.warning(f"⚠️ **Leve Overfitting Detectado**")
                st.info(f"Diferença de acurácia: {abs(acc_diff)*100:.2f}%. Considere reduzir um pouco a complexidade do modelo.")
            else:
                st.success("✅ **Modelo generaliza bem!** Diferença entre treino e teste é pequena.")
        else:
            # Se não houver métricas de treino, mostra apenas teste
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Acurácia", f"{results['accuracy']:.4f}", delta=None)
            with col2:
                st.metric("Precisão", f"{results['precision']:.4f}", delta=None)
            with col3:
                st.metric("Recall", f"{results['recall']:.4f}", delta=None)
            with col4:
                st.metric("F1-Score", f"{results['f1_score']:.4f}", delta=None)
        
        # Explicação sobre 93%+ de acurácia
        if results['accuracy'] >= 0.93 and results['accuracy'] < 0.99:
            st.info("💡 **Sobre acurácia de 93%+:**")
            st.markdown("""
            Uma acurácia de 93%+ **não necessariamente indica overfitting**. Depende da diferença entre treino e teste:
            
            - **Se a diferença for pequena (< 2%)**: O modelo está generalizando bem! ✅
            - **Se a diferença for grande (> 5%)**: Há overfitting, mesmo com 93% no teste ❌
            
            Para o dataset N-BaIoT (detecção de botnet), 93-97% de acurácia é **razoável e esperado** 
            quando o modelo está bem ajustado, pois os padrões de tráfego normal vs ataque são relativamente distintos.
            """)
        
        # Aviso sobre métricas muito altas (independente de overfitting)
        if results['accuracy'] >= 0.99:
            st.warning("⚠️ **Atenção: Métricas muito altas (≥99%)**")
            st.markdown("""
            Métricas perfeitas ou quase perfeitas podem indicar:
            
            **1. Data Leakage (Vazamento de Dados)**
            - Alguma feature pode conter informação direta sobre a classe target
            - Verifique se há colunas derivadas da target nas features
            
            **2. Problema Muito Simples**
            - O dataset pode ser trivialmente separável
            - Verifique a distribuição das classes e a complexidade do problema
            
            **3. Overfitting Extremo**
            - O modelo pode estar memorizando os dados de treino
            - Tente reduzir a complexidade do modelo (menos árvores, menor profundidade)
            
            **4. Dataset Muito Pequeno ou Desbalanceado**
            - Poucos dados podem levar a resultados enganosos
            - Verifique o tamanho do dataset e a distribuição das classes
            """)
            
            # Diagnósticos adicionais
            with st.expander("🔍 Ver Diagnósticos Detalhados", expanded=False):
                if st.session_state.X_train is not None and st.session_state.y_train is not None:
                    X_train = st.session_state.X_train
                    y_train = st.session_state.y_train
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test
                    
                    st.markdown("### 📊 Informações do Dataset")
                    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                    with col_d1:
                        st.metric("Treino (amostras)", f"{len(X_train):,}")
                    with col_d2:
                        st.metric("Teste (amostras)", f"{len(X_test):,}")
                    with col_d3:
                        st.metric("Features", len(X_train.columns))
                    with col_d4:
                        n_classes = len(np.unique(y_train))
                        st.metric("Classes", n_classes)
                    
                    st.markdown("### 📈 Distribuição das Classes")
                    col_dist1, col_dist2 = st.columns(2)
                    
                    with col_dist1:
                        st.markdown("**Treino:**")
                        train_dist = pd.Series(y_train).value_counts().sort_index()
                        st.dataframe(train_dist.to_frame(name="Amostras"), width='stretch')
                        
                    with col_dist2:
                        st.markdown("**Teste:**")
                        test_dist = pd.Series(y_test).value_counts().sort_index()
                        st.dataframe(test_dist.to_frame(name="Amostras"), width='stretch')
                    
                    # Verifica se há classes com muito poucas amostras
                    train_min = train_dist.min()
                    test_min = test_dist.min()
                    if train_min < 5 or test_min < 5:
                        st.warning(f"⚠️ Algumas classes têm muito poucas amostras (mínimo: treino={train_min}, teste={test_min}). Isso pode causar métricas enganosas.")
                    
                    # Verifica correlação entre features e target (possível data leakage)
                    st.markdown("### 🔍 Verificação de Data Leakage")
                    try:
                        # Calcula correlação entre features numéricas e target
                        if len(X_train.columns) > 0:
                            # Cria um DataFrame temporário para análise
                            temp_df = X_train.copy()
                            temp_df['target'] = y_train
                            
                            # Calcula correlações
                            correlations = temp_df.corr()['target'].drop('target').abs().sort_values(ascending=False)
                            
                            high_corr = correlations[correlations > 0.9]
                            if len(high_corr) > 0:
                                st.error(f"🚨 **Possível Data Leakage Detectado!**")
                                st.warning(f"Encontradas {len(high_corr)} feature(s) com correlação > 0.9 com a target:")
                                st.dataframe(high_corr.to_frame(name="Correlação"), width='stretch')
                                st.info("💡 Features com correlação muito alta podem estar vazando informação sobre a classe target. Considere removê-las.")
                            else:
                                st.success("✅ Nenhuma feature com correlação suspeitamente alta (>0.9) encontrada.")
                            
                            # Mostra top 10 correlações
                            st.markdown("**Top 10 Features com Maior Correlação (absoluta) com Target:**")
                            top_corr = correlations.head(10)
                            st.dataframe(top_corr.to_frame(name="Correlação"), width='stretch')
                    except Exception as e:
                        st.info(f"ℹ️ Não foi possível calcular correlações: {str(e)}")
                    
                    # Verifica se o problema é muito simples (classes muito separadas)
                    st.markdown("### 🎯 Análise de Separabilidade")
                    if n_classes == 2:
                        st.info("ℹ️ Problema binário (2 classes). Verifique se as classes são facilmente separáveis.")
                    elif n_classes < 5:
                        st.info(f"ℹ️ Problema com {n_classes} classes. Poucas classes podem facilitar a classificação.")
                    else:
                        st.info(f"ℹ️ Problema multiclasse com {n_classes} classes.")
                    
                    # Verifica balanceamento
                    train_balance = train_dist.min() / train_dist.max()
                    if train_balance < 0.1:
                        st.warning("⚠️ Dataset muito desbalanceado! A classe menor tem menos de 10% das amostras da classe maior.")
                        st.info("💡 Considere usar técnicas de balanceamento (SMOTE, undersampling, etc.) ou métricas adequadas para dados desbalanceados.")
        
        # Matriz de confusão
        st.subheader("3. Matriz de Confusão")
        
        cm = results['confusion_matrix']
        
        # Cria visualização da matriz de confusão
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig_cm.update_layout(
            title="Matriz de Confusão",
            xaxis_title="Predição",
            yaxis_title="Valor Real",
            width=700,
            height=600
        )
        
        st.plotly_chart(fig_cm, width='stretch')
        
        # Relatório de classificação
        st.subheader("4. Relatório de Classificação Detalhado")
        
        report_df = pd.DataFrame(results['classification_report']).transpose()
        # Converte valores numéricos para float64 explícito
        for col in report_df.select_dtypes(include=[np.number]).columns:
            report_df[col] = report_df[col].astype('float64')
        st.dataframe(report_df, width='stretch')
        
        # Feature Importance
        st.subheader("5. Importância das Features (Top 20)")
        
        if st.session_state.model is not None and st.session_state.X_train is not None:
            feature_names = st.session_state.X_train.columns.tolist()
            importance_df = get_feature_importance(st.session_state.model, feature_names, top_n=20)
            
            if importance_df is not None:
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 20 Features Mais Importantes",
                    labels={'Importance': 'Importância', 'Feature': 'Feature'}
                )
                fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_importance, width='stretch')
                
                # Garante tipos compatíveis
                importance_df['Importance'] = importance_df['Importance'].astype('float64')
                st.dataframe(importance_df, width='stretch')
            else:
                st.info("ℹ️ Este algoritmo não fornece feature importance direta.")
        
        # Distribuição de predições
        st.subheader("6. Distribuição de Predições")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição das classes reais
            y_test = results['y_test']
            unique, counts = np.unique(y_test, return_counts=True)
            fig_real = px.pie(
                values=counts,
                names=[f"Classe {u}" for u in unique],
                title="Distribuição das Classes Reais (Teste)"
            )
            st.plotly_chart(fig_real, width='stretch')
        
        with col2:
            # Distribuição das predições
            y_pred = results['y_pred']
            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            fig_pred = px.pie(
                values=counts_pred,
                names=[f"Classe {u}" for u in unique_pred],
                title="Distribuição das Predições"
            )
            st.plotly_chart(fig_pred, width='stretch')
        
        # Comparação Real vs Predito
        st.subheader("6. Comparação: Real vs Predito")
        
        comparison_df = pd.DataFrame({
            'Real': y_test,
            'Predito': y_pred
        })
        
        # Contagem de acertos e erros
        comparison_df['Acerto'] = comparison_df['Real'] == comparison_df['Predito']
        
        accuracy_by_class = comparison_df.groupby('Real').agg({
            'Acerto': 'mean'
        }).reset_index()
        accuracy_by_class.columns = ['Classe', 'Taxa de Acerto']
        
        fig_accuracy = px.bar(
            accuracy_by_class,
            x='Classe',
            y='Taxa de Acerto',
            title="Taxa de Acerto por Classe",
            labels={'Classe': 'Classe', 'Taxa de Acerto': 'Taxa de Acerto'}
        )
        st.plotly_chart(fig_accuracy, width='stretch')

# Rodapé
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Projeto de Mestrado - Aprendizado de Máquina | "
    "Dataset: N-BaIoT | Algoritmo: Random Forest"
    "</div>",
    unsafe_allow_html=True
)

