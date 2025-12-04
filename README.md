# Detecção de Botnets IoT - Aplicação Streamlit

## 📋 Descrição

Aplicação interativa desenvolvida em Streamlit para detecção de ataques de botnet em dispositivos IoT utilizando o dataset N-BaIoT. O projeto implementa algoritmos de aprendizado de máquina com interface amigável para ajuste de hiperparâmetros e visualização de resultados.

## 🎯 Objetivo

Desenvolver uma aplicação prática que permita:
- Carregar e explorar o dataset N-BaIoT
- Treinar modelos de classificação para detecção de botnets
- Ajustar hiperparâmetros de forma interativa
- Visualizar métricas de desempenho e resultados

## 🚀 Como Executar

### Pré-requisitos

1. Python 3.8 ou superior
2. Conta Kaggle configurada (para download do dataset)

### Instalação

1. Clone o repositório:
```bash
git clone <seu-repositorio>
cd Mestrado
```

2. Crie e ative um ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # No macOS/Linux
# ou
venv\Scripts\activate  # No Windows
```

3. Instale as dependências:
```bash
# Primeiro, instale cmake e apache-arrow (necessários para pyarrow)
brew install cmake apache-arrow  # macOS
# ou use o gerenciador de pacotes do seu sistema Linux

# Depois instale pyarrow
pip install pyarrow

# Por fim, instale as outras dependências
pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly kagglehub joblib
pip install "altair<6"  # Compatibilidade com streamlit
```

3. Configure as credenciais do Kaggle:
   - Acesse https://www.kaggle.com/ → Account → API → Create New Token
   - Coloque o arquivo `kaggle.json` em `~/.kaggle/kaggle.json`
   - Ou defina as variáveis de ambiente:
   ```bash
   export KAGGLE_USERNAME=seu_usuario
   export KAGGLE_KEY=sua_chave_api
   ```
   - Veja mais detalhes em `kaggle_setup.md`

4. Execute a aplicação:
```bash
streamlit run app.py
```

A aplicação será aberta automaticamente no navegador em `http://localhost:8501`

## 📊 Dataset

O projeto utiliza o **N-BaIoT Dataset to Detect IoT Botnet Attacks**, disponível no Kaggle:
- Dataset: `mkashifn/nbaiot-dataset`
- Descrição: Dataset contendo dados de tráfego de rede de dispositivos IoT para detecção de ataques de botnet

## 🔧 Funcionalidades

- **Carregamento de Dados**: Download e carregamento automático do dataset via Kaggle Hub
- **Exploração de Dados**: Visualização estatística e distribuição das classes
- **Pré-processamento**: Normalização e preparação dos dados
- **Treinamento de Modelos**: Implementação de Random Forest com ajuste de hiperparâmetros
- **Avaliação**: Métricas de desempenho, matriz de confusão e curvas de aprendizado
- **Visualizações**: Gráficos interativos para análise dos resultados

## 📁 Estrutura do Projeto

```
Mestrado/
├── app.py                 # Aplicação principal Streamlit
├── utils.py               # Funções auxiliares
├── requirements.txt       # Dependências do projeto
├── README.md              # Documentação
└── .gitignore            # Arquivos ignorados pelo Git
```

## 🧪 Algoritmos Implementados

- **Random Forest**: Ensemble de árvores de decisão com ajuste de:
  - Número de estimadores
  - Profundidade máxima
  - Número mínimo de amostras para split
  - Critério de divisão

## 📝 Entregáveis

- ✅ Código-fonte completo
- ✅ Interface interativa Streamlit
- ✅ Documentação do projeto
- ✅ README com instruções de uso

## 👤 Autor

Projeto desenvolvido para disciplina de Aprendizado de Máquina - Mestrado

## 📄 Licença

Este projeto é para fins educacionais.
