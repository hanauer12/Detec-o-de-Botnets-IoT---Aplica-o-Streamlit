# Detecção de Botnets IoT - Aplicação Streamlit

Aplicação interativa em Streamlit para detecção de ataques de botnet em dispositivos IoT usando o dataset N-BaIoT.

## 🚀 Instalação e Execução

### Pré-requisitos
- Python 3.10+

### Passos

1. **Clone o repositório:**
```bash
git clone https://github.com/hanauer12/Detec-o-de-Botnets-IoT---Aplica-o-Streamlit.git
```

2. **Crie e ative o ambiente virtual:**
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# ou
venv\Scripts\activate  # Windows
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

**Nota:** Se houver erro ao instalar `pyarrow` (especialmente no Python 3.14), use:
```bash
pip install --only-binary :all: -r requirements.txt
```

4. **Execute a aplicação:**
```bash
streamlit run app.py
```

A aplicação abrirá automaticamente em `http://localhost:8501`

## 📊 Dataset

**N-BaIoT Dataset** (`mkashifn/nbaiot-dataset` no Kaggle)
- Dados de tráfego de rede de dispositivos IoT
- Classes: tráfego benigno e diferentes tipos de ataques de botnet

## 🔧 Funcionalidades

- Download automático do dataset via Kaggle Hub
- Exploração e visualização de dados
- Pré-processamento automático
- Treinamento de modelos (Random Forest e XGBoost)
- Ajuste interativo de hiperparâmetros
- Visualização de métricas e resultados

## 📁 Estrutura

```
Mestrado/
├── app.py              # Aplicação principal
├── utils.py            # Funções auxiliares
├── requirements.txt    # Dependências
└── README.md           # Este arquivo
```

## 🧪 Algoritmos

- **Random Forest**: Ajuste de n_estimators, max_depth, min_samples_split, criterion
- **XGBoost**: Ajuste de learning_rate, max_depth, subsample, colsample_bytree

