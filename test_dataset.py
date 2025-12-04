"""
Script de teste para verificar o carregamento do dataset N-BaIoT
Execute este script antes de rodar a aplicação Streamlit para garantir que tudo está configurado corretamente.
"""

import sys

def test_kaggle_setup():
    """Testa se o Kaggle está configurado corretamente"""
    print("🔍 Verificando configuração do Kaggle...")
    try:
        import kagglehub
        print("✅ kagglehub instalado")
    except ImportError:
        print("❌ kagglehub não está instalado. Execute: pip install kagglehub")
        return False
    
    # Tenta fazer um download de teste
    try:
        print("📥 Tentando baixar dataset (pode levar alguns minutos)...")
        path = kagglehub.dataset_download("mkashifn/nbaiot-dataset")
        print(f"✅ Dataset baixado com sucesso em: {path}")
        return True
    except Exception as e:
        print(f"❌ Erro ao baixar dataset: {str(e)}")
        print("\n💡 Dicas:")
        print("   - Verifique se suas credenciais do Kaggle estão configuradas")
        print("   - Veja kaggle_setup.md para instruções detalhadas")
        return False

def test_utils():
    """Testa as funções utilitárias"""
    print("\n🔍 Testando funções utilitárias...")
    try:
        from utils import load_dataset, preprocess_data
        print("✅ Módulo utils importado com sucesso")
        
        # Testa carregamento (com amostra pequena)
        print("📊 Testando carregamento do dataset (amostra pequena)...")
        df, path = load_dataset(max_files=1, sample_size=1000)
        print(f"✅ Dataset carregado: {len(df)} linhas, {len(df.columns)} colunas")
        
        # Testa pré-processamento
        print("🔧 Testando pré-processamento...")
        X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_data(df, test_size=0.2)
        print(f"✅ Pré-processamento concluído:")
        print(f"   - Treino: {len(X_train)} amostras")
        print(f"   - Teste: {len(X_test)} amostras")
        print(f"   - Features: {len(X_train.columns)}")
        
        return True
    except Exception as e:
        print(f"❌ Erro ao testar utils: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """Testa se todas as dependências estão instaladas"""
    print("🔍 Verificando dependências...")
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'plotly',
        'kagglehub',
        'joblib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} não instalado")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Pacotes faltando: {', '.join(missing)}")
        print("Execute: pip install -r requirements.txt")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTE DE CONFIGURAÇÃO - Detecção de Botnets IoT")
    print("=" * 60)
    
    all_ok = True
    
    # Testa dependências
    if not test_dependencies():
        all_ok = False
        sys.exit(1)
    
    # Testa configuração do Kaggle
    if not test_kaggle_setup():
        all_ok = False
        sys.exit(1)
    
    # Testa funções utilitárias
    if not test_utils():
        all_ok = False
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES PASSARAM!")
    print("🚀 Você pode executar a aplicação com: streamlit run app.py")
    print("=" * 60)





