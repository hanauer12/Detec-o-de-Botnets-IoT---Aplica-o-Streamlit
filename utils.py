"""
Funções auxiliares para carregamento, pré-processamento e treinamento de modelos
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import os


def count_csv_files(dataset_path):
    """
    Conta quantos arquivos CSV existem no dataset
    
    Args:
        dataset_path: Caminho do dataset
    
    Returns:
        Número de arquivos CSV encontrados
    """
    if not dataset_path or not os.path.exists(dataset_path):
        return None
    
    csv_count = 0
    for root, dirs, files in os.walk(dataset_path):
        csv_count += sum(1 for f in files if f.endswith('.csv') and f.lower() != 'device_info.csv')
    
    return csv_count


def find_suitable_target_columns(df, max_unique_values=50):
    """
    Encontra colunas adequadas para classificação (target)
    
    Args:
        df: DataFrame com os dados
        max_unique_values: Número máximo de valores únicos para considerar adequado
    
    Returns:
        Lista de tuplas (nome_coluna, num_valores_unicos, adequada)
    """
    suitable_columns = []
    
    for col in df.columns:
        unique_count = df[col].nunique()
        total_count = len(df[col].dropna())
        
        # Verifica se é adequada para classificação
        is_suitable = (
            unique_count <= max_unique_values and 
            unique_count >= 2 and
            unique_count <= total_count * 0.5  # Não mais de 50% dos valores são únicos
        )
        
        # Prioriza colunas com nomes que sugerem classe/label
        priority_keywords = ['label', 'class', 'target', 'attack', 'type', 'category']
        has_keyword = any(keyword in col.lower() for keyword in priority_keywords)
        
        suitable_columns.append({
            'column': col,
            'unique_count': unique_count,
            'total_count': total_count,
            'is_suitable': is_suitable,
            'has_keyword': has_keyword,
            'percentage': (unique_count / total_count * 100) if total_count > 0 else 0
        })
    
    # Ordena: primeiro adequadas com keywords, depois adequadas, depois outras
    suitable_columns.sort(key=lambda x: (
        not x['is_suitable'],  # Adequadas primeiro
        not x['has_keyword'],  # Com keywords primeiro
        x['unique_count']  # Menos valores únicos primeiro
    ))
    
    return suitable_columns


def get_default_device_names():
    """
    Retorna nomes padrão dos dispositivos baseados no paper N-BaIoT
    """
    return {
        1: "Danmini Doorbell",
        2: "Ecobee Thermostat",
        3: "Ennio Doorbell",
        4: "Philips B120N/10 Baby Monitor",
        5: "Provision PT-737E Security Camera",
        6: "Provision PT-838 Security Camera",
        7: "Samsung SNH1011 N Webcam",
        8: "SimpleHome XCS7-1002-WHT Security Camera",
        9: "SimpleHome XCS7-1003-WHT Security Camera"
    }


def load_device_names(dataset_path):
    """
    Carrega os nomes dos dispositivos do arquivo device_info.csv
    
    Args:
        dataset_path: Caminho do dataset
    
    Returns:
        Dicionário mapeando número do dispositivo para nome
    """
    device_names = {}
    
    # Procura pelo arquivo device_info.csv
    device_info_path = None
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower() == 'device_info.csv':
                device_info_path = os.path.join(root, file)
                break
        if device_info_path:
            break
    
    if device_info_path and os.path.exists(device_info_path):
        try:
            device_info_df = pd.read_csv(device_info_path)
            
            # Tenta diferentes formatos do arquivo
            # Formato 1: Colunas nomeadas (ex: "device", "name")
            if 'device' in device_info_df.columns and 'name' in device_info_df.columns:
                for _, row in device_info_df.iterrows():
                    try:
                        device_num = int(row['device'])
                        device_name = str(row['name']).strip()
                        device_names[device_num] = device_name
                    except (ValueError, TypeError, KeyError):
                        continue
            # Formato 2: Primeira coluna = número, segunda = nome
            elif len(device_info_df.columns) >= 2:
                for _, row in device_info_df.iterrows():
                    try:
                        device_num = int(row.iloc[0])
                        device_name = str(row.iloc[1]).strip()
                        device_names[device_num] = device_name
                    except (ValueError, TypeError, IndexError):
                        continue
            # Formato 3: Índice = número, primeira coluna = nome
            else:
                for idx, row in device_info_df.iterrows():
                    try:
                        device_num = int(idx) if isinstance(idx, (int, float)) else int(str(idx))
                        device_name = str(row.iloc[0]).strip()
                        device_names[device_num] = device_name
                    except (ValueError, TypeError, IndexError):
                        continue
            
            if device_names:
                print(f"✅ Carregados {len(device_names)} nomes de dispositivos do device_info.csv")
            else:
                print(f"⚠️ device_info.csv encontrado mas nenhum nome foi extraído. Usando nomes padrão.")
                device_names = get_default_device_names()
        except Exception as e:
            print(f"⚠️ Erro ao carregar device_info.csv: {str(e)}")
            device_names = get_default_device_names()
    else:
        device_names = get_default_device_names()
        print(f"ℹ️ Usando nomes padrão dos dispositivos (device_info.csv não encontrado)")
    
    return device_names


def get_available_devices(dataset_path):
    """
    Retorna lista de dispositivos disponíveis no dataset
    
    Args:
        dataset_path: Caminho do dataset
    
    Returns:
        Lista de números de dispositivos disponíveis e dicionário de nomes
    """
    if not dataset_path or not os.path.exists(dataset_path):
        return [], {}
    
    # Carrega nomes dos dispositivos
    device_names = load_device_names(dataset_path)
    
    # Lista arquivos CSV
    csv_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv') and file.lower() != 'device_info.csv':
                csv_files.append(os.path.join(root, file))
    
    # Extrai números dos dispositivos
    devices = set()
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        filename_no_ext = filename.replace('.csv', '').strip()
        parts = filename_no_ext.split('.')
        
        if len(parts) >= 2:
            try:
                device_num = int(parts[0])
                devices.add(device_num)
            except ValueError:
                continue
    
    return sorted(list(devices)), device_names


def load_dataset(devices_to_load=None, sample_size=None, dataset_path=None):
    """
    Carrega o dataset N-BaIoT do Kaggle
    
    Args:
        devices_to_load: Lista de números de dispositivos para carregar (ex: [1, 2, 3]). 
                        Se None, carrega todos os dispositivos disponíveis.
        sample_size: Se fornecido, amostra aleatória de linhas por arquivo (útil para datasets grandes)
        dataset_path: Caminho do dataset já baixado (opcional). Se None, faz o download.
    
    Returns:
        DataFrame combinado, caminho do dataset e dicionário de nomes dos dispositivos
    """
    try:
        import kagglehub
        
        # Se o caminho não foi fornecido, faz o download
        if dataset_path is None:
            # Faz o download do dataset
            path = kagglehub.dataset_download("mkashifn/nbaiot-dataset")
        else:
            path = dataset_path
        
        # Carrega nomes dos dispositivos primeiro
        device_names = load_device_names(path)
        
        # Lista arquivos CSV no diretório baixado
        all_csv_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv') and file.lower() != 'device_info.csv':
                    all_csv_files.append(os.path.join(root, file))
        
        if not all_csv_files:
            raise FileNotFoundError("Nenhum arquivo CSV encontrado no dataset")
        
        # Ordena os arquivos por nome para garantir ordem consistente
        all_csv_files.sort()
        
        # Agrupa arquivos por dispositivo
        files_by_device = {}
        for csv_file in all_csv_files:
            filename = os.path.basename(csv_file)
            filename_no_ext = filename.replace('.csv', '').strip()
            parts = filename_no_ext.split('.')
            
            if len(parts) >= 2:
                try:
                    device_num = int(parts[0])
                    if device_num not in files_by_device:
                        files_by_device[device_num] = []
                    files_by_device[device_num].append(csv_file)
                except ValueError:
                    continue
        
        # Determina quais dispositivos carregar
        available_devices = sorted(files_by_device.keys())
        print(f"📱 Dispositivos disponíveis no dataset: {available_devices}")
        
        if devices_to_load is None:
            # Se não especificado, carrega todos
            devices_to_load = available_devices
            print(f"Carregando todos os {len(devices_to_load)} dispositivos disponíveis")
        else:
            # Filtra apenas dispositivos disponíveis
            devices_to_load = [d for d in devices_to_load if d in available_devices]
            if not devices_to_load:
                raise ValueError(f"Nenhum dos dispositivos solicitados está disponível. Dispositivos disponíveis: {available_devices}")
            print(f"Carregando dispositivos: {devices_to_load}")
        
        # Coleta todos os arquivos dos dispositivos selecionados
        csv_files = []
        for device_num in devices_to_load:
            device_files = files_by_device[device_num]
            csv_files.extend(device_files)
            device_name = device_names.get(device_num, f"Device {device_num}")
            print(f"  - {device_name} (Device {device_num}): {len(device_files)} arquivos CSV")
        
        print(f"Total de arquivos CSV a carregar: {len(csv_files)}")
        
        # Carrega e combina os arquivos CSV
        dataframes = []
        for csv_file in csv_files:
            try:
                if sample_size:
                    # Carrega uma amostra do arquivo
                    df_chunk = pd.read_csv(csv_file, nrows=sample_size)
                else:
                    # Tenta carregar o arquivo completo
                    # Para arquivos muito grandes, pode ser necessário usar chunks
                    try:
                        df_chunk = pd.read_csv(csv_file)
                    except MemoryError:
                        # Se der erro de memória, carrega apenas uma amostra
                        df_chunk = pd.read_csv(csv_file, nrows=100000)
                        print(f"Aviso: Arquivo {csv_file} muito grande. Carregando apenas 100k linhas.")
                
                # Extrai o label do nome do arquivo (N-BaIoT usa nomes como "1.benign.csv", "1.mirai.scan.csv", "1.gafgyt.combo.csv", etc.)
                filename = os.path.basename(csv_file)
                # Remove extensão .csv
                filename_no_ext = filename.replace('.csv', '').strip()
                
                # N-BaIoT usa formato: "número.tipo.csv" ou "número.família.tipo.csv"
                # Exemplos: "1.benign", "1.mirai.scan", "1.gafgyt.udp"
                parts = filename_no_ext.split('.')
                
                # Extrai o número do dispositivo (primeira parte)
                device_number = None
                if len(parts) >= 2:
                    try:
                        device_number = int(parts[0])
                    except ValueError:
                        device_number = None
                    
                    # Remove o número inicial (primeira parte)
                    # Se tiver 2 partes: "1.benign" -> "benign"
                    # Se tiver 3 partes: "1.mirai.scan" -> "mirai.scan"
                    label = '.'.join(parts[1:])
                else:
                    label = filename_no_ext
                
                # Normaliza o label (remove espaços, converte para minúsculas)
                label = label.strip().lower()
                
                # Adiciona coluna de label (sempre sobrescreve para garantir consistência)
                df_chunk['label'] = label
                
                # Adiciona coluna de dispositivo
                if device_number is not None:
                    df_chunk['device'] = device_number
                    print(f"✅ Arquivo '{filename}' -> Device: {device_number}, Label: '{label}' ({len(df_chunk)} linhas)")
                else:
                    df_chunk['device'] = 0  # Dispositivo desconhecido
                    print(f"✅ Arquivo '{filename}' -> Device: desconhecido, Label: '{label}' ({len(df_chunk)} linhas)")
                
                dataframes.append(df_chunk)
                print(f"Carregado: {os.path.basename(csv_file)} ({len(df_chunk)} linhas, label: {label})")
            except Exception as e:
                print(f"Erro ao carregar {csv_file}: {str(e)}")
                continue
        
        if not dataframes:
            raise ValueError("Nenhum arquivo CSV foi carregado com sucesso")
        
        # Combina todos os DataFrames
        df = pd.concat(dataframes, ignore_index=True)
        
        # Mostra informações sobre os labels e dispositivos encontrados
        if 'label' in df.columns:
            unique_labels = df['label'].unique()
            label_counts = df['label'].value_counts()
            print(f"\n📊 Labels encontrados no dataset:")
            for label, count in label_counts.items():
                print(f"  - '{label}': {count:,} amostras ({count/len(df)*100:.1f}%)")
            print(f"Total: {len(df):,} amostras, {len(unique_labels)} classes distintas")
        
        if 'device' in df.columns:
            unique_devices = sorted(df['device'].unique())
            device_counts = df['device'].value_counts().sort_index()
            print(f"\n📱 Dispositivos encontrados no dataset:")
            for device, count in device_counts.items():
                device_name = device_names.get(device, f"Device {device}")
                print(f"  - {device_name} (Device {device}): {count:,} amostras ({count/len(df)*100:.1f}%)")
            print(f"Total: {len(unique_devices)} dispositivos distintos\n")
        
        return df, path, device_names
    
    except Exception as e:
        raise Exception(f"Erro ao carregar dataset: {str(e)}")


def preprocess_data(df, target_column=None, test_size=0.2, random_state=42, binary_classification=False):
    """
    Pré-processa os dados para treinamento
    
    Args:
        df: DataFrame com os dados
        target_column: Nome da coluna alvo (se None, tenta detectar automaticamente)
        test_size: Proporção dos dados para teste
        random_state: Seed para reprodutibilidade
        binary_classification: Se True, converte para classificação binária (benigno vs ataque)
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    # Faz uma cópia para não modificar o original
    df_processed = df.copy()
    
    # Remove valores nulos
    df_processed = df_processed.dropna()
    
    # Detecta coluna alvo se não fornecida
    if target_column is None:
        # Usa a função para encontrar colunas adequadas
        suitable_cols = find_suitable_target_columns(df_processed)
        
        # Prioriza colunas adequadas com keywords
        recommended = [col['column'] for col in suitable_cols if col['is_suitable'] and col['has_keyword']]
        if recommended:
            target_column = recommended[0]
        else:
            # Tenta colunas adequadas sem keywords
            suitable = [col['column'] for col in suitable_cols if col['is_suitable']]
            if suitable:
                target_column = suitable[0]
            else:
                # Fallback: procura por keywords mesmo que não seja "suitable"
                possible_targets = [col['column'] for col in suitable_cols if col['has_keyword']]
                if possible_targets:
                    target_column = possible_targets[0]
                else:
                    # Último recurso: última coluna
                    target_column = df_processed.columns[-1]
                    print(f"Aviso: Usando última coluna '{target_column}' como target. Verifique se é adequada para classificação.")
    
    # Separa features e target
    # Remove também a coluna 'device' se existir (não deve ser usada como feature)
    columns_to_drop = [target_column]
    if 'device' in df_processed.columns and 'device' != target_column:
        columns_to_drop.append('device')
    
    X = df_processed.drop(columns=columns_to_drop)
    y = df_processed[target_column]
    
    # Verifica se a coluna target foi realmente removida
    if target_column in X.columns:
        raise ValueError(f"Erro: A coluna target '{target_column}' ainda está presente nas features!")
    
    # Verifica se a coluna device foi removida
    if 'device' in X.columns:
        X = X.drop(columns=['device'])
        print("Aviso: Coluna 'device' removida das features (não deve ser usada como feature).")
    
    # Remove colunas não numéricas (se houver)
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_columns]
    
    # Verifica se há features duplicadas ou com variância zero
    if len(X.columns) == 0:
        raise ValueError("Nenhuma feature numérica encontrada após o pré-processamento!")
    
    # Remove features com variância zero (constantes)
    constant_features = X.columns[X.nunique() <= 1]
    if len(constant_features) > 0:
        print(f"Aviso: {len(constant_features)} feature(s) constante(s) removida(s): {list(constant_features)}")
        X = X.drop(columns=constant_features)
    
    if len(X.columns) == 0:
        raise ValueError("Nenhuma feature útil encontrada após remover features constantes!")
    
    # Verifica quantos valores únicos existem na variável target
    unique_values = y.nunique()
    total_values = len(y)
    
    # Se houver muitos valores únicos (mais de 50% dos valores são únicos), 
    # provavelmente é um problema de regressão, não classificação
    if unique_values > max(50, total_values * 0.5):
        raise ValueError(
            f"A variável target '{target_column}' tem {unique_values} valores únicos de {total_values} amostras. "
            f"Isso parece ser um problema de regressão, não classificação. "
            f"Por favor, selecione uma coluna com valores categóricos (classes)."
        )
    
    # Se binary_classification=True, converte labels para binário (benigno vs ataque)
    if binary_classification:
        # Verifica se há dados benignos
        benign_count = y.apply(lambda x: str(x).lower() == 'benign').sum()
        attack_count = len(y) - benign_count
        
        if benign_count == 0:
            raise ValueError(
                "Nenhum dado 'benign' encontrado no dataset! "
                "A classificação binária requer pelo menos alguns dados benignos. "
                "Certifique-se de carregar arquivos com tráfego benigno (ex: '1.benign.csv')."
            )
        
        if attack_count == 0:
            raise ValueError(
                "Nenhum dado de ataque encontrado no dataset! "
                "A classificação binária requer dados de ataque. "
                "Certifique-se de carregar arquivos com tráfego de ataque."
            )
        
        # Converte para binário: "benign" -> 0 (benigno), tudo mais -> 1 (ataque)
        y_binary = y.apply(lambda x: 0 if str(x).lower() == 'benign' else 1)
        y = y_binary
        print(f"✅ Modo binário ativado: 'benign' -> 0 (benigno, {benign_count:,} amostras), outros -> 1 (ataque, {attack_count:,} amostras)")
    
    # Sempre codifica a variável target como categórica para classificação
    # Isso garante que valores numéricos sejam tratados como classes discretas
    label_encoder = LabelEncoder()
    
    if y.dtype == 'object' or y.dtype.name == 'category':
        # Se for string/categórica, usa LabelEncoder
        y_encoded = label_encoder.fit_transform(y)
    else:
        # Se for numérica, verifica se são valores inteiros discretos
        # Se sim, pode ser classes numéricas (0, 1, 2, etc.)
        if y.dtype in ['int64', 'int32', 'int16', 'int8']:
            # Valores inteiros - pode ser classes numéricas
            # Verifica se são poucos valores únicos (classificação)
            if unique_values <= 50:
                # Trata como classes categóricas
                y_encoded = y.values.astype(int)
                # Cria um LabelEncoder para manter consistência
                label_encoder.fit(y_encoded)
            else:
                raise ValueError(
                    f"A variável target '{target_column}' tem {unique_values} valores únicos. "
                    f"Isso parece ser um problema de regressão. "
                    f"Para classificação, a variável target deve ter poucos valores únicos (classes)."
                )
        else:
            # Valores float - provavelmente contínuos
            # Verifica se podem ser discretizados
            if unique_values <= 50:
                # Arredonda para inteiros e trata como classes
                y_rounded = np.round(y.values).astype(int)
                y_encoded = label_encoder.fit_transform(y_rounded)
                print(f"Aviso: Valores float foram arredondados e convertidos para classes inteiras.")
            else:
                raise ValueError(
                    f"A variável target '{target_column}' tem valores contínuos (float) com {unique_values} valores únicos. "
                    f"Isso é um problema de regressão, não classificação. "
                    f"Por favor, selecione uma coluna com valores categóricos ou discretos."
                )
    
    # Verifica a distribuição das classes antes de fazer a divisão
    unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
    min_class_count = class_counts.min()
    
    # Remove classes com menos de 2 amostras (necessário para stratify)
    if min_class_count < 2:
        # Filtra classes com pelo menos 2 amostras
        valid_classes = unique_classes[class_counts >= 2]
        mask = np.isin(y_encoded, valid_classes)
        X = X.iloc[mask] if isinstance(X, pd.DataFrame) else X[mask]
        y_encoded = y_encoded[mask]
        
        if len(valid_classes) == 0:
            raise ValueError("Nenhuma classe tem pelo menos 2 amostras. Não é possível fazer divisão treino/teste.")
        
        print(f"Aviso: {len(unique_classes) - len(valid_classes)} classe(s) com menos de 2 amostras foram removidas.")
    
    # Normaliza as features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Verifica novamente se todas as classes têm pelo menos 2 amostras
    unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
    min_class_count = class_counts.min()
    
    # Garante que y_encoded seja do tipo inteiro (necessário para classificação)
    y_encoded = y_encoded.astype(int)
    
    # Validação final: verifica se é realmente um problema de classificação
    unique_classes_final = np.unique(y_encoded)
    if len(unique_classes_final) < 2:
        raise ValueError(
            f"Após o pré-processamento, restaram apenas {len(unique_classes_final)} classe(s). "
            f"É necessário pelo menos 2 classes para classificação."
        )
    
    # Divide em treino e teste
    # Usa stratify apenas se todas as classes tiverem pelo menos 2 amostras
    if min_class_count >= 2:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
        except ValueError:
            # Se ainda houver problema com stratify, divide sem estratificação
            print("Aviso: Usando divisão sem estratificação devido a problemas com distribuição de classes.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=test_size, random_state=random_state
            )
    else:
        # Se ainda houver problema, divide sem stratify
        print("Aviso: Usando divisão sem estratificação devido a classes com poucas amostras.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=random_state
        )
    
    # Garante que y_train e y_test também sejam inteiros
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    return X_train, X_test, y_train, y_test, scaler, label_encoder


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, 
                        min_samples_split=2, min_samples_leaf=1, criterion='gini', random_state=42):
    """
    Treina um modelo Random Forest
    
    Args:
        X_train: Features de treino
        y_train: Labels de treino
        n_estimators: Número de árvores
        max_depth: Profundidade máxima das árvores
        min_samples_split: Número mínimo de amostras para dividir um nó
        min_samples_leaf: Número mínimo de amostras em uma folha
        criterion: Critério de divisão ('gini' ou 'entropy')
        random_state: Seed para reprodutibilidade
    
    Returns:
        Modelo treinado
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, 
                  learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, 
                  min_child_weight=1, random_state=42):
    """
    Treina um modelo XGBoost
    
    Args:
        X_train: Features de treino
        y_train: Labels de treino
        n_estimators: Número de árvores
        max_depth: Profundidade máxima das árvores
        learning_rate: Taxa de aprendizado
        subsample: Proporção de amostras para treinar cada árvore
        colsample_bytree: Proporção de features para cada árvore
        min_child_weight: Peso mínimo necessário em uma folha
        random_state: Seed para reprodutibilidade
    
    Returns:
        Modelo treinado
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost não está instalado. Execute: pip install xgboost")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    return model


def train_model(algorithm, X_train, y_train, **kwargs):
    """
    Treina um modelo baseado no algoritmo selecionado
    
    Args:
        algorithm: Nome do algoritmo ('random_forest', 'xgboost')
        X_train: Features de treino
        y_train: Labels de treino
        **kwargs: Parâmetros específicos do algoritmo
    
    Returns:
        Modelo treinado
    """
    if algorithm == 'random_forest':
        return train_random_forest(X_train, y_train, **kwargs)
    elif algorithm == 'xgboost':
        return train_xgboost(X_train, y_train, **kwargs)
    else:
        raise ValueError(f"Algoritmo '{algorithm}' não suportado. Use 'random_forest' ou 'xgboost'.")


def evaluate_model(model, X_test, y_test, label_encoder=None, X_train=None, y_train=None):
    """
    Avalia o modelo e retorna métricas
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Labels de teste
        label_encoder: Encoder de labels (opcional)
        X_train: Features de treino (opcional, para calcular métricas de treino)
        y_train: Labels de treino (opcional, para calcular métricas de treino)
    
    Returns:
        Dicionário com métricas e predições
    """
    y_pred_test = model.predict(X_test)
    
    # Métricas de teste
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall_test = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    
    # Métricas de treino (se fornecidas)
    train_metrics = None
    if X_train is not None and y_train is not None:
        y_pred_train = model.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
        recall_train = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
        f1_train = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
        
        train_metrics = {
            'accuracy': accuracy_train,
            'precision': precision_train,
            'recall': recall_train,
            'f1_score': f1_train
        }
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred_test)
    
    # Relatório de classificação
    if label_encoder:
        target_names = label_encoder.classes_
    else:
        target_names = None
    
    report = classification_report(y_test, y_pred_test, target_names=target_names, output_dict=True)
    
    return {
        'accuracy': accuracy_test,
        'precision': precision_test,
        'recall': recall_test,
        'f1_score': f1_test,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred_test,
        'y_test': y_test,
        'train_metrics': train_metrics  # Métricas de treino para comparação
    }


def get_feature_importance(model, feature_names, top_n=20):
    """
    Retorna as features mais importantes do modelo
    
    Args:
        model: Modelo treinado (Random Forest ou XGBoost)
        feature_names: Nomes das features
        top_n: Número de features top para retornar
    
    Returns:
        DataFrame com features e importâncias, ou None se não disponível
    """
    # Random Forest e XGBoost têm feature_importances_
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        feature_importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        }).sort_values('Importance', ascending=False)
        
        return feature_importance_df
    
    # Se não tiver feature importance
    else:
        return None

