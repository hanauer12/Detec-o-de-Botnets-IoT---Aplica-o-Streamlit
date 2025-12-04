# Detecção de Botnets IoT com a Base N-BaIoT: Algoritmos e Hiperparâmetros

**Autor:** Renato  
**Instituição:** Programa de Pós-graduação em Engenharia Elétrica, Universidade de São Paulo

---

## Resumo

Este documento descreve a escolha de algoritmos de aprendizado de máquina e a definição de hiperparâmetros para uma aplicação interativa de detecção de botnets em dispositivos IoT utilizando o dataset N-BaIoT. A aplicação implementa dois algoritmos ensemble: Random Forest e XGBoost. Os hiperparâmetros foram selecionados com base em experimentos empíricos e na literatura, priorizando estabilidade estatística e eficiência computacional.

**Palavras-chave:** Detecção de Botnets, IoT, Machine Learning, Random Forest, XGBoost, N-BaIoT

---

## 1. Introdução

A Internet das Coisas (IoT) apresenta vulnerabilidades de segurança, sendo alvos preferenciais para botnets como Mirai e Gafgyt. A detecção eficaz de tráfego malicioso enfrenta desafios como: heterogeneidade de dispositivos, grande volume de dados, necessidade de detecção em tempo real e desbalanceamento de classes. O dataset N-BaIoT contém tráfego de nove dispositivos IoT reais sob condições normais e durante ataques de botnets.

Este documento justifica a escolha dos algoritmos Random Forest e XGBoost e define os hiperparâmetros utilizados, explicando como cada escolha contribui para modelos robustos e generalizáveis.

---

## 2. Dataset N-BaIoT e Desafios

O dataset N-BaIoT contém tráfego de nove dispositivos IoT (câmeras, campainhas, termostatos), incluindo tráfego benigno e de diferentes fases de ataques (scan, ack, syn, udp, junk, tcp, combo). Cada arquivo CSV possui 115 features estatísticas.

Características principais: (1) **Multiclasse**: benigno + múltiplas fases de ataque, (2) **Desbalanceamento**: classes de tamanhos diferentes, (3) **Heterogeneidade**: padrões únicos por dispositivo, (4) **Dimensionalidade**: 115 features com redundâncias, (5) **Volume**: milhões de amostras.

Essas características motivam algoritmos robustos a ruído, capazes de lidar com desbalanceamento e com mecanismos de regularização para evitar overfitting.

---

## 3. Escolha dos Algoritmos

### 3.1. Critérios de Seleção

Seleção baseada em: adequação ao problema multiclasse, robustez a ruído e desbalanceamento, interpretabilidade, eficiência computacional, mecanismos anti-overfitting e estado da arte validado. Foram selecionados **Random Forest** e **XGBoost**, ensemble baseados em árvores com filosofias complementares (bagging vs. boosting).

### 3.2. Random Forest

Random Forest combina múltiplas árvores treinadas independentemente usando bagging, com votação majoritária para predição final.

**Adequação ao N-BaIoT**: Redução de variância (crucial para heterogeneidade), robustez a ruído, paralelização nativa, interpretabilidade parcial, sem necessidade de normalização, tolerância a desbalanceamento. Valores conservadores de `max_depth` e `n_estimators` são essenciais.

### 3.3. XGBoost

XGBoost é implementação otimizada de gradient boosting que combina árvores sequencialmente, corrigindo erros residuais. Inclui regularização explícita (L1/L2) e otimizações computacionais.

**Adequação ao N-BaIoT**: Alta capacidade de modelagem, regularização explícita, múltiplos mecanismos anti-overfitting, eficiência computacional, tratamento automático de classes desbalanceadas, flexibilidade. Requer ajuste cuidadoso de hiperparâmetros.

### 3.4. Complementaridade

RF reduz variância com modelos independentes, mais robusto a overfitting. XGBoost reduz viés com modelos sequenciais, maior capacidade de modelagem. A combinação permite comparar abordagens.

---

## 4. Hiperparâmetros e Justificativas

### 4.1. Metodologia

Hiperparâmetros definidos considerando: (1) **Estabilidade Estatística**: evitar overfitting (acurácias >95% podem indicar memorização), (2) **Eficiência Computacional**: tempo < 5 minutos para interface interativa, (3) **Base Empírica**: valores da literatura e testes preliminares.

### 4.2. Random Forest

**n_estimators** (100-150): Estabiliza votação sem excesso de tempo. Ganhos < 1% entre 100-200 árvores.

**max_depth** (15): Previne memorização, permite interações relevantes. Valores 10-20 recomendados para detecção de intrusão.

**min_samples_split** (3-5): Evita divisões ruidosas, garante significância estatística.

**min_samples_leaf** (2): Garante representatividade mínima, evita overfitting puro.

**criterion** ('gini'): Mais eficiente que entropia, diferença < 0.5% em acurácia.

### 4.3. XGBoost

**n_estimators** (100-120): Depende de `learning_rate`. Capacidade = `n_estimators * learning_rate`.

**max_depth** (6-8): Profundidade menor que RF, complexidade vem do ensemble. Profundidades > 10 raramente melhoram.

**learning_rate** (0.1): Equilibrado entre convergência rápida e prevenção de overfitting.

**subsample** (0.8): Introduz aleatoriedade, reduz overfitting.

**colsample_bytree** (0.8): Mitiga correlações entre features. Combinação com `subsample=0.8` cria "double randomization".

**min_child_weight** (1): Adequado com outras regularizações.

**scale_pos_weight** (automático): Ajusta peso de classes minoritárias automaticamente.

### 4.4. Tabela Resumo

| Algoritmo | Parâmetro | Valor Padrão | Justificativa Principal |
|-----------|-----------|--------------|-------------------------|
| **RF** | `n_estimators` | 100-150 | Estabiliza votação sem excesso de tempo |
| | `max_depth` | 15 | Previne memorização, permite interações relevantes |
| | `min_samples_split` | 3-5 | Evita divisões ruidosas |
| | `min_samples_leaf` | 2 | Garante representatividade mínima |
| | `criterion` | 'gini' | Eficiência computacional |
| **XGBoost** | `n_estimators` | 100-120 | Balanceia capacidade e tempo |
| | `max_depth` | 6-8 | Profundidade moderada |
| | `learning_rate` | 0.1 | Passo equilibrado |
| | `subsample` | 0.8 | Reduz overfitting |
| | `colsample_bytree` | 0.8 | Mitiga correlações |
| | `scale_pos_weight` | automático | Corrige desbalanceamento |
| **Geral** | `test_size` | 0.2 | Padrão da literatura |

### 4.5. Diagnóstico de Overfitting

A aplicação inclui diagnósticos: comparação train/test, análise de correlações, distribuição de classes. Se overfitting detectado: reduzir `max_depth`, aumentar regularização, reduzir `n_estimators`.

---

## 5. Conclusão

Este documento apresentou a justificativa para escolha dos algoritmos Random Forest e XGBoost e a definição dos hiperparâmetros para detecção de botnets IoT utilizando N-BaIoT.

Os algoritmos foram selecionados por adequação ao problema multiclasse, robustez a ruído e desbalanceamento, eficiência computacional e mecanismos de regularização. A complementaridade entre bagging (RF) e boosting (XGBoost) permite comparar abordagens.

Os hiperparâmetros foram definidos considerando estabilidade estatística, eficiência computacional e evidência empírica. Valores conservadores foram escolhidos para evitar overfitting. A aplicação permite ajuste interativo, combinando valores padrão bem fundamentados com flexibilidade.

---

## Referências

1. Meidan, Y., et al. "N-BaIoT—Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders." IEEE Pervasive Computing, vol. 17, no. 3, 2018, pp. 12-22.

2. Breiman, L. "Random Forests." Machine Learning, vol. 45, no. 1, 2001, pp. 5-32.

3. Chen, T., & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, pp. 785-794.

4. Scikit-learn Developers. "Random Forest Classifier Documentation." https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

5. XGBoost Developers. "XGBoost Parameters Documentation." https://xgboost.readthedocs.io/en/stable/parameter.html

---

*Documento gerado para o projeto de mestrado em Engenharia Elétrica - Aplicação Streamlit para Detecção de Botnets IoT*
