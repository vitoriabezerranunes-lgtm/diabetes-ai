# 🩺 DiabetesAI — Previsão de Risco de Diabetes

## O Problema

O diabetes tipo 2 afeta mais de **537 milhões de adultos** no mundo (IDF, 2021) e é uma das principais causas de cegueira, amputações e doenças renais. No Brasil, cerca de 16 milhões de pessoas vivem com a doença — muitas sem saber. O diagnóstico precoce é crucial para evitar complicações graves.

## A Solução

DiabetesAI é uma aplicação web interativa que utiliza Machine Learning para estimar o risco de diabetes a partir de dados clínicos simples. O modelo é treinado no clássico dataset **Pima Indians Diabetes** (UCI Machine Learning Repository) e apresenta os resultados de forma clara e acessível, em português.

## Tecnologias

- **Python 3.10+**
- **XGBoost** — modelo de gradient boosting de alta performance
- **scikit-learn** — pré-processamento e métricas
- **Streamlit** — interface web interativa
- **pandas** — manipulação de dados

## Estrutura do Projeto

```
diabetes-ai/
├── diabetes.csv          # Dataset com 768 registros
├── train.py              # Treinamento do modelo
├── app.py                # Interface Streamlit
├── model.pkl             # Modelo treinado (gerado após treino)
├── requirements.txt      # Dependências
└── README.md
```

## Como Rodar

### 1. Clone ou baixe o projeto

```bash
cd diabetes-ai
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Baixe o dataset

```bash
python download_data.py
```

### 4. Treine o modelo

```bash
python train.py
```

Saída esperada:
```
Acurácia no conjunto de teste: ~79%
Modelo salvo como model.pkl
```

### 5. Inicie a aplicação

```bash
streamlit run app.py
```

Acesse **http://localhost:8501** no navegador.

## Campos da Interface

| Campo | Descrição |
|---|---|
| Glicose | Concentração de glicose plasmática (mg/dL) |
| IMC | Índice de Massa Corporal (kg/m²) |
| Idade | Idade em anos |
| Pressão Diastólica | Pressão arterial diastólica (mmHg) |
| Gestações | Número de gestações anteriores |
| Insulina | Insulina sérica 2h após teste (µU/mL) |
| Espessura da Pele | Dobra cutânea do tríceps (mm) |
| Função Pedigree | Score de histórico familiar de diabetes |

## Desempenho do Modelo

| Métrica | Valor |
|---|---|
| Acurácia | ~79% |
| Dataset | 768 pacientes, 8 features |
| Algoritmo | XGBoost (200 estimadores) |

## Impacto Social

- **Acesso democratizado** a triagem de risco sem custo
- **Linguagem acessível** — interface 100% em português
- **Educação em saúde** — cada campo tem explicação do que significa
- **Empoderamento** — o usuário entende seu perfil de risco e é incentivado a buscar cuidados médicos
- Pode ser adaptado para uso em **UBSs (Unidades Básicas de Saúde)** e programas de saúde digital do SUS

## Aviso Legal

Este projeto é educacional e não substitui avaliação médica. Os resultados devem ser interpretados por profissionais de saúde.

---

Desenvolvido com ❤️ usando Python, XGBoost e Streamlit.
