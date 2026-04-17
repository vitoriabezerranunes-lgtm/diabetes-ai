import requests
import pandas as pd
import io

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

print("Baixando dataset...")
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(io.StringIO(response.text), header=None, names=columns)
df.to_csv("diabetes.csv", index=False)
print(f"Dataset salvo como diabetes.csv — {len(df)} registros, {len(df.columns)} colunas")
print(df.head())
