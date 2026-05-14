# 🧠 Sistema de Triaje Médico con Red Neuronal Artificial (RNA)

## 📌 Descripción del proyecto

Este proyecto es un sistema de clasificación de pacientes en triaje médico basado en una **Red Neuronal Artificial (RNA)** desarrollada en **Python con TensorFlow**.

El modelo predice el nivel de prioridad de atención médica (**ESI 1 a ESI 5**) a partir de datos clínicos como signos vitales, laboratorio y motivo de consulta.

La aplicación incluye una interfaz gráfica (GUI) creada con **Tkinter**, permitiendo ingresar datos de un paciente y obtener una predicción en tiempo real.

## 🏥 ¿Qué es el sistema ESI?

El sistema **Emergency Severity Index (ESI)** clasifica pacientes según la urgencia:

| Nivel | Descripción |
|-------|-------------|
| ESI 1 | Crítico (riesgo vital inmediato) |
| ESI 2 | Emergencia |
| ESI 3 | Urgente |
| ESI 4 | Menos urgente |
| ESI 5 | No urgente |

## ⚙️ Tecnologías utilizadas

| Tecnología | Uso |
|------------|-----|
| 🐍 Python 3.11 | Lenguaje principal |
| 🤖 TensorFlow / Keras | Red neuronal |
| 📊 Pandas | Manipulación de datos |
| 🔢 NumPy | Operaciones numéricas |
| 🧪 Scikit-learn | Preprocesamiento |
| 🖥 Tkinter | Interfaz gráfica |
| 📁 CSV | Dataset clínico |

## 🧠 ¿Cómo funciona?

1. Se carga un dataset clínico de pacientes
2. Se limpian y procesan los datos
3. Se convierten variables categóricas en numéricas
4. Se entrena una red neuronal multicapa
5. El modelo aprende a clasificar el nivel ESI
6. La interfaz permite ingresar datos de un paciente
7. El sistema predice el nivel de urgencia en tiempo real

## 🚀 Cómo ejecutar el proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/LitzyCastellanos/Sistema-de-Triaje-RNA.git
```

## 2. Entrar a la carpeta
bash
cd Sistema-de-Triaje-RNA
3. Instalar dependencias
bash
pip install pandas numpy scikit-learn tensorflow
Nota: Este proyecto requiere Python 3.11 específicamente. Verifica tu versión con:

bash
python --version
4. Ejecutar el sistema
bash
python triaje_RNA.py
🖥 Interfaz del sistema
Al ejecutar el programa se abre una ventana donde puedes ingresar:

Edad

Sexo

Presión arterial

Frecuencia cardíaca

Temperatura

Laboratorios

Motivo de consulta

Luego presionas:

👉 "Predecir nivel ESI"

y el sistema mostrará:

Nivel ESI (1–5)

Probabilidades por clase

Indicador visual por color

📊 Características del modelo
Red neuronal con capas densas

Dropout para evitar sobreajuste

Batch normalization

Early stopping

📁 Estructura del proyecto
text
Sistema-de-Triaje-RNA/
│
├── triaje_RNA.py                 # Código principal + GUI
├── fedmml_ed_triage_dataset.csv  # Dataset de entrenamiento
├── modelo_triaje_esi.h5          # Modelo entrenado (generado)
└── README.md                     # Este archivo
👩‍💻 Autor
Karen Castellanos
Proyecto académico — Inteligencia Artificial aplicada a la salud
