import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ETAPA 1 — CARGA DEL DATASET________________________________________________________________________________________-
print("=" * 60)
print("ETAPA 1: Cargando dataset...")
print("=" * 60)

ds = pd.read_csv("fedmml_ed_triage_dataset.csv")

print(f"Total de registros: {len(ds)}")
print(f"Total de columnas:  {len(ds.columns)}")


#ESI : Indice de emergencia 
esi_nombres = {1: "Crítico", 2: "Emergencia", 3: "Urgente", 4: "Menos urgente", 5: "No urgente"}
esi_colores = {1: "#c0392b", 2: "#e67e22", 3: "#d4ac0d", 4: "#27ae60", 5: "#2980b9"}

# Resumen de Datos
print("\nDistribución de ESI Level:") 
for nivel, nombre in esi_nombres.items():
    count = (ds["esi_level"] == nivel).sum()
    print(f"  ESI {nivel} ({nombre}): {count} pacientes ({count/len(ds)*100:.1f}%)")


# ETAPA 2 — LIMPIEZA DE DATOS_____________________________________________________________________________________________________
print("\n" + "=" * 60)
print("ETAPA 2: Preprocesando datos...")
print("=" * 60)

columnas_numericas = [ # Datos que no se modifican porque son numericos
    "age",
    "systolic_bp", "diastolic_bp", "heart_rate",
    "respiratory_rate", "temperature", "spo2", "pain_score",
    "wbc", "hemoglobin", "platelet_count",
    "sodium", "potassium", "creatinine",
    "glucose", "troponin", "bnp", "lactate", "inr"
]

# Sexo F=0, M=1
ds["sex_enc"] = (ds["sex"] == "M").astype(int)

# Rellena espacios vacios
chief_dummies = pd.get_dummies(ds["chief_complaint"], prefix="cc") #Binario de razon de consulta

# Traducción de motivos de consulta inglés → español
traduccion_complaints = {
    "Back pain":                    "Dolor de espalda",
    "Cardiac arrest":               "Paro cardíaco",
    "Chest pain":                   "Dolor de pecho",
    "Chronic pain stable":          "Dolor crónico estable",
    "Cold symptoms":                "Síntomas de resfriado",
    "Cough":                        "Tos",
    "Fever":                        "Fiebre",
    "Follow-up visit":              "Visita de seguimiento",
    "Head injury with LOC":         "Traumatismo craneal con pérdida de consciencia",
    "Headache severe":              "Dolor de cabeza severo",
    "High fever with confusion":    "Fiebre alta con confusión",
    "Laceration requiring sutures": "Laceración que requiere sutura",
    "Massive hemorrhage":           "Hemorragia masiva",
    "Medication question":          "Consulta sobre medicación",
    "Mild abdominal discomfort":    "Malestar abdominal leve",
    "Minor injury":                 "Lesión menor",
    "Minor rash":                   "Sarpullido menor",
    "Moderate abdominal pain":      "Dolor abdominal moderado",
    "Prescription refill":          "Renovación de receta",
    "Rash":                         "Sarpullido",
    "Seizure ongoing":              "Convulsión activa",
    "Severe abdominal pain":        "Dolor abdominal severo",
    "Severe respiratory distress":  "Dificultad respiratoria severa",
    "Severe shortness of breath":   "Disnea severa",
    "Sprain":                       "Esguince",
    "Stroke symptoms":              "Síntomas de accidente cerebrovascular",
    "Unresponsive":                 "Sin respuesta",
    "Vomiting":                     "Vómitos",
}

# Lista ordenada en español para el dropdown
lista_complaints_es = sorted(traduccion_complaints.values())

# Diccionario inverso: español → inglés
traduccion_inversa  = {v: k for k, v in traduccion_complaints.items()}

# Eliminacion de nulos con mediana
medianas = {}
for col in columnas_numericas:
    medianas[col] = ds[col].median()
    nulos = ds[col].isnull().sum()
    if nulos > 0:
        print(f"  Eliminando {nulos} nulos en '{col}' con mediana={medianas[col]:.2f}")
    ds[col] = ds[col].fillna(medianas[col])

X = pd.concat([ds[columnas_numericas], ds[["sex_enc"]], chief_dummies], axis=1) #tabla gigante completa
y = ds["esi_level"].values - 1  # ESI 1-5 → clases 0-4

print(f"\nFeatures totales: {X.shape[1]}")


# ETAPA 3 — DIVISIÓN Y NORMALIZACIÓN_____________________________________________________________________________________________________
print("\n" + "=" * 60)
print("ETAPA 3: Dividiendo y normalizando datos...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y #proporciones iguales
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Entrenamiento: {X_train_scaled.shape[0]} registros")
print(f"Prueba:        {X_test_scaled.shape[0]} registros")


# ETAPA 4 — CONSTRUCCIÓN DE LA RNA_____________________________________________________________________________________________________
print("\n" + "=" * 60)
print("ETAPA 4: Construyendo la Red Neuronal Artificial...")
print("=" * 60)

n_features = X_train_scaled.shape[1] ##Numero de columnas
n_clases   = 5


modelo = keras.Sequential([
    layers.Input(shape=(n_features,)),

    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(), ## PESOS CALCULADOS NO APRENDIDOS PARA MANTENER DATOS NORMALIZADOS
    layers.Dropout(0.3),

    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(n_clases, activation="softmax")
])


modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

modelo.summary()


# ETAPA 5 — ENTRENAMIENTO_____________________________________________________________________________________________________
print("\n" + "=" * 60)
print("ETAPA 5: Entrenando el modelo...")
print("=" * 60)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10, #Espera 10 epocas sin cambios
    restore_best_weights=True
)

historial = modelo.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)


# ETAPA 6 — EVALUACIÓN_____________________________________________________________________________________________________
print("\n" + "=" * 60)
print("ETAPA 6: Evaluando el modelo...")
print("=" * 60)

loss, accuracy = modelo.evaluate(X_test_scaled, y_test, verbose=1)#que y como se imprime
print(f"\nAccuracy en datos de prueba: {accuracy * 100:.2f}%")
print(f"Loss en datos de prueba:     {loss:.4f}")

y_pred = np.argmax(modelo.predict(X_test_scaled, verbose=0), axis=1)
print("\nReporte de Clasificación:")
print(classification_report(
    y_test, y_pred,
    target_names=[f"ESI {i} - {esi_nombres[i]}" for i in range(1, 6)]
))


# ETAPA 7 — INTERFAZ TKINTER_____________________________________________________________________________________________________
print("\n" + "=" * 60)
print("ETAPA 7: Abriendo interfaz de predicción...")
print("=" * 60)

def predecir_paciente(valores_form, complaint_es, sexo):
    complaint_en = traduccion_inversa.get(complaint_es, complaint_es)

    fila_num = [valores_form[col] for col in columnas_numericas]
    fila_sex = [1 if sexo == "M" else 0]
    cc_cols  = list(chief_dummies.columns)
    fila_cc  = [1 if col == f"cc_{complaint_en}" else 0 for col in cc_cols]

    fila        = np.array([fila_num + fila_sex + fila_cc])
    fila_scaled = scaler.transform(fila)

    probs = modelo.predict(fila_scaled, verbose=0)[0]
    clase = int(np.argmax(probs)) + 1
    return clase, probs


def abrir_ventana():
    ventana = tk.Tk()
    ventana.title("IS701 — Predicción de Severidad ESI")
    ventana.geometry("860x860")
    ventana.configure(bg="#f0f4f8")
    ventana.resizable(False, False)

    #Header
    header = tk.Frame(ventana, bg="#1a3a5c", pady=14)
    header.pack(fill="x")
    tk.Label(header, text="🏥  Sistema de Predicción de Severidad en Urgencias",
             font=("Helvetica", 15, "bold"), bg="#1a3a5c", fg="white").pack()
    

    #Scroll
    canvas    = tk.Canvas(ventana, bg="#f0f4f8", highlightthickness=0)
    scrollbar = ttk.Scrollbar(ventana, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas, bg="#f0f4f8")

    scroll_frame.bind("<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_mousewheel(event):
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def _unbind_mousewheel(event):
        canvas.unbind_all("<MouseWheel>")

    canvas.bind("<Enter>", _bind_mousewheel)
    canvas.bind("<Leave>", _unbind_mousewheel)

    # crear sección de campos
    entries = {}

    def crear_seccion(parent, titulo, campos):
        outer = tk.Frame(parent, bg="#f0f4f8", padx=18, pady=4)
        outer.pack(fill="x")
        tk.Label(outer, text=titulo, font=("Helvetica", 11, "bold"),
                 bg="#f0f4f8", fg="#1a3a5c").pack(anchor="w", pady=(8, 4))

        inner = tk.Frame(outer, bg="white", padx=12, pady=10,
                         highlightbackground="#d0dce8", highlightthickness=1)
        inner.pack(fill="x")

        for i, (label, key, default, unidad) in enumerate(campos):
            r = i // 2
            c = (i % 2) * 3
            tk.Label(inner, text=label, font=("Helvetica", 9),
                     bg="white", fg="#444", anchor="w", width=22
                     ).grid(row=r, column=c, padx=(8, 2), pady=5, sticky="w")
            var = tk.StringVar(value=str(default))
            tk.Entry(inner, textvariable=var, font=("Helvetica", 10),
                     width=10, relief="solid", bd=1
                     ).grid(row=r, column=c+1, padx=2, pady=5)
            tk.Label(inner, text=unidad, font=("Helvetica", 8),
                     bg="white", fg="#888", width=7
                     ).grid(row=r, column=c+2, padx=(0, 10), sticky="w")
            entries[key] = var

    #Sección 1 Datos del paciente
    crear_seccion(scroll_frame, "👤  Datos del Paciente", [
        ("Edad",          "age",        45, "años"),
        ("Dolor (0-10)",  "pain_score",  5, "/10"),
    ])

    # Sexo + Motivo
    outer_sel = tk.Frame(scroll_frame, bg="#f0f4f8", padx=18, pady=4)
    outer_sel.pack(fill="x")
    inner_sel = tk.Frame(outer_sel, bg="white", padx=12, pady=10,
                          highlightbackground="#d0dce8", highlightthickness=1)
    inner_sel.pack(fill="x")

    tk.Label(inner_sel, text="Sexo", font=("Helvetica", 9),
             bg="white", fg="#444").grid(row=0, column=0, padx=8, sticky="w")
    sexo_var = tk.StringVar(value="M")
    ttk.Combobox(inner_sel, textvariable=sexo_var,
                 values=["M", "F"], state="readonly", width=8
                 ).grid(row=0, column=1, padx=8, pady=6)

    tk.Label(inner_sel, text="Motivo de consulta", font=("Helvetica", 9),
             bg="white", fg="#444").grid(row=0, column=2, padx=(20, 4), sticky="w")
    complaint_var = tk.StringVar(value=lista_complaints_es[0])
    ttk.Combobox(inner_sel, textvariable=complaint_var,
                 values=lista_complaints_es, state="readonly", width=38
                 ).grid(row=0, column=3, padx=8, pady=6)

    #Sección 2 Signos Vitales
    crear_seccion(scroll_frame, "💓  Signos Vitales", [
        ("Presión sistólica",  "systolic_bp",      120,  "mmHg"),
        ("Presión diastólica", "diastolic_bp",      80,  "mmHg"),
        ("Frec. cardíaca",     "heart_rate",        75,  "bpm"),
        ("Frec. respiratoria", "respiratory_rate",  16,  "rpm"),
        ("Temperatura",        "temperature",      36.6, "°C"),
        ("SpO2",               "spo2",              98,  "%"),
    ])

    #Sección 3 Labs
    crear_seccion(scroll_frame, "🧪  Laboratorios", [
        ("Leucocitos (WBC)",  "wbc",            7.0,  "K/µL"),
        ("Hemoglobina",       "hemoglobin",     13.5, "g/dL"),
        ("Plaquetas",         "platelet_count", 250,  "K/µL"),
        ("Sodio",             "sodium",         140,  "mEq/L"),
        ("Potasio",           "potassium",      4.0,  "mEq/L"),
        ("Creatinina",        "creatinine",     1.0,  "mg/dL"),
        ("Glucosa",           "glucose",        100,  "mg/dL"),
        ("Troponina",         "troponin",       0.01, "ng/mL"),
        ("BNP",               "bnp",            50,   "pg/mL"),
        ("Lactato",           "lactate",        1.2,  "mmol/L"),
        ("INR",               "inr",            1.0,  ""),
    ])

    #Botón predecir
    tk.Frame(scroll_frame, bg="#f0f4f8", height=10).pack()
    tk.Button(scroll_frame,
              text="  🔍  Predecir nivel ESI  ",
              font=("Helvetica", 12, "bold"),
              bg="#1a3a5c", fg="white",
              activebackground="#2d5986", activeforeground="white",
              relief="flat", padx=20, pady=10, cursor="hand2",
              command=lambda: on_predecir()
              ).pack()

    #Panel de resultado
    frame_res = tk.Frame(scroll_frame, bg="#f0f4f8", padx=18, pady=10)
    frame_res.pack(fill="x", pady=(10, 80))

    lbl_esi    = tk.Label(frame_res, text="", font=("Helvetica", 26, "bold"), bg="#f0f4f8")
    lbl_nombre = tk.Label(frame_res, text="", font=("Helvetica", 14), bg="#f0f4f8")
    lbl_sub    = tk.Label(frame_res, text="Probabilidades por clase:",
                           font=("Helvetica", 10, "bold"), bg="#f0f4f8", fg="#555")
    barras_frame = tk.Frame(frame_res, bg="#f0f4f8")

    barras = {}
    for nivel in range(1, 6):
        fila_b = tk.Frame(barras_frame, bg="#f0f4f8")
        fila_b.pack(fill="x", pady=3)
        tk.Label(fila_b, text=f"ESI {nivel}  {esi_nombres[nivel]}",
                 font=("Helvetica", 9), bg="#f0f4f8", fg="#333",
                 width=24, anchor="w").pack(side="left")
        cv = tk.Canvas(fila_b, width=320, height=18,
                        bg="#e8eef4", highlightthickness=0)
        cv.pack(side="left", padx=6)
        lp = tk.Label(fila_b, text="", font=("Helvetica", 9),
                       bg="#f0f4f8", fg="#333", width=7)
        lp.pack(side="left")
        barras[nivel] = (cv, lp)

    #Lógica del botón
    def on_predecir():
        try:
            valores = {}
            for col in columnas_numericas:
                val = entries[col].get().strip()
                valores[col] = float(val) if val != "" else medianas[col]

            clase, probs = predecir_paciente(valores, complaint_var.get(), sexo_var.get())
            color = esi_colores[clase]

            lbl_esi.config(text=f"ESI {clase}", fg=color)
            lbl_nombre.config(text=f"● {esi_nombres[clase]}", fg=color)
            lbl_esi.pack(pady=(6, 0))
            lbl_nombre.pack()
            lbl_sub.pack(pady=(12, 4))
            barras_frame.pack(fill="x")

            for nivel in range(1, 6):
                prob  = probs[nivel - 1]
                ancho = int(prob * 320)
                cv, lp = barras[nivel]
                cv.delete("all")
                if ancho > 0:
                    cv.create_rectangle(0, 0, ancho, 18,
                                        fill=esi_colores[nivel], outline="")
                lp.config(text=f"{prob*100:.1f}%")

        except ValueError:
            messagebox.showerror("Error", "Ingresa solo valores numéricos en los campos.")
    tk.Frame(scroll_frame, bg="#f0f4f8", height=60).pack()
    ventana.mainloop()
    


abrir_ventana()