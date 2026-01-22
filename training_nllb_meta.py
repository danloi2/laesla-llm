# ============================================================
# 🌏 META NLLB-200: ENTRENAMIENTO BIDIRECCIONAL PROFESIONAL
# ============================================================
# Este script usa el modelo de Meta para traducir Latín ↔ Español simultáneamente.

!pip install -q transformers[sentencepiece] datasets accelerate -U sacremoses

import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
import shutil
from google.colab import files

# 1️⃣ Carga de Datos
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name, sep='|', quoting=3)

# 2️⃣ Aumentar datos para BIDIRECCIONALIDAD
# Creamos el sentido A -> B
df_la_es = df.copy()
df_la_es.columns = ["src", "tgt"]
df_la_es["src_lang"] = "lat_Latn"
df_la_es["tgt_lang"] = "spa_Latn"

# Creamos el sentido B -> A
df_es_la = df.copy()
df_es_la.columns = ["tgt", "src"] # Invertimos columnas
df_es_la["src_lang"] = "spa_Latn"
df_es_la["tgt_lang"] = "lat_Latn"

# Unimos ambos (Ahora tenemos 64,000 frases)
df_bidirectional = pd.concat([df_la_es, df_es_la], ignore_index=True)
print(f"✅ Dataset bidireccional listo: {len(df_bidirectional)} filas.")

# 3️⃣ Cargar Modelo de META (NLLB 600M - Versión equilibrada)
MODEL_NAME = "facebook/nllb-200-distilled-600M"
tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 4️⃣ Preprocesamiento
dataset = Dataset.from_pandas(df_bidirectional).shuffle(seed=42)
dataset_split = dataset.train_test_split(test_size=0.05)

def preprocess(examples):
    # NLLB requiere setear el lenguaje para cada frase
    model_inputs = tokenizer(examples["src"], max_length=128, truncation=True, padding="max_length")
    
    # Tokenizar el objetivo
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["tgt"], max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = dataset_split.map(preprocess, batched=True)

# 5️⃣ Entrenamiento (Configuración para 32k/64k frases)
training_args = Seq2SeqTrainingArguments(
    output_dir="./nllb_latin_es_bi",
    per_device_train_batch_size=8, # NLLB es más grande, cuidado con la memoria
    gradient_accumulation_steps=4, # Batch efectivo de 32
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=True,
    save_total_limit=2,
    evaluation_strategy="epoch",
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer
)

print("🚀 Iniciando entrenamiento con Meta NLLB...")
trainer.train()

# 6️⃣ Guardado
model_dir = "./modelo_meta_nllb_final"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

shutil.make_archive("nllb_bi_model", 'zip', model_dir)
files.download("nllb_bi_model.zip")
