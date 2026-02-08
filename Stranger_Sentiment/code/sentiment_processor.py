import pandas as pd
import os
import sys
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import pipeline

# --- CONFIGURAZIONE GLOBALE ---
VIDEO_MAP_HYPE = {
    'S1': {'FILE_PREFIX': 'S1_Hype'},
    'S2': {'FILE_PREFIX': 'S2_Hype'},
    'S3': {'FILE_PREFIX': 'S3_Hype'},
    'S4': {'FILE_PREFIX': 'S4_Hype'},
    'S5': {'FILE_PREFIX': 'S5_Hype'},
}

# Definisce i percorsi
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')
DATA_RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')
os.makedirs(DATA_RESULTS_DIR, exist_ok=True)

# Nomi dei file
VALIDATION_SET_LABELED_FILE = os.path.join(VALIDATION_DIR, 'validation_set_labeled.csv')
ANALYSIS_RESULTS_FILE = os.path.join(DATA_RESULTS_DIR, 'sentiment_analysis_results.csv')
# NUOVO FILE: Qui salviamo le previsioni per l'app
VALIDATION_PREDICTIONS_FILE = os.path.join(DATA_RESULTS_DIR, 'validation_predictions.csv')


# --- INIZIALIZZAZIONE MODELLO ---
def initialize_pipeline():
    """Carica il modello DistilBERT su GPU o CPU."""
    if torch.backends.mps.is_available():
        device_id = "mps"
    elif torch.cuda.is_available():
        device_id = 0
    else:
        device_id = -1
        
    try:
        pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device_id
        )
        print(f"\n[MODELLO] Pipeline caricata su: {device_id}")
        return pipe
    except Exception as e:
        print(f"[ERRORE CRITICO] Caricamento modello fallito: {e}")
        sys.exit(1)

sentiment_pipeline = initialize_pipeline()


# --- FUNZIONI DI PREDIZIONE ---
def predict_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        result = sentiment_pipeline(text, truncation=True, max_length=512)[0]
        return result['label']
    except Exception:
        return None


# --- FASE 3A: ANALISI COMPLETA ---
def run_full_analysis():
    print("\n--- FASE 3A: ANALISI SENTIMENT COMPLETA ---")
    
    full_results_path = ANALYSIS_RESULTS_FILE.replace('.csv', '_full.csv')
    all_data = []

    for stagione, data in VIDEO_MAP_HYPE.items():
        processed_file = os.path.join(DATA_PROCESSED_DIR, f"{data['FILE_PREFIX']}_processed.csv")

        if not os.path.exists(processed_file):
            continue
        
        try:
            df = pd.read_csv(processed_file)
            if 'text' not in df.columns: df = pd.read_csv(processed_file, sep=';')
        except:
            continue
        
        if df.empty or 'text' not in df.columns: continue
            
        print(f"Analisi {stagione}...")
        df['Predicted_Sentiment'] = df['text'].apply(predict_sentiment)
        df_clean = df.dropna(subset=['Predicted_Sentiment']).copy()
        df_clean['season'] = stagione
        all_data.append(df_clean)

    if all_data:
        df_results = pd.concat(all_data)
        sentiment_counts = df_results.groupby('season')['Predicted_Sentiment'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()
        df_results.to_csv(full_results_path, index=False)
        sentiment_counts.to_csv(ANALYSIS_RESULTS_FILE, index=False)
        print(f"[OK] Risultati analisi salvati.")
    else:
        print("[ERRORE] Nessun dato analizzato.")


# --- FASE 3B: VALIDAZIONE E SALVATAGGIO ---
def validate_and_save():
    """Convalida il modello e SALVA i risultati per l'App."""
    print("\n--- FASE 3B: VALIDAZIONE E SALVATAGGIO ---")
    
    if not os.path.exists(VALIDATION_SET_LABELED_FILE):
        print(f"[ERRORE] File validazione non trovato.")
        return

    try:
        df_val = pd.read_csv(VALIDATION_SET_LABELED_FILE)
    except:
        return

    # Pulizia base
    df_val.columns = [c.strip() for c in df_val.columns]
    df_val.dropna(subset=['Ground_Truth_Label', 'text'], inplace=True)
    df_val['Ground_Truth_Label'] = df_val['Ground_Truth_Label'].astype(str).str.upper().str.strip()
    df_val = df_val[df_val['Ground_Truth_Label'].isin(['POSITIVE', 'NEGATIVE'])].copy()

    print(f"Validazione su {len(df_val)} commenti...")
    
    # Eseguiamo le predizioni ORA (così l'app non deve farlo)
    df_val['Predicted_Sentiment'] = df_val['text'].apply(predict_sentiment)
    df_val = df_val.dropna(subset=['Predicted_Sentiment'])

    # Salviamo questo file prezioso per Streamlit!
    df_val.to_csv(VALIDATION_PREDICTIONS_FILE, index=False)
    print(f"[OK] Previsioni validazione salvate in: {VALIDATION_PREDICTIONS_FILE}")

    # Stampiamo report rapido
    acc = accuracy_score(df_val['Ground_Truth_Label'], df_val['Predicted_Sentiment'])
    print(f"Accuratezza calcolata: {acc*100:.2f}%")


if __name__ == "__main__":
    run_full_analysis()
    validate_and_save()
    print("\n--- ELABORAZIONE COMPLETATA ---")
