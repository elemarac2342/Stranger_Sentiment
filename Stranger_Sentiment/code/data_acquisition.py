import os
import json
import time
import sys
from datetime import datetime
import pandas as pd
from googleapiclient.discovery import build
from langdetect import detect, DetectorFactory

# Imposta la seed per la riproducibilità di langdetect
DetectorFactory.seed = 0

# Importa la chiave API dal file che devi creare manualmente
try:
    from api_key import YOUTUBE_API_KEY
    YOUTUBE = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
except ImportError:
    print("ERRORE: Devi creare il file 'code/api_key.py' con la tua YOUTUBE_API_KEY.")
    sys.exit(1)
except Exception as e:
    print(f"ERRORE: Impossibile inizializzare l'API di YouTube. {e}")
    sys.exit(1)

# --- CONFIGURAZIONE GLOBALE (CON DATE RIGOROSE) ---
VIDEO_MAP_HYPE = {
    'S1': {'ID': 'b9EkMc79ZSU', 'FILE_PREFIX': 'S1_Hype', 'RELEASE_DATE': '2016-07-15'},
    'S2': {'ID': 'R1ZXOOLMJ8s', 'FILE_PREFIX': 'S2_Hype', 'RELEASE_DATE': '2017-10-27'},
    'S3': {'ID': 'PH3kBCSfL-4', 'FILE_PREFIX': 'S3_Hype', 'RELEASE_DATE': '2019-07-04'},
    'S4': {'ID': 'oB2GYwbIAlM', 'FILE_PREFIX': 'S4_Hype', 'RELEASE_DATE': '2022-05-27'}, 
    # S5 con ID video corretto e data di rilascio del Volume 1
    'S5': {'ID': 'PssKpzB0Ah0', 'FILE_PREFIX': 'S5_Hype', 'RELEASE_DATE': '2025-11-26'} 
}

# Definisce i percorsi delle cartelle relative
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

# --- FUNZIONI DI FILTRAGGIO ---

def detect_language(text):
    """Verifica se il testo del commento è in inglese usando langdetect."""
    if not text or len(text.split()) < 3: 
        return False
    try:
        # Filtro Rigoroso: solo lingua 'en' (inglese)
        return detect(text) == 'en'
    except:
        return False

def is_comment_pre_release(comment_time_str, release_date_str):
    """Verifica se il commento è rigorosamente prima della data di rilascio (compreso il giorno stesso)."""
    try:
        # Converti la data di rilascio a mezzanotte
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        
        # Converti il timestamp API (ISO 8601) e rimuovi fuso orario per confronto
        comment_dt = datetime.fromisoformat(comment_time_str.replace('Z', '+00:00')).replace(tzinfo=None)
        
        # Filtro Rigoroso: commento DEVE essere prima o al massimo il giorno della release
        return comment_dt.date() <= release_dt.date()
        
    except ValueError as e:
        return False

# --- PROCESSO PRINCIPALE DI ACQUISIZIONE E FILTRAGGIO ---

def raccogli_e_filtra_dati(video_id, file_prefix, release_date_str):
    """Esegue lo scraping, applica i filtri e salva i dati processati."""
    
    output_path_processed = os.path.join(DATA_PROCESSED_DIR, f"{file_prefix}_processed.csv")
    
    if os.path.exists(output_path_processed) and os.path.getsize(output_path_processed) > 100:
        print(f"[PROCESSATO] File processato {output_path_processed} esiste già. Salto la raccolta/filtro.")
        return 0 # Ritorna 0 per non alterare il conteggio totale

    print(f"\n--- INIZIO: Raccolta e Filtro per {file_prefix} (Video ID: {video_id}) ---")
    print(f"   Filtro Temporale Rigoroso: SOLO commenti PRIMA o IL {release_date_str}")
    
    commenti_validi = []
    commenti_totali_letti = 0
    next_page_token = None
    
    # L'API è limitata a 100 commenti per pagina
    while True:
        try:
            request = YOUTUBE.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                order="time" 
            )
            response = request.execute()

            for item in response['items']:
                commenti_totali_letti += 1
                comment_snippet = item['snippet']['topLevelComment']['snippet']
                text = comment_snippet.get('textDisplay', '')
                time_str = comment_snippet.get('publishedAt', '')

                # 1. Filtro Linguistico (Inglese)
                if not detect_language(text):
                    continue
                
                # 2. Filtro Temporale (Pre-uscita Rigoroso)
                if not is_comment_pre_release(time_str, release_date_str):
                    continue 

                # Se passa entrambi, aggiungiamo
                commenti_validi.append({
                    'text': text,
                    'time': time_str,
                    'season': file_prefix.split('_')[0]
                })

            next_page_token = response.get('nextPageToken')
            
            if not next_page_token:
                break
            
            time.sleep(0.5) 

        except Exception as e:
            print(f"[ERRORE API] Errore durante la richiesta: {e}")
            break
            
    
    # 3. Salvataggio finale solo dei dati validi
    df_validi = pd.DataFrame(commenti_validi)
    df_validi.to_csv(output_path_processed, index=False, encoding='utf-8')
    
    print(f"[SUCCESSO] Raccolta e Filtro completati per {file_prefix}.")
    print(f"   Commenti totali letti: {commenti_totali_letti}")
    print(f"   Commenti validi/filtrati (Inglese, Pre-uscita): {len(df_validi)}")
    print(f"   Dati salvati in: {output_path_processed}")
    
    return len(df_validi)


if __name__ == "__main__":
    print("--- ESECUZIONE FASE EXTRACT & TRANSFORM: ACQUISIZIONE E FILTRO DIRETTO (API) ---")
    
    # Esegue il processo per S1, S2, S3, S4, S5
    commenti_totali_filtrati = 0
    for stagione, data in VIDEO_MAP_HYPE.items():
        
        count = raccogli_e_filtra_dati(
            video_id=data['ID'],
            file_prefix=data['FILE_PREFIX'],
            release_date_str=data['RELEASE_DATE']
        )
        commenti_totali_filtrati += count

    print(f"\n[RISULTATO FINALE] TOTALE Commenti Pre-uscita & Inglese raccolti: {commenti_totali_filtrati}")
    print("\n--- data_acquisitiond.py COMPLETATO ---")