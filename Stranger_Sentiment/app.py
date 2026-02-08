import os
# --- FIX SALVAVITA PER MAC (BUS ERROR) ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# --- CONFIGURAZIONE E PERCORSI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')

# File da caricare
ANALYSIS_RESULTS_FILE = os.path.join(DATA_RESULTS_DIR, 'sentiment_analysis_results.csv')      # Percentuali
FULL_RESULTS_FILE = os.path.join(DATA_RESULTS_DIR, 'sentiment_analysis_results_full.csv')     # Dati completi per i conteggi
VALIDATION_PREDICTIONS_FILE = os.path.join(DATA_RESULTS_DIR, 'validation_predictions.csv')    # Validazione


# --- CARICAMENTO DATI ---

@st.cache_data
def load_full_counts():
    """Carica il dataset completo SOLO per calcolare i numeri assoluti (Counts)."""
    if not os.path.exists(FULL_RESULTS_FILE):
        return pd.DataFrame()
    
    try:
        # Leggiamo solo le colonne necessarie per risparmiare memoria
        df = pd.read_csv(FULL_RESULTS_FILE, usecols=['season', 'Predicted_Sentiment'])
        # Filtriamo solo Pos/Neg
        df = df[df['Predicted_Sentiment'].isin(['POSITIVE', 'NEGATIVE'])]
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def load_analysis_percentages():
    """Carica i dati aggregati (Percentuali) per i grafici."""
    if not os.path.exists(ANALYSIS_RESULTS_FILE):
        return pd.DataFrame()
    
    df = pd.read_csv(ANALYSIS_RESULTS_FILE)
    df.rename(columns={'Percentage': 'Percentuale', 'season': 'Stagione', 'Predicted_Sentiment': 'Sentiment'}, inplace=True)
    df['Stagione'] = pd.Categorical(df['Stagione'], categories=['S1', 'S2', 'S3', 'S4', 'S5'], ordered=True)
    df = df[df['Sentiment'].isin(['POSITIVE', 'NEGATIVE'])]
    return df

@st.cache_data
def load_validation_predictions():
    """Carica le previsioni di validazione per la matrice di confusione."""
    if not os.path.exists(VALIDATION_PREDICTIONS_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(VALIDATION_PREDICTIONS_FILE)
        return df
    except:
        return pd.DataFrame()


# --- FUNZIONI GRAFICHE ---

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_text = [[str(y) for y in x] for x in cm]
    fig = ff.create_annotated_heatmap(z=cm, x=labels, y=labels, annotation_text=cm_text, colorscale='Blues')
    fig.update_layout(title='Matrice di Confusione', xaxis_title='Predizioni', yaxis_title='Reale')
    return fig


# --- DASHBOARD MAIN ---

def main():
    st.set_page_config(layout="wide", page_title="Analisi Hype Stranger Things", page_icon="🔥")
    
    # --- COLORI ---
    colors = {'POSITIVE': '#D62728', 'NEGATIVE': '#1F77B4'} # Rosso vs Blu

    # --- DATI ---
    df_full = load_full_counts()       # Per i numeri totali
    df_perc = load_analysis_percentages() # Per i grafici %
    df_val = load_validation_predictions() # Per la validazione

    # --- HEADER E DESCRIZIONE ---
    st.title("🔥 Stranger Things: Sentiment Analysis (S1-S5)")
    st.markdown("### 📊 Dashboard di Monitoraggio Hype")
    
    with st.expander("ℹ️  Dettagli del Progetto e Metodologia (Clicca per espandere)", expanded=True):
        st.markdown("""
        Questo progetto analizza l'evoluzione del sentiment dei fan (Hype) attraverso i commenti YouTube sotto i trailer ufficiali USA.
        
        **Pipeline del Progetto:**
        1.  **Data Acquisition:** Estrazione commenti tramite YouTube API.
        2.  **Preprocessing:** Pulizia testo, rimozione emoji, filtro lingua inglese.
        3.  **Sentiment Analysis:** Utilizzo del modello Transformer **DistilBERT** (fine-tuned SST-2).
        4.  **Validazione:** Confronto con un set di dati etichettato manualmente (Ground Truth).
        """)

    st.markdown("---")

    # =================================================================
    # SEZIONE 0: PANORAMICA NUMERICA (KPI)
    # =================================================================
    if not df_full.empty:
        st.subheader("📈 Panoramica Dati Estratti")
        
        # Creiamo 5 colonne per le 5 stagioni
        cols = st.columns(5)
        seasons = ['S1', 'S2', 'S3', 'S4', 'S5']
        
        for i, season in enumerate(seasons):
            # Filtriamo i dati per stagione
            subset = df_full[df_full['season'] == season]
            count_tot = len(subset)
            count_pos = len(subset[subset['Predicted_Sentiment'] == 'POSITIVE'])
            count_neg = len(subset[subset['Predicted_Sentiment'] == 'NEGATIVE'])
            
            # Visualizziamo la metrica
            with cols[i]:
                st.metric(
                    label=f"Stagione {season}",
                    value=f"{count_tot} Comm.",
                    delta=f"Pos: {count_pos} | Neg: {count_neg}",
                    delta_color="off" # Grigio neutro
                )
    else:
        st.error("Dati completi non trovati. Esegui 'python code/sentiment_processord.py'.")

    st.markdown("---")

    # =================================================================
    # SEZIONE 1: ANALISI GLOBALE (ISTOGRAMMA COMPLETO)
    # =================================================================
    st.header("1. Evoluzione Temporale (Tutte le Stagioni)")
    
    if not df_full.empty:
        # Creiamo un grafico che conta i commenti (o usa le percentuali se preferisci)
        # Qui usiamo i CONTEGGI ASSOLUTI per far vedere la mole di commenti
        counts_by_season = df_full.groupby(['season', 'Predicted_Sentiment']).size().reset_index(name='Conteggio')
        
        fig_global = px.bar(
            counts_by_season,
            x='season',
            y='Conteggio',
            color='Predicted_Sentiment',
            barmode='group',
            color_discrete_map=colors,
            title="Volume di Commenti Positivi vs Negativi per Stagione",
            text_auto=True
        )
        st.plotly_chart(fig_global, use_container_width=True)
    
    st.markdown("---")


    # =================================================================
    # SEZIONE 2: CONFRONTO DIRETTO A vs B
    # =================================================================
    st.header("2. Confronto Diretto (Focus Percentuale)")
    
    # Sidebar Filtri (spostata qui logicamente)
    if not df_perc.empty:
        available_seasons = sorted(df_perc['Stagione'].dropna().unique())
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            season_a = st.selectbox("Seleziona Stagione A", options=available_seasons, index=0)
        with col_sel2:
            season_b = st.selectbox("Seleziona Stagione B", options=available_seasons, index=1 if len(available_seasons)>1 else 0)
        
        df_compare = df_perc[df_perc['Stagione'].isin([season_a, season_b])]
        
        col_bar, col_pie = st.columns([2, 1])
        
        with col_bar:
            fig_stacked = px.bar(
                df_compare, x='Stagione', y='Percentuale', color='Sentiment',
                color_discrete_map=colors, text_auto='.1f', barmode='group',
                title=f"Distribuzione Percentuale: {season_a} vs {season_b}"
            )
            st.plotly_chart(fig_stacked, use_container_width=True)

        with col_pie:
            # Filtriamo solo i positivi per vedere "Chi ha vinto l'Hype"
            df_pos = df_compare[df_compare['Sentiment'] == 'POSITIVE']
            if not df_pos.empty:
                fig_pie = px.pie(
                    df_pos, values='Percentuale', names='Stagione',
                    title='Confronto Hype Relativo (Solo Positivi)',
                    hole=0.4, color_discrete_sequence=['#FF4B4B', '#4B4BFF']
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # =================================================================
    # SEZIONE 3: VALIDAZIONE
    # =================================================================
    st.header("3. Affidabilità del Modello (Validazione)")
    st.markdown("Performance calcolata su Dataset Bilanciato (Ground Truth).")
    
    if not df_val.empty:
        y_true = df_val['Ground_Truth_Label']
        y_pred = df_val['Predicted_Sentiment']
        
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report).transpose().drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        
        col_met, col_mat = st.columns([1, 2])
        
        with col_met:
            st.metric("Accuratezza Totale", f"{accuracy * 100:.2f}%")
            st.dataframe(df_report.style.format("{:.2f}"), use_container_width=True)
            
        with col_mat:
            labels = sorted(list(set(y_true) | set(y_pred)))
            fig_cm = plot_confusion_matrix(y_true, y_pred, labels)
            st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.warning("⚠️ Dati di validazione non trovati.")

if __name__ == "__main__":
    main()
