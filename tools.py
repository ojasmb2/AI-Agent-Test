# tools.py
import requests
import os
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from langchain_community.tools import PythonREPLTool

import config

# --- Strumenti per l'API del Corso ---

@tool
def get_all_questions():
    """
    Recupera l'elenco di tutte le domande disponibili per la valutazione finale.
    Da usare solo una volta all'inizio per ottenere i task_id.
    """
    try:
        response = requests.get(f"{config.BASE_API_URL}/questions", headers=config.HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Errore API durante il recupero delle domande: {e}"

@tool
def download_file_by_task_id(task_id: str) -> str:
    """
    Scarica un file associato a uno specifico task_id.
    L'input deve essere il task_id.
    Salva il file localmente e restituisce il percorso del file scaricato.
    Usare questo strumento quando la domanda richiede di analizzare un file.
    """
    try:
        # Crea la directory 'downloads' se non esiste
        if not os.path.exists('downloads'):
            os.makedirs('downloads')

        response = requests.get(f"{config.BASE_API_URL}/files/{task_id}", headers=config.HEADERS)
        response.raise_for_status()

        # Estrai il nome del file dagli header se possibile, altrimenti usa un nome di default
        content_disposition = response.headers.get('content-disposition')
        if content_disposition:
            try:
                filename = content_disposition.split('filename=')[1].strip('"')
            except IndexError:
                filename = f"{task_id}_file" # Fallback
        else:
            filename = f"{task_id}_file"

        file_path = os.path.join('downloads', filename)
        with open(file_path, 'wb') as f:
            f.write(response.content)

        return f"File per il task_id '{task_id}' scaricato con successo in '{file_path}'. Contenuto del file:\n---\n{response.content.decode('utf-8', errors='ignore')[:2000]}\n---"
    except requests.exceptions.RequestException as e:
        return f"Errore API durante il download del file per task_id {task_id}: {e}"
    except Exception as e:
        return f"Errore durante il salvataggio del file per task_id {task_id}: {e}"


@tool
def submit_answer(task_id: str, answer: str) -> str:
    """
    Invia la risposta finale per un dato task_id.
    Prende come input il task_id e la stringa della risposta.
    La risposta deve essere esatta, senza prefissi o formattazione extra.
    """
    payload = {
        "task_id": task_id,
        "submitted_answer": answer,
        "username": config.HF_USERNAME
    }
    try:
        response = requests.post(f"{config.BASE_API_URL}/submit", headers=config.HEADERS, json=payload)
        response.raise_for_status()
        return f"Risposta per {task_id} inviata con successo: {response.json()}"
    except requests.exceptions.RequestException as e:
        return f"Errore API durante l'invio della risposta per {task_id}: {e}"

# --- Strumenti Generici ---

@tool
def web_search(query: str) -> str:
    """
    Esegue una ricerca web utilizzando DuckDuckGo per trovare informazioni aggiornate
    o dati non presenti nel contesto. Utilizza questo strumento quando la domanda
    riguarda eventi recenti, fatti specifici, o richiede conoscenza esterna.
    L'input è la query di ricerca.
    """
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
        return "\n".join([f"Fonte: {r['href']}\nSnippet: {r['body']}" for r in results]) if results else "Nessun risultato trovato."

# Strumento per calcoli e esecuzione codice
python_interpreter = PythonREPLTool()