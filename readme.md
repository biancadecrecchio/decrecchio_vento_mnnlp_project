## **Classificazione Binaria del Sentiment in italiano: Confronto Multi-Seed di BERT-base Italian uncased e XLM-RoBERTa-base**



## Abstract
Questo progetto esplora il task di classificazione binaria del sentiment in italiano mediante il fine-tuning di modelli transformer (“BERT-base Italian uncased” e “XLM-RoBERTa-base” multilingue) su un dataset combinato di risorse esistenti (FEEL-IT, MultiEmotions-It, SENTIPOLC16). L’adozione di un approccio multi-seed ha permesso di valutarne la robustezza, mostrando prestazioni ampiamente superiori rispetto alla baseline casuale stratificata. I risultati evidenziano una lieve superiorità del modello BERT-base in Accuracy e Macro-F1, mentre il modello XLM-RoBERTa si distingue nelle metriche probabilistiche, come ROC-AUC. Valutato su uno split stratificato del dataset combinato, BERT-base supera UmBERTo – addestrato per lo stesso task su FEEL-IT e valutato sul test set di SENTIPOLC16 – nelle metriche Macro-F1 e Accuracy.  

## Dataset

Per l’addestramento e la valutazione dei modelli BERT e RoBERTa abbiamo utilizzato dataset già annotati per la sentiment analysis in italiano, provenienti da social media (Twitter, YouTube e Facebook).

- **FEEL-IT**: 2.037 tweet annotati dal MilaNLP Group per quattro emozioni di base (rabbia, paura, gioia, tristezza). Tali emozioni sono state da noi rimappate secondo il sentiment da esse espresso, positivo o negativo.

- **MultiEmotions-It**: 3.240 commenti annotati manualmente per polarità (positiva/negativa), rilevanza, emozioni e sarcasmo, raccolti da YouTube e Facebook durante un seminario universitario.

- **SENTIPOLC 2016 (EVALITA)**: 9.410 tweet annotati per polarità, soggettività e ironia. Questo dataset, usato nel task ufficiale EVALITA 2016, è già suddiviso in train (7.410 esempi) e test (2.000 esempi).


## Struttura del codice

Il codice è organizzato nel seguente modo:

- Analisi preliminare del dataset

- Funzioni di supporto (loss personalizzata, splitting stratificato, logging)

- Training multi-seed con HuggingFace Accelerate

- Generazione e salvataggio dei grafici (curve di loss, ROC)

- Salvataggio metriche aggregate in formato .json e .txt

## Installazione e utilizzo
Per riprodurre il progetto, seguire questi passaggi:
Assicurarsi di avere installato Python 3.12.11.

Installare le dipendenze richieste eseguendo:  
	`pip install -r requirements.txt`

Una volta installate le dipendenze, eseguire il codice con il seguente comando:  
	`python code.py`

Se si vuole eseguire il codice con XLM RoBERTa, modificare la variabile RUN all'inizio del file code.py:
	`RUN = ["xlmr"]` 
e poi eseguire nuovamente:
	`python code.py` 

## File e cartelle del progetto
- **code.py**: Contiene il codice per l'analisi del testo e le visualizzazioni.  
- **requirements.txt**: Elenca le librerie necessarie per il progetto.  
- **data/**: Contiene il dataset utilizzato per l'analisi.  
- **results/**: Contiene i risultati e i grafici di BERT generati dal codice, e quelli di RoBERTa computati offline.
- **compute_logs/**: Contiene i valori dei costi computazionali di BERT, e quelli di RoBERTa computati offline.

## Licenze dei dataset utilizzati
- FEEL-IT: la licenza si trova nella cartella licenses\.
- MultiEmotions-It: licenza non esplicitata, dataset rilasciato per fini accademici nel contesto di un seminario universitario.
- SENTIPOLC 2016: rilasciato da EVALITA 2016, disponibile per scopi di ricerca, nessuna licenza commerciale esplicita.

















