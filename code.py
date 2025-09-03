#!/usr/bin/env python
# coding: utf-8

# 1. INSTALLAZIONE PACCHETTI

import os
import random
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim import AdamW
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    DataCollatorWithPadding, get_scheduler
)
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    auc, classification_report, f1_score, accuracy_score,
    precision_recall_curve, roc_auc_score, roc_curve, confusion_matrix,
    average_precision_score, matthews_corrcoef, balanced_accuracy_score
)


# Definizione del modello
MODELS = {
    "bert": "dbmdz/bert-base-italian-uncased",
    "xlmr": "xlm-roberta-base"
}
RUN = ["bert"]

# CONFIGURAZIONE PATH

DATA_DIR = "./data"
RESULTS_DIR = "./results"
COMPUTE_DIR = "./compute_logs"


USE_COLAB = True

if USE_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/data"
    RESULTS_DIR = "/content/drive/MyDrive/Colab Notebooks/results"
    COMPUTE_DIR = "/content/drive/MyDrive/Colab Notebooks/compute_logs"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(COMPUTE_DIR, exist_ok=True)

# Configurazione log dei file

log_file = os.path.join(RESULTS_DIR, f"training_log_{RUN[0]}.txt")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    filemode="w",
    format="%(message)s",
    force=True
)
log = logging.getLogger()


# FUNZIONE CARICAMENTO E ADATTAMENTO DATASET

def load_and_merge_datasets():

    # 1. Feel-IT
    data1 = pd.read_csv(os.path.join(DATA_DIR, "feel_it_dataset.csv"))
    def map_emotion_to_sentiment(label):
        if label == "joy (R)":
            return 1
        elif label in ["sadness (E)", "fear (Q)", "anger (W)"]:
            return 0
        return -1
    data1["labels"] = data1["labels"].apply(map_emotion_to_sentiment)
    data1 = data1[data1["labels"].isin([0, 1])][["text", "labels"]]

    # 2. MultiEmotions-It
    data2 = pd.read_csv(os.path.join(DATA_DIR, "MultiEmotions-It.tsv"), sep="\t")
    data2 = data2.loc[(data2["POS"] == 1) ^ (data2["NEG"] == 1), ["comment", "POS"]]
    data2 = data2.rename(columns={"comment": "text", "POS": "labels"})

    # 3. Sentipolc16 train
    data3 = pd.read_csv(os.path.join(DATA_DIR, "training_set_sentipolc16_anon_rev.csv"), sep=";")
    data3 = data3[data3["opos"] != data3["oneg"]]
    data3["labels"] = data3["oneg"].apply(lambda x: 0 if x == 1 else 1)
    data3 = data3[["text", "labels"]]

    # 4. Sentipolc16 test
    data4 = pd.read_csv(os.path.join(DATA_DIR, "test_set_sentipolc16_gold2000_anon_rev.csv"), sep=";")
    data4 = data4.rename(columns={
        data4.columns[2]: "opos",
        data4.columns[3]: "oneg",
        data4.columns[8]: "text",
    })
    data4 = data4[["opos", "oneg", "text"]]
    data4 = data4[data4["oneg"].isin([0, 1])]
    data4 = data4[data4["opos"] != data4["oneg"]]
    data4["labels"] = data4["oneg"].apply(lambda x: 0 if x == 1 else 1)
    data4 = data4[["text", "labels"]].astype({"labels": int})

    # Merge finale e shuffle
    data = pd.concat([data1, data2, data3, data4], ignore_index=True)
    data = (
        data.dropna()
            .query("text != ''")
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
    )
    data["labels"] = data["labels"].astype(int)
    return data


# Funzione per split stratificato in train (80%), val (10%) e test (10%)
def stratified_split(df, seed):
    train_df, temp_df = train_test_split(
        df, stratify=df['labels'], test_size=0.2, random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, stratify=temp_df['labels'], test_size=0.5, random_state=seed
    )
    return train_df, val_df, test_df

data = load_and_merge_datasets()

# Distribuzione delle classi nei set di training e validation
train_df, val_df, test_df = stratified_split(data, 42)

def log_class_distribution(df, name="Data"):
    counts = df['labels'].value_counts()
    total = len(df)
    perc_pos = counts.get(1, 0) / total * 100
    perc_neg = counts.get(0, 0) / total * 100
    log.info(f"{name}: Positivi={perc_pos:.2f}% | Negativi={perc_neg:.2f}%")

log_class_distribution("\n Distribuzione delle classi nei set di training e validation")
log_class_distribution(train_df, "Training set")
log_class_distribution(val_df, "Validation set")

# Calcolo della random baseline stratificata 

y_true = data['labels'].values 
class_probs = np.bincount(y_true) / len(y_true)  
rng = np.random.default_rng(seed=42) 
y_pred = rng.choice([0, 1], size=len(y_true), p=class_probs) 
log.info(f"Metriche della baseline")
log.info(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
log.info(f"Macro-F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
log.info("\n" + classification_report(y_true, y_pred))

# Configurazione iperparametri

NUM_EPOCHS = 6
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY  = 0.01
max_length = 160
patience = 4
SEEDS = [42, 123, 999]


# Multi-run con Accelerate e HuggingFace

accelerator = Accelerator(mixed_precision="fp16")
all_metrics = []

# Richaimo al nome del modello prima del ciclo di training
MODEL_NAME = MODELS[RUN[0]]

# Tokenizzatore
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
def tokenize_fn(batch):
    return tokenizer(batch['text'], truncation=True, padding=False, max_length=max_length)

losses_across_seeds = []
roc_across_seeds = []

# Implementazione della Focal Loss per classi sbilanciate

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.5, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits, labels):
        labels = labels.float()
        bce_loss = self.bce(logits, labels)
        probs = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probs, 1 - probs)
        focal_factor = (1 - pt) ** self.gamma
        return (self.alpha * focal_factor * bce_loss).mean()


# Calcolo e log costi computazionali totali: parte 1
start_time = time.time()
torch.cuda.reset_peak_memory_stats()


# Ciclo for di training loop in base al SEED
for SEED in SEEDS:
    log.info(f"\n Inizio training con SEED = {SEED}")

    # RIproducibilità
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Split stratificato
    train_df, val_df, test_df = stratified_split(data, SEED)

    # Conversione in Dataset HuggingFace
    train_dataset = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True)
    cols = ['input_ids','attention_mask','labels']
    for ds in [train_dataset, val_dataset, test_dataset]:
        ds.set_format(type='torch', columns=cols)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    # Definizione modello, scheduler e preparazione con Accelerate

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    num_training_steps = len(train_loader) * NUM_EPOCHS
    num_warmup_steps = max(1, int(0.06 * num_training_steps))
    lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                 num_warmup_steps=num_warmup_steps,
                                 num_training_steps=num_training_steps)

    model, optimizer, train_loader, val_loader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, lr_scheduler
    )


    # Calcolo pesi per classi sbilanciate e richiamo della Focal Loss
    class_counts = np.bincount(train_df['labels'])
    neg, pos = int(class_counts[0]), int(class_counts[1])
    pos_weight = torch.tensor(neg / max(pos, 1) * 1.2, dtype=torch.float32)

    loss_fn = FocalLoss(alpha=1.0, gamma=1.5, pos_weight=pos_weight.to(accelerator.device))

    # Ciclo di training
    best_macro_f1 = -1.0
    best_threshold = 0.5
    patience_counter = 0
    train_losses = []
    val_losses = []


    for epoch in range(NUM_EPOCHS):

        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = loss_fn(outputs.logits.view(-1), batch['labels'].float())

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(1, num_batches)

        # Validation
        model.eval()
        all_preds_val, all_labels_val = [], []
        val_loss_total, val_batches = 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                loss = loss_fn(outputs.logits.view(-1), batch['labels'].float())
                val_loss_total += loss.item()
                val_batches += 1

                probs = torch.sigmoid(outputs.logits.view(-1))
                all_preds_val.extend(probs.cpu().numpy())
                all_labels_val.extend(batch['labels'].cpu().numpy())

        avg_val_loss = val_loss_total / max(1, val_batches)
        # Salvataggio losses
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)


        # Calcolo F1 e threshold ottimale
        all_preds_val = np.array(all_preds_val)
        all_labels_val = np.array(all_labels_val)
        prec, rec, thr = precision_recall_curve(all_labels_val, all_preds_val, pos_label=1)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_thr_epoch = thr[f1_scores.argmax()]
        preds_val_bin = (all_preds_val >= best_thr_epoch).astype(int)
        val_f1 = f1_score(all_labels_val, preds_val_bin, average="macro")


        log.info(f"\n Epoch {epoch+1}/{NUM_EPOCHS} - "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val F1: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_macro_f1 + 1e-6:
            best_macro_f1 = val_f1
            best_threshold = best_thr_epoch
            patience_counter = 0

            torch.save(model.state_dict(), f"best_model_seed{SEED}.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            log.info(">>> Early stopping triggered, uso il best model precedente <<<")

        # Ricarica i pesi migliori salvati
    model.load_state_dict(torch.load(f"best_model_seed{SEED}.pt"))


    # Test finale e metriche
    model.eval()
    all_preds_test, all_labels_test = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(input_ids=batch['input_ids'].to(accelerator.device),
                            attention_mask=batch['attention_mask'].to(accelerator.device))
            probs = torch.sigmoid(outputs.logits.view(-1))
            all_preds_test.extend(probs.cpu().numpy())
            all_labels_test.extend(batch['labels'].cpu().numpy())

    all_preds_test = np.array(all_preds_test)
    all_labels_test = np.array(all_labels_test)

    fpr, tpr, _ = roc_curve(all_labels_test, all_preds_test)
    roc_auc_val = auc(fpr, tpr)

    test_preds_bin = (all_preds_test >= best_threshold).astype(int)
    acc = accuracy_score(all_labels_test, test_preds_bin)
    f1 = f1_score(all_labels_test, test_preds_bin, average="macro")
    roc_auc = roc_auc_score(all_labels_test, all_preds_test)
    report = classification_report(all_labels_test, test_preds_bin, output_dict=True)
    pr_auc = average_precision_score(all_labels_test, all_preds_test)
    mcc = matthews_corrcoef(all_labels_test, test_preds_bin)
    bal_acc = balanced_accuracy_score(all_labels_test, test_preds_bin)
    metrics_per_class = {cls: {'precision': report[cls]['precision'],
                               'recall': report[cls]['recall'],
                               'support': report[cls]['support']}
                         for cls in ['0', '1']}

    log.info(f"\n Run SEED={SEED} --> Accuracy={acc:.4f}, Macro-F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, MCC={mcc:.4f}, Balanced Accuracy={bal_acc:.4f}")
    for cls, m in metrics_per_class.items():
        log.info(f"  Class {cls}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, Support={m['support']}")


    # Matrice di confusione
    cm = confusion_matrix(all_labels_test, test_preds_bin)
    TN, FP, FN, TP = cm.ravel()

    # False Positive Rate e False Negative Rate
    fpr = FP / (FP + TN) if (FP + TN) != 0 else np.nan
    fnr = FN / (FN + TP) if (FN + TP) != 0 else np.nan

    log.info("\nConfusion Matrix:")
    log.info(cm)
    log.info(f"False Positive Rate (FPR) = {fpr:.4f} ({fpr*100:.2f}%)")
    log.info(f"False Negative Rate (FNR) = {fnr:.4f} ({fnr*100:.2f}%)")



    # Error analysis
    test_df = test_df.reset_index(drop=True)
    errors_idx = np.where(test_preds_bin != all_labels_test)[0]
    errors = test_df.iloc[errors_idx]
    errors['predicted_label'] = test_preds_bin[errors_idx]
    errors['predicted_prob'] = all_preds_test[errors_idx]

    log.info("\n Error Analysis: le prime 10 frasi")
    for i, row in errors.head(10).iterrows():
        log.info(f"{i}: Text = {row['text']}\n   True label = {row['labels']}, Predicted = {row['predicted_label']}, Prob = {row['predicted_prob']:.4f}")


    # Salvataggio per grafici
    losses_across_seeds.append((SEED, train_losses, val_losses))
    fpr, tpr, _ = roc_curve(all_labels_test, all_preds_test)
    roc_across_seeds.append((SEED, fpr, tpr, roc_auc))


    # Log metriche aggregate per ogni seed
    all_metrics.append({
        "seed": SEED,
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "mcc": float(mcc),
        "bal_acc": float(bal_acc),
        "best_threshold": float(best_threshold),
        "class_metrics": {cls: {'precision': float(report[cls]['precision']),
                               'recall': float(report[cls]['recall']),
                               'support': int(report[cls]['support'])}
                         for cls in ['0', '1']}
    })

# Calcolo e log costi computazionali totali: parte 2
end_time = time.time()
elapsed_min = (end_time - start_time) / 60
max_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

total_compute_log = {
    "model": MODEL_NAME,
    "epochs_per_seed": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "max_len": max_length,
    "seeds_run": len(SEEDS),
    "total_elapsed_min": round(elapsed_min,2),
    "max_mem_gb": round(max_mem_gb,2)
}

# GRAFICI
# Curve di Loss
plt.figure(figsize=(7,5))
for seed, train_l, val_l in losses_across_seeds:
    plt.plot(range(1, len(train_l)+1), train_l, marker='o', label=f"Train Seed {seed}")
    plt.plot(range(1, len(val_l)+1), val_l, marker='x', label=f"Val Seed {seed}")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curves across Seeds")
plt.legend()
plt.grid(alpha=0.3)
loss_plot_path = os.path.join(RESULTS_DIR, f"loss_curves_{RUN[0]}.png")
plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
plt.close()

# Curve di ROC
plt.figure(figsize=(7,5))
for seed, fpr, tpr, auc_val in roc_across_seeds:
    plt.plot(fpr, tpr, label=f"Seed {seed} (AUC={auc_val:.3f})")

plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves across Seeds")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
loss_plot_path = os.path.join(RESULTS_DIR, f"ROC_curves_{RUN[0]}.png")
plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
plt.close()

# Metriche aggregate
accuracy_mean = np.mean([m["accuracy"] for m in all_metrics])
accuracy_std  = np.std([m["accuracy"] for m in all_metrics])
f1_mean = np.mean([m["macro_f1"] for m in all_metrics])
f1_std  = np.std([m["macro_f1"] for m in all_metrics])
roc_mean = np.mean([m["roc_auc"] for m in all_metrics])
roc_std  = np.std([m["roc_auc"] for m in all_metrics])
pr_auc_mean = np.mean([m["pr_auc"] for m in all_metrics])
pr_auc_std  = np.std([m["pr_auc"] for m in all_metrics])
mcc_mean = np.mean([m["mcc"] for m in all_metrics])
mcc_std  = np.std([m["mcc"] for m in all_metrics])
bal_acc_mean = np.mean([m["bal_acc"] for m in all_metrics])
bal_acc_std  = np.std([m["bal_acc"] for m in all_metrics])

log.info("\n  Metriche aggregate finali")
log.info(f"Accuracy            : {accuracy_mean:.4f} ± {accuracy_std:.4f}")
log.info(f"Macro-F1            : {f1_mean:.4f} ± {f1_std:.4f}")
log.info(f"ROC-AUC             : {roc_mean:.4f} ± {roc_std:.4f}")
log.info(f"PR-AUC              : {pr_auc_mean:.4f} ± {pr_auc_std:.4f}")
log.info(f"MCC                 : {mcc_mean:.4f} ± {mcc_std:.4f}")
log.info(f"Balanced Accuracy   : {bal_acc_mean:.4f} ± {bal_acc_std:.4f}")

for cls in ['0','1']:
    precisions = [m["class_metrics"][cls]['precision'] for m in all_metrics]
    recalls = [m["class_metrics"][cls]['recall'] for m in all_metrics]
    supports = [m["class_metrics"][cls]['support'] for m in all_metrics]
    log.info(f"Class {cls}: Precision={np.mean(precisions):.4f} ± {np.std(precisions):.4f}, "
          f"Recall={np.mean(recalls):.4f} ± {np.std(recalls):.4f}, "
          f"Support avg={np.mean(supports):.0f}")

# Log dei file e shutdown

results_file = os.path.join(RESULTS_DIR, f"all_metrics_{RUN[0]}.json")
with open(results_file, "w") as f:
    json.dump(all_metrics, f, indent=4)

compute_file = os.path.join(COMPUTE_DIR, f"total_compute_log_{RUN[0]}.json")
with open(compute_file, "w") as f:
    json.dump(total_compute_log, f, indent=4)

logging.shutdown()
