import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import re

df = pd.read_csv('./train_cleaned.csv')
df_val = pd.read_csv('./validation_cleaned.csv')

# Binarize labels
mlb = MultiLabelBinarizer(classes=['positive','negative', 'ambiguous'])
y_train = mlb.fit_transform(df['super_categories'])
y_val = mlb.transform(df_val['super_categories'])


# Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def get_class_weights(labels):
    class_counts = np.sum(labels, axis=0)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # Inverse frequency weighting with smoothing
    weights = total_samples / (num_classes * (class_counts + 1e-6))
    return torch.tensor(weights, dtype=torch.float32)


# Model and Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3,
        problem_type="multi_label_classification"
       
    )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from torch.nn import BCEWithLogitsLoss
# Custom Trainer
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = BCEWithLogitsLoss(
            pos_weight=self.class_weights.to(labels.device),
            reduction='mean'
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Optimized Training Arguments
training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,  
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=64,
        learning_rate=3e-5,  
        warmup_ratio=0.05,
        weight_decay=0.001,
        logging_steps=100,
        eval_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model='micro_f1',
        fp16=True,
        gradient_accumulation_steps=2  
    )

# Metrics Calculation
def compute_metrics(p):
    preds = torch.sigmoid(torch.tensor(p.predictions)).numpy()
    y_true = p.label_ids

    # Define different threshold ranges per class if needed
    class_threshold_ranges = {
        'positive': np.linspace(0.2, 0.6, 9),
        'negative': np.linspace(0.2, 0.6, 9),
        'ambiguous': np.linspace(0.2, 0.6, 9),
        

    }

    best_thresholds = {}
    class_names = ['positive','negative', 'ambiguous']
    
    # Find optimal threshold for each class independently
    for idx, cls in enumerate(class_names):
        cls_preds = preds[:, idx]
        cls_true = y_true[:, idx]
        
        best_f1 = 0
        best_thresh = class_threshold_ranges[cls][3]  # Default middle value
        
        for thresh in class_threshold_ranges[cls]:
            cls_pred = (cls_preds > thresh).astype(int)
            current_f1 = f1_score(cls_true, cls_pred, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_thresh = thresh
        
        best_thresholds[cls] = best_thresh

    # Apply class-specific thresholds
    y_pred = np.zeros_like(preds)
    for idx, cls in enumerate(class_names):
        y_pred[:, idx] = (preds[:, idx] > best_thresholds[cls]).astype(int)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    return {
        'micro_f1': report['micro avg']['f1-score'],
        'macro_f1': report['macro avg']['f1-score'],
        **{f"{cls}_threshold": best_thresholds[cls] for cls in class_names},
        **{f"{cls}_f1": report[cls]['f1-score'] for cls in class_names}
    }

class_weights = get_class_weights(y_train)
print("Class weights:", class_weights)
# Initialize and train
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=EmotionDataset(df['text_clean'].tolist(), y_train, tokenizer),
    eval_dataset=EmotionDataset(df_val['text_clean'].tolist(), y_val, tokenizer),
    compute_metrics=compute_metrics,
    class_weights=class_weights
)

# Train and save
trainer.train()
trainer.save_model("./optimized_emotion_classifier4")
tokenizer.save_pretrained("./optimized_emotion_classifier4")

# Final evaluation
print("\nBest Model Evaluation:")
results = trainer.evaluate()
print(f"Micro F1: {results['eval_micro_f1']:.4f}")
print(f"Macro F1: {results['eval_macro_f1']:.4f}")

import matplotlib.pyplot as plt

# After training is complete
logs = trainer.state.log_history

# Extract training metrics
train_loss = []
train_steps = []
for log in logs:
    if 'loss' in log and 'step' in log:
        train_loss.append(log['loss'])
        train_steps.append(log['step'])

# Extract validation metrics
eval_steps = []
eval_loss = []
eval_micro_f1 = []
eval_macro_f1 = []
for log in logs:
    if 'eval_loss' in log and 'step' in log:
        eval_steps.append(log['step'])
        eval_loss.append(log['eval_loss'])
        eval_micro_f1.append(log.get('eval_micro_f1', 0))
        eval_macro_f1.append(log.get('eval_macro_f1', 0))

# Create two plots
plt.figure(figsize=(15, 6))

# First plot: Combined Loss
plt.subplot(1, 2, 1)
plt.plot(train_steps, train_loss, label='Train Loss', color='blue')
plt.plot(eval_steps, eval_loss, label='Val Loss', color='red', marker='o')
plt.title('Training vs Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Second plot: Validation Metrics
plt.subplot(1, 2, 2)
plt.plot(eval_steps, eval_micro_f1, label='Micro F1', color='green')
plt.plot(eval_steps, eval_macro_f1, label='Macro F1', color='purple')
plt.title('Validation Metrics')
plt.xlabel('Steps')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_plots.png')
plt.close()

print("Plots saved as training_plots.png")