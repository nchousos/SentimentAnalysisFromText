import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import numpy as np

# Load preprocessed data
df = pd.read_csv('train_cleaned.csv')
df_val = pd.read_csv('validation_cleaned.csv')

# Define the same mappings as in main script
super_category_map = { 
    'positive': ['admiration', 'amusement', 'approval', 'caring', 'gratitude', 'excited_joyful', 'love', 'optimism'],
    'negative': ['irritation', 'disapproval', 'disgust', 'disappointment', 'fear', 'remorse', 'sadness', 'embarrassment'],
    'ambiguous': ['confusion', 'curiosity','desire','surprise']
}


# Dataset Class (same as main script)
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=250):
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
def compute_metrics(p):
    preds = torch.sigmoid(torch.tensor(p.predictions)).numpy()
    y_true = p.label_ids

    # Find optimal threshold between 0.2-0.5
    thresholds = np.linspace(0.2, 0.5, 7)
    best_f1 = 0
    best_thresh = 0.3
    
    for thresh in thresholds:
        current_preds = (preds > thresh).astype(int)
        current_f1 = f1_score(y_true, current_preds, average='micro', zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = thresh

    # Final prediction with best threshold
    y_pred = (preds > best_thresh).astype(int)
    
    return {
        'eval_micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'eval_macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'eval_threshold': best_thresh
    }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
import matplotlib.pyplot as plt

def train_submodels(super_category):
    
    emotions_in_super = super_category_map[super_category]
    
    train_mask = df['emotions'].apply(lambda x: any(e in emotions_in_super for e in eval(x)))
    val_mask = df_val['emotions'].apply(lambda x: any(e in emotions_in_super for e in eval(x)))
    
    df_super_train = df[train_mask].copy()
    df_super_val = df_val[val_mask].copy()

    # Extract specific emotions
    df_super_train['specific_emotions'] = df_super_train['emotions'].apply(
        lambda x: [e for e in eval(x) if e in emotions_in_super]
    )
    df_super_val['specific_emotions'] = df_super_val['emotions'].apply(
        lambda x: [e for e in eval(x) if e in emotions_in_super]
    )

    
    mlb_specific = MultiLabelBinarizer(classes=emotions_in_super)
    y_super_train = mlb_specific.fit_transform(df_super_train['specific_emotions'])
    y_super_val = mlb_specific.transform(df_super_val['specific_emotions'])

    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    emotions_in_super = super_category_map[super_category]

    
    id2label = {i: label for i, label in enumerate(emotions_in_super)}
    label2id = {label: i for i, label in enumerate(emotions_in_super)}

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(emotions_in_super),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'./results_{super_category}',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f'./logs_{super_category}',
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_micro_f1',  
        fp16=True
    )

    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=EmotionDataset(df_super_train['text_clean'].tolist(), y_super_train, tokenizer),
        eval_dataset=EmotionDataset(df_super_val['text_clean'].tolist(), y_super_val, tokenizer),
        compute_metrics=compute_metrics  # Add this line
    )

    # Train and save
    trainer.train()
    model.save_pretrained(f"./specific_models/{super_category}")
    tokenizer.save_pretrained(f"./specific_models/{super_category}")
    print(f"Saved model for {super_category}")

    generate_training_plots(trainer, super_category)

def generate_training_plots(trainer, super_category):
    logs = trainer.state.log_history
    
    # Extract metrics
    train_metrics = {
        'loss': [],
        'steps': []
    }
    eval_metrics = {
        'loss': [],
        'micro_f1': [],
        'macro_f1': [],
        'steps': []
    }
    
    for log in logs:
        if 'loss' in log and 'step' in log:  # Training log
            train_metrics['loss'].append(log['loss'])
            train_metrics['steps'].append(log['step'])
        elif 'eval_loss' in log:  # Evaluation log
            eval_metrics['loss'].append(log['eval_loss'])
            eval_metrics['micro_f1'].append(log.get('eval_micro_f1', 0))
            eval_metrics['macro_f1'].append(log.get('eval_macro_f1', 0))
            eval_metrics['steps'].append(log['step'])
    
    # Create plots
    plt.figure(figsize=(15, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_metrics['steps'], train_metrics['loss'], label='Train Loss', color='blue')
    if eval_metrics['steps']:
        plt.plot(eval_metrics['steps'], eval_metrics['loss'], label='Val Loss', color='red', marker='o')
    plt.title(f'{super_category} - Training vs Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Metrics plot
    plt.subplot(1, 2, 2)
    if eval_metrics['steps']:
        plt.plot(eval_metrics['steps'], eval_metrics['micro_f1'], label='Micro F1', color='green')
        plt.plot(eval_metrics['steps'], eval_metrics['macro_f1'], label='Macro F1', color='purple')
    plt.title(f'{super_category} - Validation Metrics')
    plt.xlabel('Steps')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_plots_{super_category}.png')
    plt.close()
    print(f"Saved training plots for {super_category}")

# Train models for all categories with multiple emotions
if __name__ == "__main__":
    categories_to_train = [
        'positive',
        'negative',
        'ambiguous'
    ]
    
    for category in categories_to_train:
        print(f"\nTraining {category} model...")
        train_submodels(category)

        