import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# Load datasets
train_df = pd.read_csv('train_cleaned.csv')
val_df = pd.read_csv('validation_cleaned.csv')

def clean_text_column(df):
    df['text_clean'] = (
        df['text_clean']
        .astype(str) 
        .str.strip()  
        .replace(r'^\s*$', '[[EMPTY_TEXT]]', regex=True)  
    )
    return df

train_df = clean_text_column(train_df)
val_df = clean_text_column(val_df)


print(train_df.head())

train_df['label'] = train_df['super_categories'].apply(lambda x: 1 if 'neutral' in x else 0)
val_df['label'] = val_df['super_categories'].apply(lambda x: 1 if 'neutral' in x else 0)

train_dataset = Dataset.from_pandas(train_df[['text_clean', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text_clean', 'label']])

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize function
def tokenize_function(examples):
    
    texts = examples['text_clean']
    if isinstance(texts, list):
        texts = [str(text) for text in texts]  
    else:
        texts = str(texts) 
    return tokenizer(texts, truncation=True, padding='max_length', max_length=128)


# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define compute metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./bert_neutral_model',
    eval_strategy="steps",
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_steps=100,
    save_strategy='steps',
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)

# Save the model and tokenizer
model.save_pretrained('./bert_neutral_model')
tokenizer.save_pretrained('./bert_neutral_model')
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
plt.savefig('training_plots_neutral.png')
plt.close()

print("Plots saved as training_plots.png")