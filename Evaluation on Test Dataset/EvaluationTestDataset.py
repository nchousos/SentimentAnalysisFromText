import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import ast
from sklearn.metrics import f1_score, classification_report
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer


# Load models and tokenizers
supercat_model = BertForSequenceClassification.from_pretrained('./optimized_emotion_classifier4')
supercat_tokenizer = BertTokenizer.from_pretrained('./optimized_emotion_classifier4')
neutral_model = BertForSequenceClassification.from_pretrained('./bert_neutral_model')
neutral_tokenizer = BertTokenizer.from_pretrained('./bert_neutral_model')
specific_models = {}
specific_tokenizers = {}
for super_category in ['positive', 'negative', 'ambiguous']:
    specific_models[super_category] = BertForSequenceClassification.from_pretrained(f'./specific_models/{super_category}')
    specific_tokenizers[super_category] = BertTokenizer.from_pretrained(f'./specific_models/{super_category}')

# Label mappings
supercat_labels = ['positive', 'negative', 'ambiguous', 'neutral']
neutral_labels = ['non-neutral', 'neutral']
specific_labels = {
    'positive': ['admiration','amusement','approval','caring','gratitude','excited_joyful','love','optimism'],
    'negative': ['irritation','disapproval','disgust','disappointment','fear','remorse','sadness','embarrassment'],
    'ambiguous': ['confusion','curiosity','desire','surprise']
}

# Predict functions
def predict(text, model, tokenizer, labels):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad(): outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0]
    idx = torch.argmax(logits, dim=1).item()
    return labels[idx], probs[idx].item(), probs

def predict_specific(text, model, tokenizer, labels, threshold=0.5):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad(): outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze().detach().cpu().numpy()
    import numpy as np; probs = np.atleast_1d(probs)
    return [(labels[i], float(probs[i])) for i in range(len(labels)) if probs[i] > threshold]

# Label parsing & cleaning
def parse_label(label):
    if pd.isnull(label): return None
    if isinstance(label, list): return label
    label = label.strip()
    if label.startswith('[') and label.endswith(']'):
        try: return ast.literal_eval(label)
        except: return [label.strip("[]").replace("'","")]
    return [label]

def clean_text(text):
    if not isinstance(text, str): return None
    text = re.sub(r'\s+',' ', text.lower().strip())
    return text if len(text)>=3 else None

# Hybrid detection
def detect_emotion(text, neutral_threshold=0.6, super_threshold=0.5):
    # Neutral check
    neutral_pred,neutral_conf,_ = predict(text, neutral_model, neutral_tokenizer, neutral_labels)
    if neutral_pred=='neutral' and neutral_conf>neutral_threshold:
        return 'neutral', []
    # Supercategory
    super_category_pred, super_category_conf, super_category_probs = predict(
        text, supercat_model, supercat_tokenizer, supercat_labels[:-1]
    )
    # handle neutral fallback
    top2 = torch.topk(super_category_probs,2)[0]
    if super_category_pred=='neutral' or (top2[0]-top2[1]<0.1) or super_category_conf<super_threshold:
        return 'neutral', []
    # Specific
    specific = predict_specific(
        text, specific_models[super_category_pred], specific_tokenizers[super_category_pred], specific_labels[super_category_pred], threshold=0.3
    )
    return super_category_pred, specific

if __name__ == '__main__':
    # Load test data
    data = pd.read_csv('test_cleaned.csv')
    true_supercats=[]
    pred_supercats=[]
    true_emotions_all=[]
    pred_emotions_all=[]
    misclassifications=[]

    for idx,row in data.iterrows():
        text=clean_text(row['text_clean'])
        if not text: continue
        sup=parse_label(row['super_categories'])
        emo=parse_label(row['emotions'])
        if not sup or not emo: continue
        true_sc=sup[0]
        pred_sc,preds=detect_emotion(text)
        true_supercats.append(true_sc)
        pred_supercats.append(pred_sc)
        if true_sc!='neutral':
            true_emotions_all.append(emo)
            pred_list=[e for e,_ in preds]
            pred_emotions_all.append(pred_list)
        # log misclassifications
        if (pred_sc!=true_sc) or (true_sc!='neutral' and not any(e in emo for e in [p for p,_ in preds])):
            if len(misclassifications)<50:
                misclassifications.append({
                    'index': idx,
                    'text': text[:100],
                    'true_super': true_sc,
                    'pred_super': pred_sc,
                    'true_emotions': emo,
                    'pred_emotions': [e for e,_ in preds]
                })

    # Metrics
    print('Super Category Report:')
    print(classification_report(true_supercats, pred_supercats, labels=supercat_labels, digits=3))

    print('\nSpecific Emotions Report:')
    mlb=MultiLabelBinarizer(classes=sum(specific_labels.values(),[]))
    y_true=mlb.fit_transform(true_emotions_all)
    y_pred=mlb.transform(pred_emotions_all)
    print(classification_report(y_true,y_pred,target_names=mlb.classes_,digits=3))

    # Print misclassifications
    print(f"\nSample Misclassifications (up to 50):")
    for m in misclassifications:
        print(m)
 

    conf_matrix_supercats = confusion_matrix(true_supercats, pred_supercats, labels=supercat_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_supercats, annot=True, fmt="d", cmap="Blues", xticklabels=supercat_labels, yticklabels=supercat_labels)
    plt.title("Confusion Matrix for Super Categories")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    from sklearn.metrics import multilabel_confusion_matrix
    import numpy as np

    # Calculate multi-label confusion matrix
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    # Create emotion-by-emotion matrix showing true vs predicted co-occurrences
    num_labels = len(mlb.classes_)
    confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)

    for i in range(len(y_true)):
        true_labels = np.where(y_true[i] == 1)[0]
        pred_labels = np.where(y_pred[i] == 1)[0]
        for tl in true_labels:
            for pl in pred_labels:
                confusion_matrix[tl, pl] += 1

    # Plotting
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=mlb.classes_,
        yticklabels=mlb.classes_,
        mask=confusion_matrix == 0  # Hide cells with 0 values
    )
    plt.title("Emotion Prediction Co-occurrence Matrix")
    plt.xlabel("Predicted Emotions")
    plt.ylabel("True Emotions")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

