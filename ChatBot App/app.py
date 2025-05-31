from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from flask_cors import CORS
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from EvaluationTestDataset import detect_emotion  
from googletrans import Translator  
import csv
from datetime import datetime
app = Flask(__name__)
CORS(app)

chat_history = []


model_name = "facebook/blenderbot-400M-distill"
chat_tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
chat_model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Load the models and tokenizers
supercat_model = BertForSequenceClassification.from_pretrained('./optimized_emotion_classifier4')
supercat_tokenizer = BertTokenizer.from_pretrained('./optimized_emotion_classifier4')

neutral_model = BertForSequenceClassification.from_pretrained('./bert_neutral_model')
neutral_tokenizer = BertTokenizer.from_pretrained('./bert_neutral_model')

specific_models = {}
specific_tokenizers = {}
for super_category in ['positive', 'negative', 'ambiguous']:
    specific_models[super_category] = BertForSequenceClassification.from_pretrained(f'./specific_models/{super_category}')
    specific_tokenizers[super_category] = BertTokenizer.from_pretrained(f'./specific_models/{super_category}')

supercat_labels = ['positive', 'negative', 'ambiguous']
neutral_labels = ['non-neutral', 'neutral']
specific_labels = {
    'positive': ['admiration','amusement','approval','caring','gratitude','excited_joyful','love','optimism'],
    'negative': ['irritation','disapproval','disgust','disappointment','fear','remorse','sadness','embarrassment'],
    'ambiguous': ['confusion','curiosity','desire','surprise']
}


def predict(text, model, tokenizer, labels):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    softmax_probs = torch.softmax(logits, dim=1)[0]
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return labels[predicted_class_id], softmax_probs[predicted_class_id].item(), softmax_probs


@app.route('/predict', methods=['POST'])
def get_emotion():
    data = request.get_json()
    text = data['text']

    supercategory, emotions_list = detect_emotion(text)
    if emotions_list:
        emotions_list.sort(key=lambda x: x[1], reverse=True)
        top_emotion, top_confidence = emotions_list[0]
        detected_emotions = [emotion for emotion, conf in emotions_list]
    else:
        top_emotion = supercategory
        top_confidence = 1.0
        detected_emotions = [supercategory]
        emotions_list = [(supercategory, 1.0)]

    print(f"[Supercategory: {supercategory}, Top Emotion: {top_emotion}, Confidence: {top_confidence:.2f}, All Emotions: {emotions_list}]")


    full_input = ""
    for turn in chat_history[-2:]:
        full_input += f"User: {turn['user']}\nBot: {turn['bot']}\n"
    full_input += f"User (feeling {top_emotion}): {text}"

    inputs = chat_tokenizer(full_input, return_tensors="pt", truncation=True, max_length=128)
 
    reply_ids = chat_model.generate(**inputs, max_length=60)
    response = chat_tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    chat_history.append({
        'user': text,
        'supercategory': supercategory,
        'specific_emotion': top_emotion,
        'confidence': top_confidence,
        'all_emotions': emotions_list, 
        'bot': response
    })


    return jsonify({
        'supercategory': supercategory,
        'specific_emotion': top_emotion,
        'confidence': top_confidence,
        'all_emotions': [{'emotion': emotion, 'confidence': conf} for emotion, conf in emotions_list],
        'bot_response': response
    })



@app.route('/history', methods=['GET'])
def get_history():
    print(f"Chat history length: {len(chat_history)}")
    return jsonify(chat_history)


@app.route('/clear_history', methods=['POST'])
def clear_history():
    chat_history.clear()
    return jsonify({'status': 'cleared'})
translator = Translator()
@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    text = data['text']
    
    try:
        translation = translator.translate(text, dest='el')
        return jsonify({
            'original': text,
            'translated': translation.text,
            'src_lang': translation.src
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    feedback = {
        'timestamp': datetime.now().isoformat(),
        'original_text': data.get('text'),
        'detected_emotion': data.get('detected_emotion'),
        'user_feedback': data.get('user_feedback'),
        'comment': data.get('comment', '')
    }
    

    with open('feedback.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=feedback.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(feedback)
    
    return jsonify({'status': 'success'})
if __name__ == '__main__':
    app.run(debug=True)
