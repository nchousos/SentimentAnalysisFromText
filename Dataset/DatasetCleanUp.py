import pandas as pd
import numpy as np
import re

contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", 
                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 
                       "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
                       "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", 
                       "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", "i'm": "i am", 
                       "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                       "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 
                       "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", 
                       "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                       "so've": "so have", "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",
                       "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
                       "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                       "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", 
                       "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                       "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", 
                       "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", 
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                       "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have",
                       "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'u.s':'america', 'e.g':'for example'}

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-",
                 "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 
                 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!':' '}



emoji_emotion_mapping = {
    "expressionless-face": "neutral",
    "face-with-steam-from-nose": "angry",
    "persevering-face": "frustrated",
    "weary-face": "exhausted",
    "winking-face": "playful",
    "slightly-frowning-face": "sad",
    "woman-facepalming-medium": "disappointed",
    "man-facepalming": "disappointed",
    "down-face": "sad",
    "cowboy-hat-face": "confident",
    "worried-face": "worried",
    "face-with-raised-eyebrow": "skeptical",
    "face-blowing-a-kiss": "affectionate",
    "grinning-face": "happy",
    "smiling-face": "happy",
    "face-with-open-mouth": "surprised",
    "zany-face": "excited",
    "unamused-face": "annoyed",
    "face-vomiting": "disgusted",
    "face-with-hand-over-mouth": "shocked",
    "fearful-face": "fear",
    "drooling-face": "desire",
    "sleepy-face": "sleepy",
    "sleeping-face": "sleepy",
    "anxious-face-with-sweat": "nervous",
    "face-with-rolling-eyes": "annoyed",
    "winking-face-with-tongue": "playful",
    "flushed-face": "embarrassed",
    "grinning-face-with-smiling-eyes": "joyful",
    "slightly-smiling-face": "content",
    "grinning-face-with-big-eyes": "happy",
    "nauseated-face": "disgusted",
    "loudly-crying-face": "very sad",
    "thinking-face": "thinking",
    "smiling-face-with-halo": "innocent",
    "shushing-face": "quiet",
    "kissing-face": "affectionate",
    "smiling-face-with-sunglasses": "cool",
    "angry-face": "angry",
    "face-with-tears-of-joy": "joy",
    "nerd-face": "intelligent",
    "confounded-face": "frustrated",
    "frowning-face": "sad",
    "sad-but-relieved-face": "relieved",
    "face-with-crossed": "uncertain",
    "enraged-face": "furious",
    "smiling-face-with-smiling-eyes": "joyful",
    "anguished-face": "distressed",
    "smiling-face-with-heart": "love",
    "grinning-face-with-sweat": "nervous",
    "confused-face": "confused",
    "tired-face": "tired",
    "man-facepalming-light-skin-tone": "frustrated",
    "pleading-face": "begging",
    "downcast-face-with-sweat": "worried",
    "pensive-face": "thoughtful",
    "woman-facepalming-light-skin-tone": "frustrated",
    "frowning-face-with-open-mouth": "shocked",
    "face-with-head": "confused",
    "person-facepalming": "disappointed",
    "squinting-face-with-tongue": "playful",
    "smirking-face": "mischievous",
    "face-with-tongue": "silly",
    "neutral-face": "neutral",
    "disappointed-face": "disappointed",
    "face-screaming-in-fear": "terrified",
    "face-savoring-food": "satisfied",
    "grinning-squinting-face": "happy",
    "crying-face": "sad",
    "grimacing-face": "awkward",
    "smiling-face-with-open-hands": "welcoming",
    "beaming-face-with-smiling-eyes": "delighted",
    "smiling-face-with-hearts": "love"
}

emoticon_mapping = {
    ":)": "happy",
    ":‑)": "happy",
    ":(": "sad",
    ":‑(": "sad",
    ":D": "laughing",
    ":'‑)": "crying",
    ":'(": "crying",
    ":‑/": "confused",
    ":'‑(": "crying",
    ">:‑(": "angry",
    ">:(": "angry",
    ";)": "wink",
    "<3": "love",
    "</3": "heartbroken",
    ":P": "playful",
    ":‑P": "playful",
    ":-|": "neutral",
    ":-O": "surprised",
    ":O": "surprised",
    ":-*": "kiss",
    ":*": "kiss",
    ">:)": "evil",
    ">:-)": "evil",
}

df = pd.read_parquet('./train-00000-of-00001.parquet')
df_val = pd.read_parquet('./validation-00000-of-00001.parquet')
df_test = pd.read_parquet('./test-00000-of-00001.parquet')

#print(df.head())

label_mapping = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral"
}

"""super_category_map = {
    'high_valence_high_arousal': ['admiration', 'amusement', 'excitement', 'joy', 'love',  'desire'],
    'high_valence_low_arousal': ['gratitude','approval', 'optimism', 'caring'],
    'low_valence_high_arousal': ['anger', 'annoyance', 'fear', 'disgust', 'disapproval'],
    'low_valence_low_arousal': ['sadness', 'disappointment', 'remorse', 'embarrassment'],
    'neutral': ['neutral'],
    'surprise': ['surprise'],
    'confusion': ['confusion', 'curiosity', 'realization'],
}"""

super_category_map = { 
    'positive': ['admiration', 'amusement', 'approval', 'caring', 'gratitude', 'excited_joyful', 'love', 'optimism'],
    'negative': ['irritation', 'disapproval', 'disgust', 'disappointment', 'fear', 'remorse', 'sadness', 'embarrassment'],
    'ambiguous': ['confusion', 'curiosity', 'desire' ,'surprise'],
    'neutral':['neutral']
}
"""super_category_map = { 
    'positive': ['admiration', 'amusement', 'approval', 'caring', 'gratitude', 'excitement','joy', 'love', 'optimism'],
    'negative': ['anger','annoyance', 'disapproval', 'disgust', 'disappointment', 'fear', 'remorse', 'sadness', 'embarrassment'],
    'ambiguous': ['confusion', 'curiosity', 'desire' ,'surprise'],
    'neutral':['neutral']
}"""
conflicting_pairs = [
    ('admiration', 'annoyance'),
    ('admiration', 'disapproval'),
    ('admiration', 'anger'),
    ('annoyance', 'gratitude'),
    ('annoyance', 'approval'),
    ('approval', 'disapproval'),
    ('admiration', 'curiosity'),
    ('disapproval', 'gratitude'),
    ('admiration', 'sadness'),
    ('annoyance', 'love')
]

emotion_replace_map = {
    'anger': 'irritation',
    'annoyance': 'irritation',
    #'confusion': 'confused_curious',
    #'curiosity': 'confused_curious',
    'excitement': 'excited_joyful',
    'joy': 'excited_joyful',
}



df['emotions'] = df['labels'].apply(lambda x: [label_mapping[i] for i in x])
df_val['emotions'] = df_val['labels'].apply(lambda x: [label_mapping[i] for i in x])
df_test['emotions'] = df_test['labels'].apply(lambda x: [label_mapping[i] for i in x])



emotions_to_drop = {"""'nervousness', 'pride', 'grief', 'relief','realization'"""}


def row_contains_dropped_emotion(emotions_list):
    return any(e in emotions_to_drop for e in emotions_list)


df = df[~df['emotions'].apply(row_contains_dropped_emotion)].reset_index(drop=True)
df_val = df_val[~df_val['emotions'].apply(row_contains_dropped_emotion)].reset_index(drop=True)
df_test = df_test[~df_test['emotions'].apply(row_contains_dropped_emotion)].reset_index(drop=True)

def filter_and_map(emotions_list):
    super_categories = []
    for emotion in emotions_list:
        for super_cat, emotions in super_category_map.items():
            if emotion in emotions:
                super_categories.append(super_cat)
                break  
    return sorted(list(set(super_categories)))

df['super_categories'] = df['emotions'].apply(filter_and_map)
df_val['super_categories'] = df_val['emotions'].apply(filter_and_map)
df_test['super_categories'] = df_test['emotions'].apply(filter_and_map)


df = df[df['super_categories'].apply(len) > 0].reset_index(drop=True)
df_val = df_val[df_val['super_categories'].apply(len) > 0].reset_index(drop=True)
df_test = df_test[df_test['super_categories'].apply(len) > 0].reset_index(drop=True)


print("\nFiltered and Mapped Categories:")
print(df[['text', 'emotions', 'super_categories']].head(3))
print("\nCategory Distribution:")
print(pd.Series(np.concatenate(df['super_categories'])).value_counts())




def replace_emojis_with_emotions(text, emoji_mapping):
    matches = re.findall(r'\b[a-zA-Z]+(?:_[a-zA-Z]+)+\b', text)  
    for match in matches:
        match_hyphen = match.replace("_", "-") 
        if match_hyphen in emoji_mapping:
            text = text.replace(match, emoji_mapping[match_hyphen])  
    return text

def clean_text(text, contraction_mapping, punct_mapping, emoji_mapping):

    text = text.replace("â€™", "'")
    
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    
    
    """for emoticon, replacement in emoticon_mapping.items():
        text = text.replace(emoticon, f' {replacement} ')
    
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = replace_emojis_with_emotions(text, emoji_mapping) """
    
    for word, replacement in contraction_mapping.items():
        text = re.sub(r"\b" + word + r"\b", replacement, text)

    
    for punc, repl in punct_mapping.items():
       text = text.replace(punc, repl)
    
   
    text = ''.join([i if ord(i) < 128 else ' ' for i in text])
    
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

df['text_clean'] = df['text'].apply(lambda x: clean_text(x, contraction_mapping, punct_mapping, emoji_emotion_mapping))  # Apply to training dataset
df_val['text_clean'] = df_val['text'].apply(lambda x: clean_text(x, contraction_mapping, punct_mapping, emoji_emotion_mapping))  # Apply to validation dataset
df_test['text_clean'] = df_test['text'].apply(lambda x: clean_text(x, contraction_mapping, punct_mapping, emoji_emotion_mapping))  # Apply to validation dataset

def replace_emotions(emotions, replace_map):
    replaced = [replace_map.get(e, e) for e in emotions]  
    return list(set(replaced))  

df['emotions'] = df['emotions'].apply(lambda x: replace_emotions(x, emotion_replace_map))
df_val['emotions'] = df_val['emotions'].apply(lambda x: replace_emotions(x, emotion_replace_map))
df_test['emotions'] = df_test['emotions'].apply(lambda x: replace_emotions(x, emotion_replace_map))

def has_conflicting_emotions(emotions, conflict_list):
    for e1, e2 in conflict_list:
        if e1 in emotions and e2 in emotions:
            return True
    return False

def filter_multi_conflicted_rows(df, conflict_list):
    df['num_labels'] = df['emotions'].apply(len)
    
    filtered_df = df[~(
        (df['num_labels'] == 3) | 
        (df['num_labels'] == 4) | 
        (df['emotions'].apply(lambda x: has_conflicting_emotions(x, conflict_list)))  
    )].drop(columns=['num_labels']).reset_index(drop=True)
    
    removed_count = len(df) - len(filtered_df)
    print(f"\nΑφαιρέθηκαν {removed_count} δείγματα με 2 ή περισσότερα αντικρουόμενα συναισθήματα ή 4 συναισθήματα.")
    
    return filtered_df


df = filter_multi_conflicted_rows(df, conflicting_pairs)
df_val = filter_multi_conflicted_rows(df_val, conflicting_pairs)
df_test = filter_multi_conflicted_rows(df_test, conflicting_pairs)


print(df.head())
print("\nCategory Distribution:")
print(pd.Series(np.concatenate(df['super_categories'])).value_counts())
df.to_csv('./train_cleaned.csv', index=False)
df_val.to_csv('./validation_cleaned.csv', index=False)
df_test.to_csv('./test_cleaned.csv', index=False)