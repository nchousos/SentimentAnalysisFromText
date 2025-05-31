# Emotion Classification using the GoEmotions Dataset

Αυτό το έργο περιλαμβάνει ένα σύστημα ανάλυσης συναισθημάτων βασισμένο στο GoEmotions dataset της Google, με στόχο την ιεραρχική ταξινόμηση συναισθημάτων και την ενσωμάτωσή του σε ένα απλό chatbot εφαρμογής.

---

## 📁 Δομή του Έργου

- `Dataset/` – Περιέχει τα αρχικά αρχεία `.parquet` του GoEmotions dataset και το script καθαρισμού δεδομένων.
- `Training Models/` – Περιλαμβάνει scripts για την εκπαίδευση των μοντέλων.
- `Evaluation on Test Dataset/` – Περιέχει script για την αξιολόγηση του τελικού μοντέλου.
- `ChatBot App/` – Περιλαμβάνει το Flask-based chatbot και το HTML interface.

---

## ✅ Οδηγίες Εκτέλεσης

### 1. Προετοιμασία Δεδομένων

Βεβαιωθείτε ότι τα αρχεία:
- `train.parquet`
- `validation.parquet`
- `test.parquet`

βρίσκονται στον φάκελο `Dataset/`.

Στη συνέχεια, τρέξτε το παρακάτω script για να καθαρίσετε τα δεδομένα και να δημιουργήσετε τα αντίστοιχα `.csv` αρχεία:

```bash
python Dataset/DatasetCleanUp.py
