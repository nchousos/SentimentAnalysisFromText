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
```

Αυτό θα δημιουργήσει τα αρχεία:

train_cleaned.csv

validation_cleaned.csv

test_cleaned.csv

### 2. Εκπαίδευση Μοντέλων

Τρέξτε διαδοχικά τα εξής scripts που βρίσκονται στον φάκελο Training Models/:

```bash
python Training Models/TrainingSuperCat.py
python Training Models/TrainingNeut.py
python Training Models/TrainingEmotions.py
```

Αυτά θα δημιουργήσουν τους αντίστοιχους φακέλους με τα εκπαιδευμένα μοντέλα και τους tokenizers για:

Supercategory (Θετικά / Αρνητικά / Ουδέτερα)

Ουδέτερα 

Ειδικά Συναισθήματα ανά Supercategory

### 3. Αξιολόγηση Ιεραρχικού Μοντέλου

Για να ελέγξετε την απόδοση του συστήματος στο test set, τρέξτε το παρακάτω script:

```bash
python Evaluation on Test Dataset/EvaluationTestDataset.py
```

Αυτό φορτώνει το test_cleaned.csv, αξιολογεί την ακρίβεια του ιεραρχικού μοντέλου και εμφανίζει τα αποτελέσματα.

### 4. Εκκίνηση Chatbot Εφαρμογής

Για να δοκιμάσετε το chatbot:

1.Τρέξτε την Flask εφαρμογή:

```bash
python ChatBot App/app.py
```

2.Στη συνέχεια, ανοίξτε χειροκίνητα το αρχείο ChatBot App/website.html με τον browser σας (κατά προτίμηση Google Chrome).

# Σημειώσεις

Η εκπαίδευση γίνεται σε τρία στάδια (supercategory, ουδέτερα, ειδικά συναισθήματα) ώστε να επιτυγχάνεται ιεραρχική ταξινόμηση.
Το σύστημα εμφανίζει ιδιαίτερη ακρίβεια στα θετικά συναισθήματα.
Δυσκολεύεται στην αναγνώριση ειρωνείας και σε κείμενα με πολλαπλά συναισθήματα.
Δημιουργήθηκε ως μέρος πτυχιακής εργασίας – University of East London
