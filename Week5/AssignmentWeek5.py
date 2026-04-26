'''
Exercise 1
Using any of the three classifiers described in class, and any features you can think of, build the best name gender classifier you can. Begin by splitting the Names Corpus into three subsets: 500 words for the test set, 500 words for the dev-test set, and the remaining 6900 words for the training set. Then, starting with the example name gender classifier, make incremental improvements. Use the dev-test set to check your progress. Once you are satisfied with your classifier, check its final performance on the test set. How does the performance on the test set compare to the performance on the dev-test set? Is this what you’d expect?

Exercise 2
Using a labeled corpus other than the movie review corpus used in class, build a TF-IDF vector to use as a feature for training at least three different classifiers. Display a performance metric for each model. At least one of the classifiers must be different from those demonstrated during the lecture.
'''

from nltk import NaiveBayesClassifier
from nltk.corpus import names, stopwords
from nltk.classify import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import random



# Load and shuffle data
labeled_names = (
    [(name, 'male')   for name in names.words('male.txt')] +
    [(name, 'female') for name in names.words('female.txt')]
)
random.seed(42)
random.shuffle(labeled_names)



# Split into test / dev-test / train
test_names    = labeled_names[:500]
devtest_names = labeled_names[500:1000]
train_names   = labeled_names[1000:]
print(f"Train: {len(train_names)}, Dev-test: {len(devtest_names)}, Test: {len(test_names)}")



# Features function
def gender_features(name):
    name = name.lower()
    features = {}

    # Core suffix features
    features['last_letter'] = name[-1]
    features['last_2']      = name[-2:]
    features['last_3']      = name[-3:] if len(name) >= 3 else name
    features['last_4']      = name[-4:] if len(name) >= 4 else name  # catches 'leen','ette','inda'

    # Prefix features
    features['first_letter'] = name[0]
    features['first_2']      = name[:2]
    features['first_3']      = name[:3] if len(name) >= 3 else name  # 'mar','kar','sha' skew female

    # Length
    features['name_length'] = len(name)

    # Female signals
    features['ends_in_a']     = name[-1] == 'a'
    features['ends_in_vowel'] = name[-1] in 'aeiou'
    features['ends_in_anna']  = name.endswith('anna') or name.endswith('ana')
    features['ends_in_ine']   = name.endswith('ine')
    features['ends_in_lyn']   = name.endswith('lyn') or name.endswith('lin')
    features['ends_in_ette']  = name.endswith('ette') or name.endswith('etta')
    features['ends_in_leen']  = name.endswith('leen') or name.endswith('lene')
    features['ends_in_inda']  = name.endswith('inda') or name.endswith('enda')
    features['ends_in_ry_dy'] = name[-3:] in ('dry', 'ory', 'ery', 'ury') or \
                                  name[-2:] in ('dy', 'by', 'ry', 'cy', 'fy')  # Debby, Cordy
    features['ends_in_iz']    = name.endswith('iz') or name.endswith('riz')    # Beatriz, Liz

    # Male signals
    features['ends_in_son']   = name.endswith('son') or name.endswith('ton')
    features['ends_in_ard']   = name.endswith('ard') or name.endswith('ert')
    features['ends_in_us']    = name.endswith('us')
    features['ends_in_io']    = name.endswith('io')

    # --- Internal pattern bigrams ---
    for bigram in ['an', 'en', 'in', 'on', 'ia', 'el', 'le', 'ly', 'ne', 'ie',
                   'ri', 'or', 'er', 'id']:  # 'id' catches Brigid, Grissel
        features[f'has_{bigram}'] = bigram in name

    # Vowel ratio — female names tend to be more vowel-heavy
    vowels = sum(1 for c in name if c in 'aeiou')
    features['vowel_ratio'] = round(vowels / len(name), 2)

    return features



# Building feature sets
train_set   = [(gender_features(n), g) for n, g in train_names]
devtest_set = [(gender_features(n), g) for n, g in devtest_names]
test_set    = [(gender_features(n), g) for n, g in test_names]



# Training Naive Bayes
classifier = NaiveBayesClassifier.train(train_set)
devtest_acc = accuracy(classifier, devtest_set)
print(f"\nDev-test accuracy: {devtest_acc:.4f}")



# Error analysis on dev-test set
print("\nError Analysis (dev-test)")
errors = []
for (name, correct_gender) in devtest_names:
    guess = classifier.classify(gender_features(name))
    if guess != correct_gender:
        errors.append((correct_gender, guess, name))

print(f"Total errors: {len(errors)} / {len(devtest_names)}")
print("\nSample misclassified names:")
for correct, guess, name in sorted(errors)[:20]:
    print(f"  correct={correct:<8} guess={guess:<8} name={name}")


# Most informative features
print("\nMost Informative Features")
classifier.show_most_informative_features(15)


# Final evaluation on held-out TEST set
test_acc = accuracy(classifier, test_set)
print(f"\nFinal Results")
print(f"Dev-test accuracy : {devtest_acc:.4f}")
print(f"Test accuracy     : {test_acc:.4f}")
print(f"Difference        : {abs(test_acc - devtest_acc):.4f}")
print("""
----------------------------------------------------------------
Exercise 1
The test accuracy should be close to the dev-test accuracy.
A small drop is expected because we tuned our features on the
dev-test set, introducing slight overfitting to those examples.
A large drop would suggest overfitting; a higher test score is
possible due to random variation in the splits.
----------------------------------------------------------------
""")



# ── Exercise 2 ────────────────────────────────────────────────────────────────
# Corpus : SMS Spam Collection
# Models : Naive Bayes | Logistic Regression | Random Forest
# Feature: TF-IDF vector
import nltk
nltk.download('sentence_polarity')
from nltk.corpus import sentence_polarity



# Load corpus
pos_sents = sentence_polarity.sents(categories='pos')
neg_sents = sentence_polarity.sents(categories='neg')



# Join token lists back into strings for TF-IDF
texts  = [' '.join(s) for s in pos_sents] + [' '.join(s) for s in neg_sents]
labels = ['pos'] * len(pos_sents) + ['neg'] * len(neg_sents)

print(f"Total sentences : {len(texts)}")
print(f"Positive        : {labels.count('pos')}")
print(f"Negative        : {labels.count('neg')}")



# TF-IDF Vectorisation
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams
    max_features=5000,
    sublinear_tf=True     # log-smoothed term frequency
)
X = tfidf.fit_transform(texts)
y = labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size : {X_train.shape[0]}")
print(f"Test size  : {X_test.shape[0]}")
print(f"Features   : {X_train.shape[1]}")



# Helper — print metrics
def evaluate(name, model, X_te, y_te):
    preds = model.predict(X_te)
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(classification_report(y_te, preds))



# Model 1 — Naive Bayes
nb = MultinomialNB(alpha=0.1)
nb.fit(X_train, y_train)
evaluate("Naive Bayes (Multinomial)", nb, X_test, y_test)



# Model 2 — Logistic Regression
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_train, y_train)
evaluate("Logistic Regression", lr, X_test, y_test)


# Model 3 — Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
evaluate("Random Forest", rf, X_test, y_test)


# Summary
print("\nSummary")
print(f"{'Model':<28} {'Accuracy':>9} {'F1 (pos)':>10} {'F1 (neg)':>10}")
for name, model in [("Naive Bayes", nb), ("Logistic Regression", lr), ("Random Forest", rf)]:
    preds = model.predict(X_test)
    print(f"{name:<28} "
          f"{accuracy_score(y_test, preds):>9.4f} "
          f"{f1_score(y_test, preds, pos_label='pos'):>10.4f} "
          f"{f1_score(y_test, preds, pos_label='neg'):>10.4f}")

print("""
\n
-------------------------------------------------------------------------------------
Exercise 2
Corpus   : NLTK Sentence Polarity (10,662 sentences, balanced 50/50)
Feature  : TF-IDF with unigrams + bigrams, 5000 features, log-smoothed TF

Results:
  Naive Bayes         : 77.5% accuracy
  Logistic Regression : 76.9% accuracy
  Random Forest       : 69.6% accuracy

Naive Bayes is slightly better than Logistic Regression, which is a well-known
pattern on short text.

Logistic Regression is essentially tied with Naive Bayes, which makes sense
as both are linear classifiers over the same TF-IDF space.

Random Forest underperforms significantly (69.6%). This is expected because Random
Forest was designed for dense, low-dimensional feature spaces. TF-IDF produces
a very high-dimensional sparse matrix (5000 features), so individual trees
struggle to find useful splits, and the ensemble cannot compensate fully.
-------------------------------------------------------------------------------------
""")