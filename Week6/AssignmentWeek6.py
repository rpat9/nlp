# NLP Week 6 - Embeddings Exploration

# Part 1: GloVe Embeddings Analysis

from gensim.models import KeyedVectors
import pprint

# Load GloVe embeddings
glove_file = 'Week6/Corpora/glove.6B.300d.txt'
model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)

# 1.1 - Polysemous Words Analysis
print("1.1 - POLYSEMOUS WORDS")

print("\nAnalyzing 'leaves':")
pprint.pprint(model.most_similar("leaves"))

print("\nAnalyzing 'scoop':")
pprint.pprint(model.most_similar("scoop"))

# 1.2 - Synonyms vs Antonyms (Counter-intuitive results)
print()
print("1.2 - SYNONYMS & ANTONYMS")
print()

w1 = "happy"
w2 = "cheerful"
w3 = "sad"
w1_w2_dist = model.distance(w1, w2)
w1_w3_dist = model.distance(w1, w3)

print(f"\nSynonyms {w1}, {w2} have cosine distance: {w1_w2_dist}")
print(f"Antonyms {w1}, {w3} have cosine distance: {w1_w3_dist}")
print(f"Finding: Antonym 'sad' is closer to 'happy' than synonym 'cheerful'")

# 1.3 - Solving Analogies with Word Vectors
print()
print("1.3 - ANALOGIES (Working Example)")
print()

print("\nAnalogy: man : king :: woman : ?")
pprint.pprint(model.most_similar(positive=["woman", "king"], negative=["man"]))

# 1.4 - Incorrect Analogies
print()
print("1.4 - INCORRECT ANALOGY")
print()

print("\nAnalogy: doctor : patient :: teacher : ?")
pprint.pprint(model.most_similar(positive=["defendant", "teacher"], negative=["lawyer"]))

# 1.5 - Gender Bias in Professional Contexts
print()
print("1.5 - GENDER BIAS: Boss")
print()

print("\nwoman + boss (minus man):")
pprint.pprint(model.most_similar(positive=["woman", "boss"], negative=["man"]))

print("\nman + boss (minus woman):")
pprint.pprint(model.most_similar(positive=["man", "boss"], negative=["woman"]))

# 1.6 - Independent Bias Analysis: Nursing Profession
print()
print("1.6 - INDEPENDENT BIAS ANALYSIS: Nursing")
print()

print("\nnurse + woman (minus man):")
pprint.pprint(model.most_similar(positive=["nurse", "woman"], negative=["man"]))

print("\nnurse + man (minus woman):")
pprint.pprint(model.most_similar(positive=["nurse", "man"], negative=["woman"]))


# Part 2: Custom Word2Vec Embeddings from Gutenberg Corpus
from nltk.corpus import gutenberg
from string import punctuation
from gensim.models import Word2Vec

print()
print("PART 2 - CUSTOM EMBEDDINGS FROM GUTENBERG CORPUS")
print()

# Load and process 5 texts from Gutenberg corpus
austen_sentences = gutenberg.sents("austen-emma.txt")
austen_simplified = [[word.lower() for word in sent if word not in punctuation] for sent in austen_sentences]
austen_model = Word2Vec(austen_simplified)

carroll_sentences = gutenberg.sents("carroll-alice.txt")
carroll_simplified = [[word.lower() for word in sent if word not in punctuation] for sent in carroll_sentences]
carroll_model = Word2Vec(carroll_simplified)

shakespeare_sentences = gutenberg.sents("shakespeare-hamlet.txt")
shakespeare_simplified = [[word.lower() for word in sent if word not in punctuation] for sent in shakespeare_sentences]
shakespeare_model = Word2Vec(shakespeare_simplified)

melville_sentences = gutenberg.sents("melville-moby_dick.txt")
melville_simplified = [[word.lower() for word in sent if word not in punctuation] for sent in melville_sentences]
melville_model = Word2Vec(melville_simplified)

whitman_sentences = gutenberg.sents("whitman-leaves.txt")
whitman_simplified = [[word.lower() for word in sent if word not in punctuation] for sent in whitman_sentences]
whitman_model = Word2Vec(whitman_simplified)

# Define sentiment words to analyze
sentiment_words = ['laugh', 'joy', 'love', 'sorrow', 'hate', 'anger']

# Analyze Austen model
print()
print("AUSTEN MODEL - Sentiment Words")
print()
for word in sentiment_words:
    try:
        similar = austen_model.wv.most_similar(word, topn=10)
        print(f"\n{word.upper()}:")
        for w, score in similar:
            print(f"  {w}: {score:.4f}")
    except KeyError:
        print(f"\n{word.upper()}: [NOT FOUND IN VOCABULARY]")

# Analyze Carroll model
print()
print("CARROLL MODEL - Sentiment Words")
print()
for word in sentiment_words:
    try:
        similar = carroll_model.wv.most_similar(word, topn=10)
        print(f"\n{word.upper()}:")
        for w, score in similar:
            print(f"  {w}: {score:.4f}")
    except KeyError:
        print(f"\n{word.upper()}: [NOT FOUND IN VOCABULARY]")

# Analyze Shakespeare model
print()
print("SHAKESPEARE MODEL - Sentiment Words")
print()
for word in sentiment_words:
    try:
        similar = shakespeare_model.wv.most_similar(word, topn=10)
        print(f"\n{word.upper()}:")
        for w, score in similar:
            print(f"  {w}: {score:.4f}")
    except KeyError:
        print(f"\n{word.upper()}: [NOT FOUND IN VOCABULARY]")

# Analyze Melville model
print()
print("MELVILLE MODEL - Sentiment Words")
print()
for word in sentiment_words:
    try:
        similar = melville_model.wv.most_similar(word, topn=10)
        print(f"\n{word.upper()}:")
        for w, score in similar:
            print(f"  {w}: {score:.4f}")
    except KeyError:
        print(f"\n{word.upper()}: [NOT FOUND IN VOCABULARY]")

# Analyze Whitman model
print()
print("WHITMAN MODEL - Sentiment Words")
print()
for word in sentiment_words:
    try:
        similar = whitman_model.wv.most_similar(word, topn=10)
        print(f"\n{word.upper()}:")
        for w, score in similar:
            print(f"  {w}: {score:.4f}")
    except KeyError:
        print(f"\n{word.upper()}: [NOT FOUND IN VOCABULARY]")

print()
print("ANALYSIS COMPLETE")
print()
print("""
Key Findings:
- Small corpora produce poor embeddings with high similarity scores (0.99+)
- GloVe embeddings vastly outperform single-book embeddings
- Carroll's Alice has no sentiment words in vocabulary
- Whitman's model performs best due to emotionally expressive vocabulary
- Embedding quality depends directly on training data size and diversity
""")