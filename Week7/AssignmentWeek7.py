'''
Information Extraction Programming Exercises

Exercise 1 - RegEx Chunker
Using the RegExp tagger for guidance, write a tag pattern to cover noun phrases that contain gerunds, e.g. `the/DT receiving/VBG end/NN`, `assistant/NN managing/VBG editor/NN`. Add these patterns to the grammar, one per line. Test your work using some tagged sentences of your own devising. Note that you can provide multiple regexp patterns for identifying by using the syntax below. Note that the order of the patterns is important.


grammar = r"""
NP:
{<DT>?<JJ>*<NN>} # chunk determiners, adjectives and nouns
{<NNP>+} # chunk sequences of proper nouns
"""

Exercise 2
Use spaCy on a corpus of your choice to explore dependency parsing and named entity recogntion. Write a short summary of what you learned about your dataset.
'''

# Download if necessary
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

from nltk import RegexpParser
from nltk.corpus import treebank
import random
import pandas as pd
import matplotlib.pyplot as plt
import spacy

# Exercise 1
grammar = r"""
NP:
  {<DT><VBG><NN>}
  {<NN>*<VBG><NN>+}
  {<DT>?<JJ>*<NN>}
  {<NNP>+}
"""

cp = RegexpParser(grammar)

test_sentences = [
    [("the", "DT"), ("haunting", "VBG"), ("memory", "NN"), ("lingered", "VBD")],
    [("the", "DT"), ("blazing", "VBG"), ("sun", "NN"), ("scorched", "VBD"), ("everything", "NN")],
    [("veteran", "NN"), ("coaching", "VBG"), ("staff", "NN"), ("impressed", "VBD"), ("scouts", "NNS")],
    [("a", "DT"), ("living", "VBG"), ("legend", "NN"), ("walked", "VBD"), ("in", "IN")],
]

for sent in test_sentences:
    result = cp.parse(sent)
    print(result)



# Exercise 2
# Loading the spaCy model
# Exercise 2
nlp = spacy.load("en_core_web_sm")
random.seed(42)

file_ids = treebank.fileids()
sample_fileIds = random.sample(file_ids, 20)

sentences = []
for fid in sample_fileIds:
    sents = treebank.sents(fid)
    for sent in sents[:2]:
        text = " ".join(sent)
        if 10 < len(sent) < 40:
            sentences.append(text)
        if len(sentences) >= 10:
            break
    if len(sentences) >= 10:
        break

print()
print("SAMPLE SENTENCES FROM WSJ CORPUS")
print()
for i, s in enumerate(sentences):
    print(f"{i+1}. {s}")

print()
print("DEPENDENCY PARSING Noun Chunks")
print()
for sentence in sentences[:5]:
    doc = nlp(sentence)
    print(f"\nSentence: {sentence}")
    print("Noun chunks found:")
    for chunk in doc.noun_chunks:
        print(f"  [{chunk.text}]  <-- root: '{chunk.root.text}', dep: '{chunk.root.dep_}'")

print()
print("NAMED ENTITY RECOGNITION")
print()
frames = []
for i, sentence in enumerate(sentences):
    doc = nlp(sentence)
    print(f"Sentence: {sentence}")
    ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents if len(e.text) > 0]
    if ents:
        for text, start, end, label in ents:
            print(f"  {text!r:40} [{label}]")
        frame = pd.DataFrame(ents, columns=["Text", "Start", "Stop", "NER_Type"])
        frame["sentence_id"] = i
        frames.append(frame)
    else:
        print("  (no entities detected)")
    print()

ner = pd.concat(frames, ignore_index=True)

print("AGGREGATE ENTITY TABLE")
print(ner.to_string(index=False))

# Plot 1: NER type distribution
color_list = list("rgbkymc")
plt.figure(figsize=(8, 5))
ner.NER_Type.value_counts().plot(kind="bar", color=color_list)
plt.title("NER Type Distribution — WSJ Treebank Corpus")
plt.xlabel("NER Types")
plt.ylabel("Counts")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("ner_type_distribution.png")
plt.show()

# Plot 2: Top ORG entities (WSJ is corporate-heavy so ORG is more interesting than PERSON)
orgs = ner[ner.NER_Type == "ORG"]
if not orgs.empty:
    plt.figure(figsize=(10, 5))
    orgs.Text.value_counts()[:15].plot(kind="bar", color="darkorange")
    plt.title("Top Organizations — WSJ Treebank Corpus")
    plt.xlabel("Organization")
    plt.ylabel("Counts")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("ner_orgs.png")
    plt.show()

# Plot 3: Top PERSON entities
persons = ner[ner.NER_Type == "PERSON"]
if not persons.empty:
    plt.figure(figsize=(8, 5))
    persons.Text.value_counts().plot(kind="bar", color="steelblue")
    plt.title("PERSON Entities — WSJ Treebank Corpus")
    plt.xlabel("Names")
    plt.ylabel("Counts")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("ner_persons.png")
    plt.show()

print("""
SUMMARY
-------------------------------------------------
The WSJ Treebank corpus consists of Wall Street Journal articles from 1989, covering corporate deals, earnings reports, and financial markets.

Dependency Parsing:
    spaCy handled the formal, dense sentence structures of WSJ text well. Complex noun phrases like "two new mortgage securities-based mutual funds" 
    and "Italian state-owned holding company" were correctly grouped as single chunks. Corporate names with legal suffixes (Inc., Ltd., Corp.) were
    reliably identified as nsubj roots, reflecting the subject-heavy nature of financial reporting.

Named Entity Recognition:
    ORG dominated as the most frequent entity type, expected given the corporate deal-making focus of the corpus. spaCy confidently tagged companies 
    like McDermott International Inc. and Fannie Mae. MONEY entities such as "$ 295 million" and "$ 701 million" were consistently identified. One 
    notable artifact of the Treebank format is the presence of trace tokens like *T*-1 and *U* from the original parse annotations, which occasionally 
    confused the NER tagger. Stripping these before processing would improve results in a production setting.
    
""")