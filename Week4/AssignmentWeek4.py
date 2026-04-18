'''
Document Similarity
Find a corpus with documents that you can group in some way. It could be a labeled corpus, or a corpus with texts from various time periods or texts with various genres. Preprocess the data, then create a bag-of-words vector and a TF-IDf vector for the documents in the corpus.

Now perform the following tasks to see what insight you can glean from the bag-of-words, TF-IDF scores and similarity measurements.

Select one document in the corpus and find the document that is the most similar using cosine similarity with TF-IDF scores.
Sum the bag-of-words vectors for the group and print out the top 20 terms for the summed vectors. These are the most frequent words for the group.
Sum the TF-IDf vectors for each group and print out the top 20 terms in the summed vector, and their weights. These are the most important words for the group.
'''
import pandas as pd
import numpy as np
import re
import contractions
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the Gutenberg corpus
books = []
for fileid in gutenberg.fileids():
    books.append((fileid, gutenberg.raw(fileid)))

books_df = pd.DataFrame(books, columns=['filename', 'text'])

# Create author groups
austen_books = books_df[books_df['filename'].str.contains('austen')]
shakespeare_books = books_df[books_df['filename'].str.contains('shakespeare')]
chesterton_books = books_df[books_df['filename'].str.contains('chesterton')]
other_books = books_df[~books_df['filename'].str.contains('austen|shakespeare|chesterton')]

# Dictionary to organize groups
groups = {
    'Austen': austen_books,
    'Shakespeare': shakespeare_books,
    'Chesterton': chesterton_books,
    'Other': other_books
}

print("\nGroup sizes:")
for group_name, group_df in groups.items():
    print(f"{group_name}: {len(group_df)} books")

print()

# Preprocessing data
# Expand contractions
books_df['text'] = books_df['text'].apply(lambda x: [contractions.fix(word) for word in x.split()])
books_df['text'] = [' '.join(map(str, l)) for l in books_df['text']]

# Noise cleaning - lowercase, spacing, special characters
books_df['text'] = books_df['text'].str.lower()
books_df['text'] = books_df['text'].str.replace("-", " ")
books_df['text'] = books_df['text'].apply(lambda x: re.sub(r'[^\w\d\s\']+', '', x))

# Tokenize
books_df['tokenized_text'] = books_df['text'].apply(word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
books_df['tokenized_text'] = books_df['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])

# Create text without stopwords for vectorization
books_df['text_no_stopwords'] = [' '.join(map(str, l)) for l in books_df['tokenized_text']]

# At this point we are done with preprocessing
print(books_df[['filename', 'text_no_stopwords']].head(1))

print()

# Creating Bag of Words vectors
vectorizer_bow = CountVectorizer()
bag_of_words = vectorizer_bow.fit_transform(books_df['text_no_stopwords'])

print("Bag of Words matrix shape:", bag_of_words.shape)

# Creating TF-IDF vectors
tdidf_vectorizer = TfidfVectorizer(
    max_df=0.3,
    stop_words='english',
    lowercase=True,
    use_idf=True,
    norm=u'l2',
    smooth_idf=True
)
tfidf_matrix = tdidf_vectorizer.fit_transform(books_df['text_no_stopwords'])

print("TF-IDF matrix shape:", tfidf_matrix.shape)

print()



'''
Doing step 1: Finding the most similar document using cosine similarity
'''
tfidf_array = tfidf_matrix.toarray()
selected_doc_index = 14 # Picking first Shakespeare book
selected_doc = books_df.iloc[selected_doc_index]['filename']
print("Finding the Most Similar Document using Cosine Similarity")
print(f"Selected Document: {selected_doc}")

# Calculate cosine similarity between all documents
cosine_similarities = cosine_similarity(tfidf_array)

# Get similarity scores for the selected document
selected_row = cosine_similarities[selected_doc_index]

# Find the most similar document (excluding itself)
similar_indices = np.argsort(selected_row)[::-1]  # Sort in descending order
most_similar_index = similar_indices[1]  # Index 1 because index 0 is the document itself
most_similar_doc = books_df.iloc[most_similar_index]['filename']
similarity_score = selected_row[most_similar_index]

print(f"Most similar document: {most_similar_doc}")
print(f"Cosine similarity score: {similarity_score:.4f}")

# Determine if they're in the same group
selected_group = None
similar_group = None
for group_name, group_df in groups.items():
    if selected_doc in group_df['filename'].values:
        selected_group = group_name
    if most_similar_doc in group_df['filename'].values:
        similar_group = group_name

print(f"Selected document group: {selected_group}")
print(f"Most similar document group: {similar_group}")
print(f"Same group: {selected_group == similar_group}")

print()



'''
Doing Step 2: Summing the bag-of-words vectors for each group and find top 20 most frequent words
'''
# Convert bag of words to array
bow_array = bag_of_words.toarray()
bow_feature_names = vectorizer_bow.get_feature_names_out()

# For each group, sum the BoW vectors and find top 20 words
for group_name, group_df in groups.items():
    group_indices = group_df.index.tolist()
    group_bow_sum = bow_array[group_indices].sum(axis=0)
    
    # Get top 20 words
    top_20_indices = np.argsort(group_bow_sum)[::-1][:20]
    top_20_words = bow_feature_names[top_20_indices]
    top_20_counts = group_bow_sum[top_20_indices]
    
    print(f"\n{group_name}:")
    for i, (word, count) in enumerate(zip(top_20_words, top_20_counts), 1):
        print(f"  {i:2d}. {word:20s} {int(count):6d}")

print()



'''
Doing Step 3: Summing the TF-IDF vectors for each group and find top 20 most important words
'''
tfidf_array = tfidf_matrix.toarray() # Did this earlier but doing it again
tfidf_feature_names = tdidf_vectorizer.get_feature_names_out()

# For each group, sum the TF-IDF vectors and find top 20 words
for group_name, group_df in groups.items():
    group_indices = group_df.index.tolist()
    group_tfidf_sum = tfidf_array[group_indices].sum(axis=0)
    
    # Get top 20 words
    top_20_indices = np.argsort(group_tfidf_sum)[::-1][:20]
    top_20_words = tfidf_feature_names[top_20_indices]
    top_20_weights = group_tfidf_sum[top_20_indices]
    
    print(f"\n{group_name}:")
    for i, (word, weight) in enumerate(zip(top_20_words, top_20_weights), 1):
        print(f"  {i:2d}. {word:20s} {weight:8.4f}")


""" REFLECTION

The selected document (shakespeare-caesar.txt) is most similar to another 
document in the SAME group (shakespeare-hamlet.txt) with a cosine similarity of 0.3780.
This makes sense because:
- Both documents are written by Shakespeare, so they share similar vocabulary and writing style/patterns
- Shakespeare's writing is unique with words like (haue, thou, thy, vpon)



There is a lot of overlap in each group
- AUSTEN: "could", "would", "must", "said" are frequent

- SHAKESPEARE: "haue", "thou", "shall" are frequent

- CHESTERTON: "said", "man", "like" are frequent

- OTHER: "shall", "unto", "lord" are frequent. Less overlap than previous three
  because this is a diverse group with different styles



There are clear differences in most frequent words between these groups:
- AUSTEN: Verbs like ("could", "would", "must") appear more because of dialogue
- SHAKESPEARE: Words like ("thou", "thy", "thee") appear more because of old english and plays
- CHESTERTON: Verbs like ("said", "seemed") and descriptive words like ("man", "like") appear more because of fiction
- OTHER: Biblical language ("shall", "unto", "lord", "thy", "god") dominates because of the bible
These differences show two things, GENRE and TIME PERIOD because novels use different language than plays, and Biblical language is very different from fiction.



There are also clear differences between most important words between these groups:
- AUSTEN: Had more character names like (elinor, emma, etc) because of genre and theme of romantic relationships
- SHAKESPEARE: Had more words like (haue, vpon, selfe) and CHARACTER NAMES (brutus, macbeth, cassius).
  They are unique to Shakespeare's Old English
- CHESTERTON: Had more character names like (syme, turnbull, flambeau, macian) because of genre in fiction
- OTHER: Had mix of character names like (alice, buster, ahab) and words like (israel, weep, david).
  This shows us the diverse content of the group (Bible, Blake poems, classic novels)
  
  The difference is that TF-IDF looks for uniqueness rather than just frequency. 
  Character names are important to TF-IDF because they appear frequently in one group but rarely in others. 
  This makes them excellent for finding differences between authors
"""