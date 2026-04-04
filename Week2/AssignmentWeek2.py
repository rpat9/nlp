"""
Write a program to compute unsmoothed unigrams and bigrams and sort the n-grams by their probabilities. Run your n-gram program on two different small corpora of your choice. Reminder: a corpora is a collection of written material. It is more than a single sentence or article.

Compare the statistics of the two corpora. Write a short summary of your results, including discussion of the following: 
What are the differences in the most common unigrams between the two? 

Discuss any interesting differences in bigrams.
"""
from nltk.corpus import webtext, inaugural
from nltk import bigrams
from collections import defaultdict

# For this assignment what I will do is compare webtext to inaugural. One of these is very formal and other one is informal so it will be interesting to see what happens with our n-gram results


# I am loading the words first
inaugural_words = inaugural.words()

# Count the total number of words in inaugural to calculate probabilities later
total_inaugural_words = len(inaugural_words)

# Creating a dictionary and storing frequency of every unigram (word)
inaugural_counts = {}
for word in inaugural_words:
    inaugural_counts[word] = inaugural_counts.get(word, 0) + 1

# Now I am creating a new dictionary to store probabilities of each unigram
inaugural_probabilities = {}
for word, count in inaugural_counts.items():
    inaugural_probabilities[word] = count / total_inaugural_words

# Computing and printing the top 20 unigrams by their probabilities
sorted_unigrams_inaugural = sorted(inaugural_probabilities.items(), key=lambda item: item[1], reverse=True) # Using lambda to sort by item[1] which is our probabilities
print("\nTop 20 unigrams in inaugural:")
for word, prob in sorted_unigrams_inaugural[:20]:
    print(f"WORD: '{word}'  PROB: {prob:.2f}")

# Now let's compute bigram counts with a nested dictionary using defaultdict
inaugural_bigram_counts = defaultdict(lambda: defaultdict(int))
for sentence in inaugural.sents():
    for w1, w2 in bigrams(sentence, pad_left=True, pad_right=True):
        inaugural_bigram_counts[w1][w2] += 1

# Calculating bigram probabilities
inaugural_bigram_probs = defaultdict(lambda: defaultdict(int))
flat_bigram_probs = {}

for w1 in inaugural_bigram_counts:
    # This is the total occurrences of the first word of bigram
    total_w1_count = float(sum(inaugural_bigram_counts[w1].values()))
    
    for w2 in inaugural_bigram_counts[w1]:
        # Probability = count(w1, w2) / count(w1)
        prob = inaugural_bigram_counts[w1][w2] / total_w1_count
        inaugural_bigram_probs[w1][w2] = prob
        
        # When I first ran this I had bunch of words that only showed up once
        # This means that probability for top 20 was always 1.0
        # It won't be as interesting when we compare it later
        # To make a good comparison I am only adding bigrams to sort if it appears at least 10 times
        if inaugural_bigram_counts[w1][w2] >= 10:
            flat_bigram_probs[(w1, w2)] = prob

# Computing and printing the top 20 bigrams by their probabilities
sorted_bigrams_inaugural = sorted(flat_bigram_probs.items(), key=lambda item: item[1], reverse=True)
print("\nTop 20 bigrams in inaugural:")
for bigram, prob in sorted_bigrams_inaugural[:20]:
    print(f"BIGRAM: '{bigram[0]} {bigram[1]}'  PROB: {prob:.2f}")



# Doing the same steps for second corpus: WEBTEXT. This one is informal compared to inaugural

# Load the words
webtext_words = webtext.words()
total_webtext_words = len(webtext_words)

# Unigram counts and probabilities
webtext_counts = {}
for word in webtext_words:
    webtext_counts[word] = webtext_counts.get(word, 0) + 1

webtext_probabilities = {}
for word, count in webtext_counts.items():
    webtext_probabilities[word] = count / total_webtext_words

# Sort and print top 20 unigrams
sorted_unigrams_webtext = sorted(webtext_probabilities.items(), key=lambda item: item[1], reverse=True)
print("\nTop 20 unigrams in webtext:")
for word, prob in sorted_unigrams_webtext[:20]:
    print(f"WORD: '{word}'  PROB: {prob:.2f}")

# Bigram counts
webtext_bigram_counts = defaultdict(lambda: defaultdict(int))
for sentence in webtext.sents():
    for w1, w2 in bigrams(sentence, pad_left=True, pad_right=True):
        webtext_bigram_counts[w1][w2] += 1

# Bigram probabilities
webtext_bigram_probs = defaultdict(lambda: defaultdict(float))
flat_bigram_probs_webtext = {}

for w1 in webtext_bigram_counts:
    total_w1_count = float(sum(webtext_bigram_counts[w1].values()))
    for w2 in webtext_bigram_counts[w1]:
        prob = webtext_bigram_counts[w1][w2] / total_w1_count
        webtext_bigram_probs[w1][w2] = prob
        
        # Using the same threshold of 10 for a fair comparison
        if webtext_bigram_counts[w1][w2] >= 10:
            flat_bigram_probs_webtext[(w1, w2)] = prob

# Sort and print top 20 bigrams
sorted_bigrams_webtext = sorted(flat_bigram_probs_webtext.items(), key=lambda item: item[1], reverse=True)
print("\nTop 20 bigrams in webtext:")
for bigram, prob in sorted_bigrams_webtext[:20]:
    print(f"BIGRAM: '{bigram[0]} {bigram[1]}'  PROB: {prob:.2f}")
    


# My thoughts on results
print(
"""

COMPARISON: INAUGURAL CORPUS VS. WEBTEXT CORPUS

UNIGRAM DIFFERENCES:

- INAUGURAL: As expected for political speeches, the most common words 
  are formal structuring words ("the", "of", "and", "to") and 
  pronouns ("our", "we"). It makes sense because it is a political 
  address in front of the nation speaking to the public.
- WEBTEXT: This corpus is more about conversational and informal 
  punctuation with stuff like (":", "'", "?", "#", "!"). It has many
  personal pronouns like "I" and "you", showing direct, one-on-one 
  communication. Something that I found interesting is the isolated 
  fragments like "t" and "s" which came from "don't", "it's", etc.

BIGRAM DIFFERENCES:

- INAUGURAL: The bigrams for this is also very formal with noun 
  phrases and language like "General Government", "Vice President", 
  "United States", "obedience to", and "Old World". These phrases 
  represent a good speech given by a president.
- WEBTEXT: The bigrams here are also very conversational like it was
  for the unigrams. We see fragments like ("Don '", "Doesn '", "isn '", 
  "shouldn '") and internet words like ("cust ]", "Plug -"). 

""")