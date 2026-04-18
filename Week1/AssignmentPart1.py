'''
Part 1: Regular Expressions
Exercise 1
Provide three words or phrases which match the following regular expressions.  Use nltk.re_show() to prove that they match.

Exercise 2
Write regular expressions to match the following classes of strings:
A single determiner. Assume that a, an, and the are the only determiners. Note: determiners can appear at the beginning of a sentence.
An arithmetic expression using integers, addition, and multiplication, such as 2*35+800.
Include code to test your regular expressions.


Part 2: NLP Workflow
Selecting a reasonably sized corpora to pre-process and analyze using the techniques from lecture. The pre-processing includes cleaning, tokenizing, and normalization. The exploratory analysis includes basic information, description, and visualization.
'''
from nltk import re_show
from nltk.book import *
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import download
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import nltk
import os

print("\nPART 1: REGULAR EXPRESSIONS - EXERCISE 1")



# PART 1: REGULAR EXPRESSIONS - EXERCISE 1
def run_and_test_patterns(examples: list[str], pattern: str):
    for example in examples:
        print(f"    Example: '{example}' -> Verified with re_show: ", end=" ")
        re_show(pattern, example)

# Pattern 1: [a-zA-Z]+
# Example here strictly disregards any number as seen with re_show on MixExample123
print("1. Pattern: [a-zA-Z]+ -> This looks for strings that have one or more letters case-insensitive")
pattern1 = r'[a-zA-Z]+'
examples1 = ['Rudra', 'PATEL', 'MixExample123']
run_and_test_patterns(examples1, pattern1)

# Pattern 2: [A-Z][a-z]*
print("\n2. Pattern: [A-Z][a-z]* -> This looks for strings that start with 1 uppercase letter followed by zero or more lowercase letters")
pattern2 = r'[A-Z][a-z]*'
examples2 = ['Rudra', 'P', 'Patel']
run_and_test_patterns(examples2, pattern2)

# Pattern 3: b[aeiou]{,2}t
print("\n3. Pattern: b[aeiou]{,2}t -> This looks for 'b' followed by 0-2 vowels (a,e,i,o,u) and ending with the letter 't'")
pattern3 = r'b[aeiou]{,2}t'
examples3 = ['bt', 'boat', 'bet']
run_and_test_patterns(examples3, pattern3)

# Pattern 4: \d+(\.\d+)?
print("\n4. Pattern: \\d+(\\.\\d+)? -> This looks for one or more digits, then an optional decimal point and then more digits")
pattern4 = r'\d+(\.\d+)?'
examples4 = ['981', '3.14159', '0.231']
run_and_test_patterns(examples4, pattern4)

# Pattern 5: ([^aeiou][aeiou][^aeiou])*
print("\n5. Pattern: ([^aeiou][aeiou][^aeiou])* -> This looks for a string where first letter is consonant, then vowel, and then consonant repeated 0 or more times. So empty strings should work too")
pattern5 = r'([^aeiou][aeiou][^aeiou])*'
examples5 = ['fiz', 'buz', 'fizbuzfizbuzfizbuz']
run_and_test_patterns(examples5, pattern5)

# Pattern 6: \w+[^\w]\w+
print("\n6. Pattern: \\w+[^\\w]\\w+ -> This looks for a string with one or more alphanumeric/underscore, then non alphanumeric/underscore, then one or more alphanumeric/underscore")
pattern6 = r'\w+[^\w]\w+'
examples6 = ['f-strings', 'Rudra.123', 'fizz-buzz123']
run_and_test_patterns(examples6, pattern6)


print("\n\nPART 1: REGULAR EXPRESSIONS - EXERCISE 2")

# Pattern 1 for part 2 - tested with the same function from part 1
print("1. Single Determiner: Matches 'a', 'an', or 'the'")
pattern_single_determiner = r'\b(a|an|the|A|An|The)\b'
examples_single_determiner = ['An apple', 'On the table, there is a fork', 'A lion in the zoo with an elephant']
run_and_test_patterns(examples_single_determiner, pattern_single_determiner)

# Pattern 2 for part 2 - tested with the same function from part 1
print("\n2. An arithmetic expression using integers, addition, and multiplication, such as 2*35+800")
pattern_arithmetic = r'\d+([+*]\d+)+'
examples_arithmetic = ["2*35+800", "21076341*2+90", "1+2*3"]
run_and_test_patterns(examples_arithmetic, pattern_arithmetic)

print()



# PART 2: NLP WORKFLOW

# Selecting a reasonably sized corpora to pre-process and analyze using the 
# techniques from lecture. The pre-processing includes cleaning, tokenizing, 
# and normalization. The exploratory analysis includes basic information, 
# description, and visualization.


# # Download required NLTK data
# download("punkt_tab")
# nltk.download('gutenberg')
# nltk.download('genesis')
# nltk.download('inaugural')
# nltk.download('nps_chat')
# nltk.download('webtext')
# nltk.download('treebank')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Loading/Displaying basic information about my selection
print("\n--- Loading/Displaying basic information about my selection ---")

print(f"Corpus: {text7.name}")
print(f"Number of tokens: {len(text7)}")
print(f"Number of types: {len(set(text7))}")
print(f"Number of types (lowercased): {len(set(word.lower() for word in text7))}")
print(f"Number of types (no punctuation): {len(set(word.lower() for word in text7 if word.isalpha()))}")

# Looking at Frequency Distributions
print("\n--- Looking at Frequency Distributions ---")

# General frequency distribution including the punctuation
frequency_distribution = FreqDist(word.lower() for word in text7)
print("Top 50 most common words:")
print(frequency_distribution.most_common(50))

# Specifically looking at words
print(f"\nFrequency of the word 'the' is: {frequency_distribution['the']}")
print(f"Frequency of word 'company' is: {frequency_distribution['company']}")
print(f"Frequency of word 'stock' is: {frequency_distribution['stock']}")

# Frequency distribution of title case words (words starting with uppercase letter)
frequency_dist_titlecase = FreqDist(word for word in text7 if word.istitle())
print("\nTop 10 title case words:")
print(frequency_dist_titlecase.most_common(10))

# Frequency distribution for word lengths
frequency_dist_wordlen = FreqDist(len(word) for word in text7)
print(f"\nMost common word length is of length {frequency_dist_wordlen.max()}")
print("Top 5 most common word lengths:")
print(frequency_dist_wordlen.most_common(5))

# Long words
word_set = set(text7)
long_words = [word for word in word_set if len(word) > 15]
print(f"\nThere are {len(long_words)} words longer than 15 characters")

# Pre-processing, Cleanup, Tokenizing, and Normalization of Data
print("\n--- Pre-processing, Cleanup, Tokenizing, and Normalization of Data ---")

# Raw Text
raw_text = ' '.join(text7)

# Cleaning the data
# We are removing punctuation and making it lowercase 
cleaned_data = re.sub(r'[^\w\s]', '', raw_text.lower())

# Tokenizing data
tokens = nltk.word_tokenize(cleaned_data)

# Removing stopwords
stop_words = set(stopwords.words('english'))
data_no_stop = [word for word in tokens if word not in stop_words and word.isalpha()]

# Stemming Porter
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in data_no_stop]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in data_no_stop]

print(f"Original tokens: {len(tokens)}")
print(f"After stopword removal: {len(data_no_stop)}")
print(f"Sample stemmed: {stemmed[:10]}")
print(f"Sample lemmatized: {lemmatized[:10]}")

# Visualizing the data
print("\n--- Visualizing the data ---")

# Get the directory where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

common_words = frequency_distribution.most_common(20)
df_freq = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Word', data=df_freq, palette='viridis')
plt.title('Top 20 Most Frequent Words in WSJ Corpus')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.savefig(os.path.join(current_dir, 'top_20_words.png'))
plt.close()
print("Saved: top_20_words.png")

word_lengths = [len(word) for word in text7]
plt.figure(figsize=(8, 5))
sns.histplot(word_lengths, bins=20, kde=True, color='skyblue')
plt.title('Distribution of Word Lengths in WSJ Corpus')
plt.xlabel('Word Length')
plt.ylabel('Frequency')
plt.savefig(os.path.join(current_dir, 'word_length_distribution.png'))
plt.close()
print("Saved: word_length_distribution.png")

title_freq_filtered = FreqDist(word for word in text7 if word.istitle() and not word.startswith('*'))
title_common = title_freq_filtered.most_common(10)
df_title = pd.DataFrame(title_common, columns=['Word', 'Frequency'])
plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Word', data=df_title, palette='coolwarm')
plt.title('Top 10 Title Case Words in WSJ Corpus (Excluding Annotations)')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.savefig(os.path.join(current_dir, 'top_10_titlecase_words.png'))
plt.close()
print("Saved: top_10_titlecase_words.png")



""" REFLECTION
- Corpus: I chose Wall Street Journal (text7)
- Pre-processing: I cleaned, tokenized, and normalized (stemmed/lemmatized) the data with stopwords.
- Analysis: I noticed that frequent words are dominated by common terms like 'the'. Word lengths peak at 3-4 characters and title case includes proper nouns.
- Insights: The corpus reflects financial news with terms like 'company' and 'stock'. Visualizations show skewed distributions.
"""