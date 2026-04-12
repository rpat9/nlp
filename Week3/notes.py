'''
This is just me messing around with code to get a better understanding
'''
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import genesis
from nltk import FreqDist

text = word_tokenize("And now for something completely different")
print(text) # ['And', 'now', 'for', 'something', 'completely', 'different']

print()

# A part-of-speech tagger will process a sequence of words and attaches a part of speech tag to each word
tags = pos_tag(text)
print(tags) # [('And', 'CC'), ('now', 'RB'), ('for', 'IN'), ('something', 'NN'), ('completely', 'RB'), ('different', 'JJ')]

print()

# Load the corpora, but for demonstration purposes I am choosing only 5K words
genesis_words = genesis.words()[:5000]
genesis_tagged = pos_tag(genesis_words)
genesis_tag_distribution = FreqDist(tag for (word, tag) in genesis_tagged)
print(genesis_tag_distribution.most_common(5))

'''
This is just me messing around with tagging words and learning some tags

Actual assignment is where I will do more with these tagged words and regular expressions
'''