'''
Part of Speech Tagging - Programming Assignment

Exercise 1 - Using a Tagger
Many words, like ski and race, can be used as nouns or verbs with no difference in pronunciation. Can you think of others? Hint: think of a commonplace object and try to put the word to before it to see if it can also be a verb or think of an action and try to put the before it to see if it can also be a noun. Now make up a sentence with both uses of this word and run a part-of-speech tagger on this sentence.

Exercise 2 - Regular Expression Tagging
Come up with at least two patterns to improve the performance of the regular expression tagger presented in chapter 5 of the NLTK book (and duplicated below).
    patterns = [
        (r'.*ing$', 'VBG'),               # gerunds
        (r'.*ed$', 'VBD'),                # simple past
        (r'.*es$', 'VBZ'),                # 3rd singular present
        (r'.*ould$', 'MD'),               # modals
        (r'.*\'s$', 'NN$'),               # possessive nouns
        (r'.*s$', 'NNS'),                 # plural nouns
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
        (r'.*', 'NN')                     # nouns (default)
    ]
regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.accuracy(brown_tagged_sents)

Exercise 3 - Part of Speech Tagging
Use a part-of-speech tagger to tag the two corpora that you used in last week’s assignment or two other substantial corpora to Investigate the tags using counts and visualization. What, if anything, did you learn about your corpora by applying part-of-speech tags?
'''
from nltk import word_tokenize
from nltk import pos_tag
from nltk import RegexpTagger
from nltk import FreqDist
import matplotlib.pyplot as plt
try:
    from nltk.corpus import gutenberg, treebank
    print("All imports successful")
except:
    import nltk
    nltk.download('gutenberg')
    nltk.download('treebank')
    print("Corpora downloaded. All imports successful")



# Exercise 1: Using a Tagger on my Sentence
my_example = "I need to book a flight after I finish reading this NLTK book"
tokens = word_tokenize(my_example)
tagged_example = pos_tag(tokens)

print("\nExercise 1 - Tagged Sentence")
print(f"SENTENCE: {my_example}")
print(f"TAGGED: {tagged_example}\n") # Output here classifies 'book' as noun both times showing why NLP is hard



# Exercise 2: Adding to Regular Expression Tagging
patterns = [
    (r'.*ing$', 'VBG'),               # gerunds
    (r'.*ed$', 'VBD'),                # simple past
    (r'.*es$', 'VBZ'),                # 3rd singular present
    (r'.*ould$', 'MD'),               # modals
    (r'.*\'s$', 'NN$'),               # possessive nouns
    (r'.*s$', 'NNS'),                 # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
]
patterns.append((r'.*ly$', 'RB'))                  # adverbs
patterns.append((r'.*able$', 'JJ'))                # adjectives (-able)
patterns.append((r'.*ible$', 'JJ'))                # adjectives (-ible)
patterns.append((r'.*ful$', 'JJ'))                 # adjectives (-ful)
patterns.append((r'.*less$', 'JJ'))                # adjectives (-less)
patterns.append((r'.*ous$', 'JJ'))                 # adjectives (-ous)
patterns.append((r'.*self$', 'PRP'))               # singular pronouns (myself)
patterns.append((r'.*selves$', 'PRP'))             # plural pronouns (themselves)
patterns.append((r'.*', 'NN'))                     # nouns (default)

reg_exp_tagger = RegexpTagger(patterns)
emma_sents = gutenberg.sents('austen-emma.txt')
emma_tagged_sents = [pos_tag(sent) for sent in emma_sents]
acc = reg_exp_tagger.accuracy(emma_tagged_sents) 

print("\nExercise 2 - Adding to Regular Expression Tagging")
# 0.1724455019637996 - Makes sense because we didn't even include punctuation, determiners, conjunctions, etc
print(f"Regular Exp Accuracy on 'austen-emma.txt' from gutenberg: {acc}\n")



# Exercise 3: Tagging and Analyzing Corpora
print("\nExercise 3 - Tagging and Analyzing Corpora")

emma_words = gutenberg.words('austen-emma.txt')
wsj_words = treebank.words()

emma_tagged_words = pos_tag(emma_words)
wsj_tagged_words = pos_tag(wsj_words)

emma_tags_distribution = FreqDist(tag for (word, tag) in emma_tagged_words)
wsj_tags_distribution = FreqDist(tag for (word, tag) in wsj_tagged_words)

print("\nTop 10 Tags in Austen's Emma:")
print(emma_tags_distribution.most_common(10))

print("\nTop 10 Tags in Wall Street Journal (Treebank):")
print(wsj_tags_distribution.most_common(10))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
emma_tags_distribution.plot(15, title="Top 15 POS Tags: Austen's Emma", show=False)

plt.subplot(1, 2, 2)
wsj_tags_distribution.plot(15, title="Top 15 POS Tags: Wall Street Journal", show=False)

plt.tight_layout()
plt.show()

'''
What did I learn?

- Austen's Emma (fiction) likely contains a higher frequency of personal pronouns (PRP) and verbs due to narrative and dialogue.

- The Wall Street Journal (news/financial) typically contains more proper nouns (NNP) for companies/people and cardinal numbers (CD) for financial data.
'''