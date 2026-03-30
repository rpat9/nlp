'''
Part 1: Regular Expressions

Exercise 1
Provide three words or phrases which match the following regular expressions.  Use nltk.re_show() to prove that they match.

Exercise 2
Write regular expressions to match the following classes of strings:

A single determiner. Assume that a, an, and the are the only determiners. Note: determiners can appear at the beginning of a sentence.
An arithmetic expression using integers, addition, and multiplication, such as 2*35+800.

Include code to test your regular expressions.
'''
from nltk import re_show

print("\nPART 1: REGULAR EXPRESSIONS - EXERCISE 1")

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