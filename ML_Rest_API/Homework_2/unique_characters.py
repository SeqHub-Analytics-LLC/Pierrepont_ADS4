"""

### **Challenge 2: Unique Character Generator**

#### **Objective**:
Write a **generator function** that processes a string and yields **unique characters**, ignoring case. The program should:
1. Ignore repeated characters, even if they appear in different cases (e.g., `'P'` and `'p'` should be treated as the same).
2. Preserve the **order of first appearance** of unique characters.
3. Skip any spaces or non-alphabetic characters.

#### **Requirements**:
- Create a generator function called `unique_characters()`.
- Use the generator to process the string `"Python Programming"`.
- Print each unique character on a new line.
- Handle edge cases like empty strings, special characters, or numeric values.

#### **Example Input/Output**:

##### **Example 1**:
Input: `"Python Programming"`

Output:
```plaintext
P
y
t
h
o
n
r
g
a
m
i
```

##### **Example 2**:
Input: `"Hello, World!"`

Output:
```plaintext
H
e
l
o
W
r
d
```

##### **Example 3**:
Input: `"12345!!Python12345!!"`

Output:
```plaintext
P
y
t
h
o
n
```

#### **Implementation Details**:
1. **Case Insensitivity**:
   - Convert characters to lowercase or uppercase before checking for uniqueness.
2. **Skip Non-Alphabetic Characters**:
3. **Preserve Order**:
   - track already-yielded characters while maintaining the input string's order.
4. **Edge Cases**:
   - Handle empty strings (no output).
   - Handle strings with only special characters or digits (no output).


"""
def unique_characters(input_string):
    """Generator to yield unique alphabetic characters from a string, ignoring case."""
    seen = set()  
    
    for char in input_string:
        if char.isalpha():  #  only alphabetic characters
            lower_char = char.lower()
            if lower_char not in seen:
                seen.add(lower_char)
                yield char  # Yield original character (preserving input order)

test_string = "Python Programming"
print("Unique characters:")
for char in unique_characters(test_string):
    print(char)
