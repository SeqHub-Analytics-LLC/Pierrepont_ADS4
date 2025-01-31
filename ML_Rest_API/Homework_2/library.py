

"""### **Challenge 6: Library Book Management System (With Type Hinting and Type Checking)**

#### **Objective**:
Build a program for managing a library's collection of books, ensuring that all functions and classes use **type hints** effectively. Use **mypy** to perform type checking and ensure type safety.

#### **Problem Statement**:
Create a library management system with the following requirements:
1. Represent each book in the library as a dictionary with the following keys:
   - `"title"`: A string representing the book's title.
   - `"author"`: A string representing the book's author.
   - `"year"`: An integer representing the publication year.

2. Implement the following features:
   - **Add a Book**: A function to add a new book to the library.
   - **Find Books by Author**: A function to search for all books by a specific author.
   - **List All Books**: A function to display all books in the library.

3. Enforce the use of type hints in all functions and data structures:
   - Use **list of dictionaries** to store books.
   - Ensure proper type annotations for arguments and return values.


#### **Requirements**:
1. Use type hints in all functions, classes, and variables.
2. Validate the code with **mypy** to ensure there are no type errors.
3. Add basic input validation to ensure data types are correct.

---

#### **Example Input/Output**:

#### **Example 1**:
```python
library = []

# Add books
add_book(library, "To Kill a Mockingbird", "Harper Lee", 1960)
add_book(library, "1984", "George Orwell", 1949)
add_book(library, "Animal Farm", "George Orwell", 1945)

# List all books
list_books(library)
# Output:
# 1. "To Kill a Mockingbird" by Harper Lee (1960)
# 2. "1984" by George Orwell (1949)
# 3. "Animal Farm" by George Orwell (1945)

# Find books by George Orwell
books_by_orwell = find_books_by_author(library, "George Orwell")
print(books_by_orwell)
# Output:
# [{"title": "1984", "author": "George Orwell", "year": 1949},
#  {"title": "Animal Farm", "author": "George Orwell", "year": 1945}]
```

---

#### **Instructions for Students**:
1. Save the code to a Python file (e.g., `library.py`).
2. Run the program normally to ensure functionality.
3. Install **mypy** if not already installed:
   ```bash
   pip install mypy
   ```
4. Use mypy to check for type errors:
   ```bash
   mypy library.py
   ```
5. If any type errors are detected, fix them and re-run the type check.


#### **Edge Cases to Test**:
1. Adding a book with missing or incorrect data types (e.g., non-integer year).
2. Searching for an author who has no books in the library.
3. Listing books when the library is empty.

"""


#Your code here
from typing import List, Dict, TypedDict

class Book(TypedDict):
    """Represents a book with title, author, and year."""
    title: str
    author: str
    year: int

def add_book(library: List[Book], title: str, author: str, year: int) -> None:
    """Adds a new book to the library."""
    if not isinstance(title, str) or not isinstance(author, str) or not isinstance(year, int):
        raise ValueError("Invalid data types: Title and Author must be strings, Year must be an integer.")

    book: Book = {"title": title, "author": author, "year": year}
    library.append(book)

def find_books_by_author(library: List[Book], author: str) -> List[Book]:
    """Finds and returns all books by a given author."""
    return [book for book in library if book["author"].lower() == author.lower()]

def list_books(library: List[Book]) -> None:
    """Displays all books in the library."""
    if not library:
        print("The library is empty.")
        return
    
    for index, book in enumerate(library, start=1):
        print(f"{index}. \"{book['title']}\" by {book['author']} ({book['year']})")
        
# Example Usage
if __name__ == "__main__":
    library: List[Book] = []
    add_book(library, "To Kill a Mockingbird", "Harper Lee", 1960)
    add_book(library, "1984", "George Orwell", 1949)
    add_book(library, "Animal Farm", "George Orwell", 1945)

    print("\nAll Books:")
    list_books(library)

    print("\nBooks by George Orwell:")
    books_by_orwell = find_books_by_author(library, "George Orwell")
    print(books_by_orwell)