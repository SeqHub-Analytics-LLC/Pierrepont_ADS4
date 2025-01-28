from datetime import date
from enum import Enum
from typing import List
from pydantic import BaseModel, ValidationError

# Gender options using Enum
class Gender(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    NON_BINARY = "NON_BINARY"

# Updated model with more fields
class Person(BaseModel):
    first_name: str
    last_name: str
    gender: Gender
    birthdate: date
    interests: List[str]

# Testing the Person model

# Valid test cases
try:
    print("Creating a valid person...")
    valid_person_1 = Person(
        first_name="Alice",
        last_name="Smith",
        gender=Gender.FEMALE,
        birthdate=date(1990, 5, 20),
        interests=["reading", "traveling", "music"]
    )
    print("Valid person 1:", valid_person_1)

    valid_person_2 = Person(
        first_name="Bob",
        last_name="Brown",
        gender=Gender.MALE,
        birthdate=date(1985, 12, 15),
        interests=["sports", "cooking"]
    )
    print("Valid person 2:", valid_person_2)
except ValidationError as e:
    print("Unexpected error with valid test case:", e)

# Invalid test cases
try:
    print("\nTesting an invalid person: Missing interests...")
    invalid_person_1 = Person(
        first_name="Charlie",
        last_name="Davis",
        gender=Gender.NON_BINARY,
        birthdate=date(2000, 7, 10),
        interests=None  # Invalid: interests must be a list
    )
    print("Invalid person 1 created (should not happen):", invalid_person_1)
except ValidationError as e:
    print("Error with invalid person 1:", e)

try:
    print("\nTesting an invalid person: Invalid gender...")
    invalid_person_2 = Person(
        first_name="Dana",
        last_name="Evans",
        gender="UNKNOWN",  # Invalid: must be a valid Gender enum value
        birthdate=date(1995, 3, 25),
        interests=["hiking", "photography"]
    )
    print("Invalid person 2 created (should not happen):", invalid_person_2)
except ValidationError as e:
    print("Error with invalid person 2:", e)

try:
    print("\nTesting an invalid person: Incorrect birthdate format...")
    invalid_person_3 = Person(
        first_name="Eli",
        last_name="Foster",
        gender=Gender.MALE,
        birthdate="1992-06-15",  # Invalid: must be a `date` object, not a string
        interests=["gaming", "reading"]
    )
    print("Invalid person 3 created (should not happen):", invalid_person_3)
except ValidationError as e:
    print("Error with invalid person 3:", e)

try:
    print("\nTesting an invalid person: Empty first name...")
    invalid_person_4 = Person(
        first_name="",
        last_name="Green",
        gender=Gender.FEMALE,
        birthdate=date(1988, 4, 18),
        interests=["art"]
    )
    print("Invalid person 4 created (should not happen):", invalid_person_4)
except ValidationError as e:
    print("Error with invalid person 4:", e)
