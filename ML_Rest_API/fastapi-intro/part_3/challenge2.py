from pydantic import BaseModel, field_serializer
from datetime import date

class Movie(BaseModel):
    title: str
    genre: list[str]
    rating: float
    release_date: date

    @field_serializer("release_date")
    def format_date(cls, value):
        return value.strftime("%d-%m-%Y")

def process_movie_data(json_input: str):
    movie = Movie.parse_raw(json_input)
    return movie.dict()

input_json = """
{
    "title": "Inception",
    "genre": ["Sci-Fi", "Thriller"],
    "rating": 9.0,
    "release_date": "2010-07-16"
}
"""

result = process_movie_data(input_json)
print(result)