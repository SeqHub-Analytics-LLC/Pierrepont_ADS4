from pydantic import BaseModel, field_serializer
from datetime import datetime

class ReportMetadata(BaseModel):
    timestamp: datetime
    source: str

class WeatherReport(BaseModel):
    city: str
    temperature: float
    humidity: int
    condition: str
    metadata: ReportMetadata

    @field_serializer("temperature")
    def convert_to_fahrenheit(cls, value):
        return (value * 9/5) + 32
    
weather_data = {
    "city": "Westport",
    "temperature": 20.0,
    "humidity": 60,
    "condition": "Sunny",
    "metadata": {
        "timestamp": "2025-01-22T15:30:00",
        "source": "NOAA"
    }
}

report = WeatherReport(**weather_data)
print(report.json())