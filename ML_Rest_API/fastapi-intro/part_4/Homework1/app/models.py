from pydantic import BaseModel, field_validator, root_validator
from enum import Enum

# Enum for Training Surface
class TrainingSurface(str, Enum):
    AT = "Artificial Turf"
    HC = "Hard Court"
    G = "Grass"
    
# Enum for Position
class Position(str, Enum):
    F = "Forward"
    M = "Midfielder"
    D = "Defender"
    G = "Goalkeeper"

# Enum for Previous Injuries  (0 or 1)
class PreviousInjuriesEnum(int, Enum):
    Yes = "Yes"
    No = "No"

# Enum for Model Types
class ModelType(str, Enum): 
    rf = "Random Forests"

# Input Data model
class InputFeatures(BaseModel):
    PlayerWeight: float
    PlayerHeight: float
    Previous_Injuries: PreviousInjuriesEnum
    Position: Position
    TrainingSurface: TrainingSurface
    PlayerAge: int
    TrainingIntensity: float
    RecoveryTime: float

    # Age validator to ensure it is between 18 and 100
    @field_validator('Player_Age')
    def validate_age(cls, v):
        if not (18 <= v <= 100):
            raise ValueError('Age must be between 18 and 100')
        return v

    # Field validators for player height
    @field_validator('Player_Height')
    def validate_height(cls, v):
        if v <= 0:
            raise ValueError(f'{v} must be greater than 0')
        return v

    # validator for player weight
    @field_validator('Player_Weight')
    def validate_weight(cls, v):
        if v <= 0:
            raise ValueError('Weight must be greater than 0')
        return v

    # Training Intensity validator to ensure it is between 0 and 1
    @field_validator('Training_Intensity')
    def validate_training_intensity(cls, v):
        if not (0 <= v <= 1):
            raise ValueError('Training Intensity must be between 0 and 1')
        return v
    

# Prediction Request Data model
class PredictionRequest(BaseModel):
    Features: InputFeatures
    model: ModelType