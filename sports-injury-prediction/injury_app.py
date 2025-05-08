from fastapi import FastAPI, Path, Query
import pickle
from sklearn.ensemble import RandomForestClassifier
from enum import Enum
from typing import List
from pydantic import BaseModel, Field, ValidationError

# Gender options using Enum


app = FastAPI()


#player weight float between 0 and 150 kg    
#player height float between 0 and 250 cm
#player age int between 0 and 100
#previous injury : true or false 
#training intesntiy : float between 0 and 1 

# Position['Midfielder' 'Forward' 'Goalkeeper' 'Defender']
# Recovery time  Int between 1-6
# Training surface ['Artificial Turf' 'Hard Court' 'Grass']


class Position(str, Enum):
    MIDFIELDER = "Midfielder"
    FORWARD = "Forward"
    GOALKEEPER = "Goalkeeper"
    DEFENDER = "Defender"
class TrainingSurface(str, Enum):
    ARTIFICIAL_TURF = "Artificial Turf"
    HARD_COURT = "Hard Court"
    GRASS = "Grass"
class InjuryPredictionInput(BaseModel):
    weight: float = Field(..., ge=0, le=150)
    height: float = Field(..., ge=0, le=250)
    age: int = Field(..., ge=0, le=100)
    previous_injury: bool
    training_intensity: float = Field(..., ge=0, le=1)
    position: Position
    recovery_time: int = Field(..., ge=1, le=6)
    training_surface: TrainingSurface

def load_model(mdl_path = "pickels/random_forest_model.pkl"): 
    with open(mdl_path, 'rb') as f:
        model = pickle.load(f)
    return model


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict")
def predict(Input: InjuryPredictionInput = Query(None, ) ):
    return {"message": "This is a prediction endpoint."}
