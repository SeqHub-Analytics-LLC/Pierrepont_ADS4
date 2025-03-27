from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from typing import Annotated


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


players = {"Lebron James": 
            {"ppg": 27.0, "team" : "Lakers", "position": "SF"},
           "Kevin Durant": 
            {"ppg": 30.0, "team": "Nets", "position": "SF"}, 
           "Stephen Curry": 
            {"ppg": 30.1, "team": "Warriors", "position": "PG"}, 
            "Kawhi Leonard": 
                {"ppg": 27.1, "team": "Clippers", "position": "SF"},
            "James Harden": 
                {"ppg": 25.0, "team": "Nets", "position": "SG"}}
            

class Position(str, Enum):
    PG = "PG"
    SG = "SG"
    SF = "SF"
    PF = "PF"
    C = "C"
    
class Player (BaseModel):
    team: str
    position: Position
    #ppg is float gt than 0 
    ppg : float = Field(..., gt=0)


@app.get("/players/{name}")
def read_player(name: str):
    return players[name]



@app.put("/players/{name}")
def update_player(name: str, player: Player):
    players[name] = player
    return players[name]

@app.delete("/players/{name}")
def delete_player(name: str):
    players.pop(name)
    return {"message": "Player deleted!"}

@app.get("/players/")
def get_players(position: Position = None, team: str = None):
    filtered_players = players
    if position:
        filtered_players = {name: info for name, info in filtered_players.items() if info["position"] == position}
    if team:
        filtered_players = {name: info for name, info in filtered_players.items() if info["team"] == team}
    return filtered_players


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("path_parameters:app", host="localhost", port=8000, reload=True)