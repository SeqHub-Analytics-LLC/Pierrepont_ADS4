#  Sports Analytics with APIs 
#  1. Retrieve Player Profiles 
#  2. Add a new player 
#  3. Get details of a player from their jersey name. 
from fastapi import FastAPI, Path, Query, HTTPException
from enum import Enum
from pydantic import BaseModel, Field

app = FastAPI()

#player = {
 #   "LeBron James":{"jersey_number":23, "team":"Lakers", "position":"Forward", "ppg": 27.1},
  #  "Steph Curry": {"jersey_number":30, "team":"Warriors", "position":"Guard","ppg":29.3}
#}

players =[
    {"name": "LeBron James", "jersey_number":23, "team":"Lakers","position": "Forward","ppg":27.1,"age":40},
    {"name": "Stephen Curry", "jersey_number":30,"team":"Warriors","position":"Guard","ppg":29.3,"age":36}

]

class Position(str, Enum):
    forward = "Forward"
    guard = "Guard"
    center = "Center"
#Potentially add height and weight
#Add additional statistics 
#Add achievements

class Player(BaseModel):
    age: int = Field(..., ge=18, le=60, description="Age has to be greater than or equal to 18 but less than or equal to 60")
    name: str = Field(..., min_length=3, description="Length of name has to be longer than 3 characters")
    jersey_number: int = Field(..., ge=0, le=99, description="Number has to be greater than or equal to 0 and less than 100")
    team: str
    position: Position
    ppg: float = Field(..., ge=0, description="Points per game can't be negative")
#Task 1
@app.get("/get_players/")
def get_players(team: str = Query(None, description="Filter players by team")):
    filtered_players=[]
    if team:
        for p in players:
            if p["team"]==team:
                filtered_players.append(p)
            #if players[p]["team"]!=team:
             #   raise HTTPException(status_code=404, detail="No players found for the specified team.")
        return filtered_players
    return players
#Task 2
@app.post("/add_player/{name}")
def add_player(name: str, player: Player):
    for player in players:
        if player["name"] == name:
            raise HTTPException(status_code=400, detail="A player with this name already exists.")

    new_player = {
        "age": player.age,
        "name": name
        "jersey_number": player.jersey_number,
        "team": player.team,
        "position": player.position.value,
        "ppg": player.ppg
    }
    players.append(new_player)
    return {"message": "Player added successfully", "player": new_player}
'''
#Task 3
#@app.get("/players/{jersey_number}")
#def get_player(jersey_number: int = Path(..., description="Jersey number of the player")):
    #Used gen AI for this line of code
 #   player = next((p for p in players if p["jersey_number"] == jersey_number), None)
  #  if not player:
   #     raise HTTPException(status_code=404, detail="Player not found.")
    #return player
'''
@app.get("/filter_players/")
def filter_players(team: str = Query(None, description="Filter players by team"), position: Position = Query(None, description="Filter players by position")):
    filtered_players = {}
    for name, player in players.items():
        if team and position: 
            if player["team"] == team and player["position"] == position:
                filtered_players[name] = player
        elif team: 
            if player["team"] == team:
                filtered_players[name] = player
        elif position:
            if player["position"] == position:
                filtered_players[name] = player
        else:
            filtered_players[name] = player

    if not filtered_players:
        raise HTTPException(status_code=404, detail="No players found matching the criteria.")
    
    return filtered_players


@app.get("/filter_players/")
def filter_players(team: str = Query(None, description="Filter players by team"), position: Position = Query(None, description="Filter players by position")):
    filtered_players = []
    for player in players:
        if team and position: 
            if player["team"] == team and player["position"] == position:
                filtered_players.append(player)
        elif team: 
            if player["team"] == team:
                filtered_players.append(player)
        elif position:
            if player["position"] == position:
                filtered_players.append(player)
        else:
            filtered_players.append(player)

    if not filtered_players:
        raise HTTPException(status_code=404, detail="No players found matching the criteria.")
    
    return filtered_players
