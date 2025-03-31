from fastapi import FastAPI, Path, Query, HTTPException
from pydantic import BaseModel, Field, validator, field_serializer
from enum import Enum
from datetime import date

app = FastAPI()

# =========================
# Existing Player Endpoints
# =========================

class Position(str, Enum):
    guard = "Guard"
    forward = "Forward"
    center = "Center"

class Player(BaseModel):
    player_name: str
    jersey_num: int = Field(..., ge=0, lt=100, description="Jersey num is a positive 2-digit number")
    team: str
    position: Position
    ppg: float = Field(..., ge=0, description="Points per game cannot be negative")

# Prepopulate with five NBA players using the Player class.
players = [
    Player(player_name="LeBron James", jersey_num=23, team="Lakers", position=Position.forward, ppg=27.1),
    Player(player_name="Stephen Curry", jersey_num=30, team="Warriors", position=Position.guard, ppg=29.3),
    Player(player_name="Kevin Durant", jersey_num=7, team="Suns", position=Position.forward, ppg=26.9),
    Player(player_name="Giannis Antetokounmpo", jersey_num=34, team="Bucks", position=Position.forward, ppg=28.5),
    Player(player_name="Anthony Davis", jersey_num=77, team="Mavericks", position=Position.forward, ppg=28.0)
]

# ===========================================
# Response Model: Players
# ===========================================

class PlayersResponse(BaseModel):
    players: list[Player]

# ===========================================
# Response Model: Messages
# ===========================================

class MessageResponse(BaseModel):
    message: str

@app.get("/")
def home():
    return {"Message": "Welcome to Tee Tog's Analytics!"}

@app.get("/get_players")
def get_players(team: str = Query(None, description="Filter players by team")):
    if team:
        filtered = [p for p in players if p.team.lower() == team.lower()]
        response = PlayersResponse(players=filtered)
    else:
        response = PlayersResponse(players=players)
    return response.json()

@app.post("/add_players")
def add_player(player: Player):
    players.append(player)
    response = MessageResponse(message="player successfully added")
    return response.json()

@app.get("/get_players/{team}/{jersey_number}")
def get_player_by_number(
    team: str,
    jersey_number: int = Path(..., ge=0, description="Jersey number must be positive")
):
    matching_team = [p for p in players if p.team.lower() == team.lower()]
    if not matching_team:
        raise HTTPException(status_code=404, detail="Player not found")
    for p in matching_team:
        if p.jersey_num == jersey_number:
            response = MessageResponse(message=p.player_name)
            return response.json()
    raise HTTPException(status_code=404, detail="Player not found")

# ===========================================
# New Feature 1: Fan Registration with Regex & Custom Date Serialization
# ===========================================

class Fan(BaseModel):
    name: str = Field(..., min_length=1, description="Fan's name")
    email: str = Field(
        ...,
        pattern=r"^[\w\.-]+@[\w\.-]+\.\w{2,}$",
        description="Email must be a valid email address"
    )
    favorite_team: str = Field(..., description="Fan's favorite team")
    registration_date: date = Field(default_factory=date.today, description="Registration date (defaults to today's date)")
    
    @field_serializer("registration_date")
    def serialize_registration_date(self, value: date) -> str:
        return value.strftime("%d-%m-%Y")
    
# ===========================================
# Response Model: Fan
# ===========================================

class FansResponse(BaseModel):
    fans: list[Fan]

# Prepopulate with three fan entries.
fans = [
    Fan(name="Jane Doe", email="jane.doe@example.com", favorite_team="Lakers", registration_date=date(2025, 3, 24)),
    Fan(name="John Smith", email="john.smith@example.com", favorite_team="Warriors", registration_date=date(2025, 3, 25)),
    Fan(name="Alice Johnson", email="alice.johnson@example.com", favorite_team="Mavericks", registration_date=date(2025, 3, 26))
]

@app.post("/register_fan")
def register_fan(fan: Fan):
    fans.append(fan)
    response = MessageResponse(message="Fan registered successfully")
    return response.json()

@app.get("/fans_by_team")
def get_fans_by_team(team: str = Query(None, description="Filter fans by favorite team")):
    if team:
        filtered_fans = [fan for fan in fans if fan.favorite_team.lower() == team.lower()]
    else:
        filtered_fans = fans
    response = FansResponse(fans=filtered_fans)
    return response.json()

# ===========================================
# New Feature 2: Game Scheduling with Date Filtering & Custom Date Serialization
# ===========================================

class GameStatus(str, Enum):
    Scheduled = "Scheduled"
    Completed = "Completed"
    Postponed = "Postponed"

class Game(BaseModel):
    game_date: date = Field(..., description="Date of the game (YYYY-MM-DD)")
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    status: GameStatus = Field(default=GameStatus.Scheduled, description="Status of the game")
    
    @validator("game_date")
    def validate_game_date(cls, v):
        if v < date.today():
            raise ValueError("Game date must not be in the past.")
        return v
    
    @field_serializer("game_date")
    def serialize_game_date(self, value: date) -> str:
        return value.strftime("%d-%m-%Y")
    
# ===========================================
# Response Model: Game
# ===========================================

class GamesResponse(BaseModel):
    games: list[Game]


# Prepopulate with three game entries.
games = [
    Game(game_date=date(2025, 4, 1), home_team="Lakers", away_team="Warriors", status=GameStatus.Scheduled),
    Game(game_date=date(2025, 4, 5), home_team="Bulls", away_team="Celtics", status=GameStatus.Postponed),
    Game(game_date=date(2025, 4, 10), home_team="Warriors", away_team="Mavericks", status=GameStatus.Completed),
    Game(game_date=date(2025, 4, 3), home_team="Bucks", away_team="Suns", status=GameStatus.Completed)
]

@app.post("/register_game")
def register_game(game: Game):
    games.append(game)
    response = MessageResponse(message="Game scheduled successfully")
    return response.json()

@app.get("/games_by_date")
def get_games_by_date(
    start_date: date = Query(None, description="Filter games starting from this date (YYYY-MM-DD)"),
    end_date: date = Query(None, description="Filter games until this date (YYYY-MM-DD)")
):
    filtered_games = games
    if start_date:
        filtered_games = [g for g in filtered_games if g.game_date >= start_date]
    if end_date:
        filtered_games = [g for g in filtered_games if g.game_date <= end_date]
    response = GamesResponse(games=filtered_games)
    return response.json()

# This endpoint uses the team name to verify that a player exists in that team.
# Then it fetches all upcoming games for that team and returns the next scheduled game.

@app.get("/team/{team_name}/next_game")
def get_team_next_game(team_name: str = Path(..., description="Name of the team")):
    
    # Get all upcoming games where the team is either home or away.
    relevant_games = [g for g in games if g.home_team.lower() == team_name.lower() or g.away_team.lower() == team_name.lower()]
    upcoming_games = [g for g in relevant_games if g.game_date >= date.today()]
    
    if not upcoming_games:
        raise HTTPException(status_code=404, detail="No upcoming games for the team")
    
    # Find the next game (earliest date).
    next_game = min(upcoming_games, key=lambda g: g.game_date)
    return next_game.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sports2:app", host="0.0.0.0", port=8000, reload=True)