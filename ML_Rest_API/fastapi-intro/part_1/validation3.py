from fastapi import FastAPI, Path

app = FastAPI()

## Always make a habit of setting the home route.
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/license-plates/{license}")
def get_license_plate(license: str = Path(..., min_length=9, max_length=9)):
    return {"license": license}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("validation3:app", host="0.0.0.0", port=8000, reload=True)


