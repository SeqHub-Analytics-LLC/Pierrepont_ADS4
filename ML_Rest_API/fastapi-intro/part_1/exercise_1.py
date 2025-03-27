from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Some HTML in here</title>
        </head>
        <body>
            <h1>Look! HTML!</h1>
        </body>
    </html>
    """

@app.get("/products/{product_name}")
def get_product(product_name: str):
    return {"product_id": product_name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("exercise_1:app", host="0.0.0.0", port=8000, reload=True)
