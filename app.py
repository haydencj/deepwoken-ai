from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from query_generate import query_and_generate_answer

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    # Render the index.html template.
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/stream")
async def stream(query: str = Query(..., description="User query to process")):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")

    def event_generator():
        # Stream tokens from the chatbot function.
        for token in query_and_generate_answer(query):
            yield f"data: {token}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
