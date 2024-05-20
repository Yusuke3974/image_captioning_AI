from fastapi import FastAPI, File, UploadFile
from models.caption_model import CaptionModel
import shutil
import os

app = FastAPI()
model = CaptionModel()

@app.post("/generate-caption/")
async def generate_caption(file: UploadFile = File(...)):
    file_location = f"data/images/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    caption = model.generate_caption(file_location)
    os.remove(file_location)
    return {"caption": caption}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)