from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from app.api.routes import router as api_router
from app.database import init_db

# Initialize database
try:
    init_db()
    print("DEBUG: Database initialized successfully")
except Exception as e:
    print(f"DEBUG ERROR: Database initialization failed: {e}")

app = FastAPI(title="AI Document Processing System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "AI Document Processing API is running"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
