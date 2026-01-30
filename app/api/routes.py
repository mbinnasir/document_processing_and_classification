from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Dict, Any
import os
import uuid
import json
from app.services.document_processor import DocumentProcessor
from app.services.classifier import DocumentClassifier
from app.services.extractor import DataExtractor
from app.services.search_engine import SearchEngine
from app.utils.helpers import ensure_dir

router = APIRouter()

# Directories
UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "outputs"
ensure_dir(UPLOADS_DIR)
ensure_dir(OUTPUTS_DIR)

# Initialize services (in a real app, use dependency injection)
processor = DocumentProcessor()
classifier = DocumentClassifier()
extractor = DataExtractor()
search_engine = SearchEngine()

# In-memory storage for results and status (use DB for production)
processing_results = {}
processing_status = {}

@router.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    uploaded_files = []
    for file in files:
        file_path = os.path.join(UPLOADS_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        uploaded_files.append(file.filename)
    
    return {"message": f"{len(uploaded_files)} files uploaded successfully", "files": uploaded_files}

@router.post("/documents/process")
async def process_documents(background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    processing_status[job_id] = {"status": "processing", "progress": 0, "current_file": ""}
    
    background_tasks.add_task(run_processing_job, job_id)
    
    return {"status": "processing", "job_id": job_id}

async def run_processing_job(job_id: str):
    files = [f for f in os.listdir(UPLOADS_DIR) if os.path.isfile(os.path.join(UPLOADS_DIR, f))]
    total_files = len(files)
    
    results = {}
    docs_for_search = []
    
    for i, filename in enumerate(files):
        file_path = os.path.join(UPLOADS_DIR, filename)
        processing_status[job_id].update({"progress": int((i / total_files) * 100), "current_file": filename})
        
        try:
            text = processor.extract_text(file_path)
            if not text:
                results[filename] = {"class": "Unclassifiable"}
                continue
                
            clean_text = processor.clean_text(text)
            doc_class = classifier.classify(clean_text)
            extracted_data = extractor.extract(clean_text, doc_class)
            
            result = {"class": doc_class, **extracted_data}
            results[filename] = result
            
            docs_for_search.append({
                "filename": filename,
                "text": clean_text,
                "metadata": result
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results[filename] = {"class": "Error", "error": str(e)}

    # Index for search
    search_engine.index_documents(docs_for_search)
    
    # Save to outputs
    output_path = os.path.join(OUTPUTS_DIR, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Update global results
    global processing_results
    processing_results = results
    
    processing_status[job_id] = {"status": "complete", "progress": 100, "current_file": ""}

@router.get("/documents/results")
async def get_results():
    if not processing_results:
        output_path = os.path.join(OUTPUTS_DIR, "results.json")
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                return json.load(f)
    return processing_results

@router.get("/documents/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in processing_status:
        # Fallback to check if results exist if status is lost
        return {"status": "unknown"}
    return processing_status[job_id]

@router.post("/search")
async def search(query_data: Dict[str, str]):
    query = query_data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    results = search_engine.search(query)
    return {"results": results, "count": len(results)}

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "classifier": classifier.classifier is not None,
            "search": search_engine.model is not None
        }
    }
