from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Dict, Any
import os
import uuid
import json
from app.services.document_processor import DocumentProcessor
from app.services.llm_extractor import LLMExtractor
from app.services.search_engine import SearchEngine
from app.services.chatbot_service import ChatbotService
from app.utils.helpers import ensure_dir
from app.database import get_db, Document
from app.models.schemas import ChatQuery, ChatResponse, DocumentResponse, ProcessingStatus
from sqlalchemy.orm import Session
from fastapi import Depends
import asyncio
from fastapi.concurrency import run_in_threadpool

router = APIRouter()

UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "outputs"
ensure_dir(UPLOADS_DIR)
ensure_dir(OUTPUTS_DIR)

processor = DocumentProcessor()
llm_extractor = LLMExtractor(model_name=os.getenv("OLLAMA_MODEL"))
search_engine = SearchEngine()
chatbot = ChatbotService(model_name=os.getenv("OLLAMA_MODEL"))

processing_results = {}
processing_status = {}

@router.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    uploaded_files = []
    processing_tasks = []
    
    for file in files:
        file_path = os.path.join(UPLOADS_DIR, file.filename)
        content_bytes = await file.read()

        with open(file_path, "wb") as f:
            f.write(content_bytes)

        text = processor.extract_text(file_path)
        clean_text = processor.clean_text(text) if text else ""
        
        embedding = None
        if clean_text:
            embedding = search_engine.model.encode(clean_text).tolist()

        db_doc = Document(
            document_name=file.filename,
            content=clean_text,
            vector_embeddings=embedding
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        uploaded_files.append(db_doc)

        job_id = str(uuid.uuid4())
        processing_tasks.append(run_in_threadpool(run_processing_job_v2, job_id, db_doc.id))

    if processing_tasks:
        await asyncio.gather(*processing_tasks)
    
    final_response_files = []
    for f in uploaded_files:
        db.refresh(f)
        final_response_files.append(DocumentResponse.model_validate(f))

    return {"message": f"{len(uploaded_files)} files uploaded and processed successfully", "files": final_response_files}

def run_processing_job_v2(job_id: str, doc_id: str):
    db = next(get_db())
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not doc.content:
            doc.processed_output = {"class": "Unclassifiable", "error": "No content"}
            db.commit()
            processing_status[job_id] = {"status": "complete"}
            raise HTTPException(status_code=400, detail="No content found for document")

        extracted_data = llm_extractor.extract(doc.content, "Unknown")
        final_class = extracted_data.get("document_type", "Processed")
        
        result = {"class": final_class, **extracted_data}
        doc.processed_output = result
        db.commit()

        processing_status[job_id] = {"status": "complete", "progress": 100}
        
    except Exception as e:
        processing_status[job_id] = {"status": "error", "error": str(e)}
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

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

            doc_class = "Unknown"
            extracted_data = llm_extractor.extract(clean_text, doc_class)
            
            final_class = extracted_data.get("document_type", "Processed")
            
            result = {"class": final_class, **extracted_data}
            results[filename] = result
            
            docs_for_search.append({
                "filename": filename,
                "text": clean_text,
                "metadata": result
            })
            
        except Exception as e:
            results[filename] = {"class": "Error", "error": str(e)}
            raise HTTPException(status_code=500, detail=str(e))

    search_engine.index_documents(docs_for_search)

    output_path = os.path.join(OUTPUTS_DIR, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    global processing_results
    processing_results = results
    
    processing_status[job_id] = {"status": "complete", "progress": 100, "current_file": ""}

@router.get("/documents/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in processing_status:
        return {"status": "unknown"}
    return processing_status[job_id]

@router.post("/chat", response_model=ChatResponse)
async def chat(query_data: ChatQuery, db: Session = Depends(get_db)):
    response = chatbot.chat(query_data.query, db)
    return ChatResponse(response=response)

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "search": search_engine.model is not None,
            "chatbot": chatbot.model_name is not None
        }
    }
