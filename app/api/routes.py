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

router = APIRouter()

# Directories
UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "outputs"
ensure_dir(UPLOADS_DIR)
ensure_dir(OUTPUTS_DIR)

# Initialize services
processor = DocumentProcessor()
llm_extractor = LLMExtractor(model_name=os.getenv("OLLAMA_MODEL", "ministral-3:latest"))
search_engine = SearchEngine()
chatbot = ChatbotService(model_name=os.getenv("OLLAMA_MODEL", "ministral-3:latest"))

# In-memory storage for results and status (use DB for production)
processing_results = {}
processing_status = {}

@router.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    uploaded_files = []
    for file in files:
        file_path = os.path.join(UPLOADS_DIR, file.filename)
        content_bytes = await file.read()
        
        # Save file to disk
        with open(file_path, "wb") as f:
            f.write(content_bytes)
        
        # Extract text and vectorize immediately
        text = processor.extract_text(file_path)
        clean_text = processor.clean_text(text) if text else ""
        
        embedding = None
        if clean_text:
            embedding = search_engine.model.encode(clean_text).tolist()
        
        # Save to DB
        db_doc = Document(
            document_name=file.filename,
            content=clean_text,
            vector_embeddings=embedding
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        
        uploaded_files.append(DocumentResponse.model_validate(db_doc))
        print(f"DEBUG: Vectorized and stored document {file.filename} in DB with ID {db_doc.id}")
    
    return {"message": f"{len(uploaded_files)} files uploaded successfully", "files": uploaded_files}

@router.post("/documents/process/{doc_id}")
async def process_document(doc_id: str, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    processing_status[job_id] = {"status": "processing", "progress": 0, "current_file": doc_id}
    
    background_tasks.add_task(run_processing_job_v2, job_id, doc_id)
    
    return {"status": "processing", "job_id": job_id}

async def run_processing_job_v2(job_id: str, doc_id: str):
    db = next(get_db())
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            print(f"DEBUG ERROR: Document {doc_id} not found in DB")
            processing_status[job_id] = {"status": "error", "error": "Document not found"}
            return

        print(f"DEBUG: Starting processing for document {doc.document_name} ({doc_id})")
        
        if not doc.content:
            print(f"DEBUG ERROR: No content found for document {doc.document_name}")
            doc.processed_output = {"class": "Unclassifiable", "error": "No content"}
            db.commit()
            processing_status[job_id] = {"status": "complete"}
            return

        # Process with LLM
        extracted_data = llm_extractor.extract(doc.content, "Unknown")
        final_class = extracted_data.get("document_type", "Processed")
        
        result = {"class": final_class, **extracted_data}
        doc.processed_output = result
        db.commit()
        
        print(f"DEBUG: Successfully processed and stored result for {doc.document_name}")
        processing_status[job_id] = {"status": "complete", "progress": 100}
        
    except Exception as e:
        print(f"DEBUG ERROR: Error processing document {doc_id}: {e}")
        processing_status[job_id] = {"status": "error", "error": str(e)}
    finally:
        db.close()

async def run_processing_job(job_id: str):
    files = [f for f in os.listdir(UPLOADS_DIR) if os.path.isfile(os.path.join(UPLOADS_DIR, f))]
    total_files = len(files)
    
    results = {}
    docs_for_search = []
    
    print(f"DEBUG: Starting processing job {job_id} for {total_files} files")
    
    for i, filename in enumerate(files):
        print(f"DEBUG: Processing file {i+1}/{total_files}: {filename}")
        file_path = os.path.join(UPLOADS_DIR, filename)
        processing_status[job_id].update({"progress": int((i / total_files) * 100), "current_file": filename})
        
        try:
            text = processor.extract_text(file_path)
            if not text:
                results[filename] = {"class": "Unclassifiable"}
                continue
                
            clean_text = processor.clean_text(text)
            
            # Use LLM for both classification and extraction
            # Strategy: Extractor now handles classification internally or we just pass a generic class
            doc_class = "Unknown" # Or let LLM determine it
            extracted_data = llm_extractor.extract(clean_text, doc_class)
            
            # If the LLM returned a document class in its JSON, use it
            final_class = extracted_data.get("document_type", "Processed")
            print(f"DEBUG: Successfully extracted data for {filename}. Class: {final_class}")
            
            result = {"class": final_class, **extracted_data}
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

@router.get("/documents/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in processing_status:
        # Fallback to check if results exist if status is lost
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
