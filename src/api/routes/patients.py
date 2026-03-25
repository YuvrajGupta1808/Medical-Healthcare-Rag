import json
import os
import logging
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

DB_FILE = "data/patients.json"

class Patient(BaseModel):
    id: str
    name: str
    age: int
    gender: str
    doc_ids: List[str] = []

def load_patients_db() -> List[dict]:
    try:
        if not os.path.exists(DB_FILE):
            return [
                {"id": "p1", "name": "John Doe", "age": 45, "gender": "Male", "doc_ids": []},
                {"id": "p2", "name": "Jane Smith", "age": 32, "gender": "Female", "doc_ids": []}
            ]
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load patient database: %s", e)
        return []

def save_patients_db(patients: List[dict]):
    try:
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
        # Use a temporary file and rename for atomicity if needed, 
        # but for simple rag we'll use direct write for now.
        with open(DB_FILE, "w") as f:
            json.dump(patients, f, indent=2)
    except Exception as e:
        logger.error("Failed to save patient database: %s", e)

def link_doc_to_patient(patient_id: str, doc_id: str):
    patients = load_patients_db()
    for p in patients:
        if p["id"] == patient_id:
            if doc_id not in p["doc_ids"]:
                p["doc_ids"].append(doc_id)
                save_patients_db(patients)
            return True
    return False

def unlink_doc_from_patient(doc_id: str):
    patients = load_patients_db()
    changed = False
    for p in patients:
        if doc_id in p["doc_ids"]:
            p["doc_ids"].remove(doc_id)
            changed = True
    if changed:
        save_patients_db(patients)
    return changed

@router.get("", response_model=List[Patient])
def get_patients():
    """List all registered patients from the persistent clinical record."""
    return load_patients_db()

@router.post("", response_model=Patient, status_code=201)
def create_patient(patient: Patient):
    """Register a new patient context for isolated vector analysis."""
    patients = load_patients_db()
    if any(p["id"] == patient.id for p in patients):
        raise HTTPException(status_code=400, detail="Patient with this ID already exists.")
    
    patients.append(patient.model_dump())
    save_patients_db(patients)
    logger.info("New patient registered: %s (ID: %s)", patient.name, patient.id)
    return patient

@router.delete("/{patient_id}")
def delete_patient(patient_id: str):
    """Remove a patient context and all associated metadata."""
    patients = load_patients_db()
    target = next((p for p in patients if p["id"] == patient_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Patient record not found.")
    
    filtered = [p for p in patients if p["id"] != patient_id]
    save_patients_db(filtered)
    logger.info("Patient record purged: %s", patient_id)
    return {"status": "success", "deleted_id": patient_id}
