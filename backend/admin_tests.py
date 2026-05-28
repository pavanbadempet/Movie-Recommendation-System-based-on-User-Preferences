"""
Enterprise Testing Command Center API.

This router allows system administrators to programmatically trigger 
extreme testing suites (Fuzzing, Integration, Security) and retrieve 
structured JSON results of the physical test outputs.
"""

import os
import subprocess
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

router = APIRouter(prefix="/v1/admin/tests", tags=["Admin Diagnostics"])

class TestTriggerRequest(BaseModel):
    suite: str  # e.g., "security", "fuzzing", "integration", "all"

@router.post("/run")
async def run_testing_suite(request: TestTriggerRequest):
    """
    Triggers a PyTest suite programmatically and returns the stdout logs.
    In a full production environment, this would run asynchronously and 
    stream results via WebSockets.
    """
    suite_map = {
        "security": "backend/tests/test_security_api.py",
        "fuzzing": "backend/tests/test_models_fuzzing.py",
        "integration": "backend/tests/test_integration_pipeline.py",
        "math": "backend/tests/test_ensemble_math.py",
        "all": "backend/tests/"
    }
    
    target = suite_map.get(request.suite.lower())
    if not target:
        raise HTTPException(status_code=400, detail="Invalid test suite.")
        
    try:
        # We run pytest and capture the output to return to the frontend dashboard
        process = subprocess.run(
            ["python", "-m", "pytest", target, "-v", "--tb=short"], 
            capture_output=True, 
            text=True,
            timeout=120 # Cap execution at 2 mins for API requests
        )
        
        return {
            "suite": request.suite,
            "status": "success" if process.returncode == 0 else "failed",
            "exit_code": process.returncode,
            "logs": process.stdout + "\n" + process.stderr
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Test suite timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
