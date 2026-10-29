"""
backend/routers/evaluation_router.py
=======================================
Serves the most recent evaluation report (see evaluation/evaluate.py) for
dashboard display.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from backend.auth import get_current_user
from backend.schemas import EvaluationSummary
from common.config import PROJECT_ROOT

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/", response_model=EvaluationSummary)
def get_latest_evaluation(current_user: dict = Depends(get_current_user)) -> EvaluationSummary:
    """Return the most recently generated evaluation_report.json."""
    path = PROJECT_ROOT / "results" / "evaluation_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No evaluation report found. Run evaluation/evaluate.py first.")
    with open(path, "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    return EvaluationSummary(metrics=metrics)


@router.get("/datasets")
def get_dataset_comparison(current_user: dict = Depends(get_current_user)) -> dict:
    """Return the latest per-dataset inventory, ranking metrics, and efficiency report."""
    path = PROJECT_ROOT / "results" / "dataset_comparison.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No per-dataset comparison found. Run "
                "`python -m evaluation.evaluate_datasets` first."
            ),
        )
    with path.open("r", encoding="utf-8") as report_file:
        return json.load(report_file)
