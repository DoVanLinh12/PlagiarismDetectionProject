from fastapi import APIRouter
from app.api.v1.endpoints import document, check

router = APIRouter(prefix="/api/v1")

router.include_router(
    document.router,
    prefix="/documents",
    tags=["Documents"],
)

router.include_router(
    check.router,
    prefix="/plagiarism",
    tags=["Plagiarism Check"],
)
