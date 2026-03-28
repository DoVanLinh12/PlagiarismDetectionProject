from fastapi import APIRouter
from app.api.v1.endpoints import document

router = APIRouter(prefix="/api/v1")

router.include_router(
    document.router,
    prefix="/documents",
    tags=["Documents"],
)
