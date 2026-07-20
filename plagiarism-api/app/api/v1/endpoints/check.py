from datetime import datetime, timezone
from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.database import get_conn, release_conn
from app.models.check import (
    CheckReport,
    CheckResponse,
)
from app.repositories import document_repo
from app.services import embedding, minhash, preprocessing
from app.services.checker import  find_candidate

router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
ALLOWED_FILE_EXTENSIONS = {".pdf", ".docx"}
MINHASH_THRESHOLD = 0


def _get_file_extension(filename: str | None) -> str:
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


@router.post(
    "/check",
    response_model=CheckReport,
    summary="Kiểm tra đạo văn của tài liệu đẩy lên",
)
async def check_plagiarism(
    subject_id: str = Form(..., description="ID môn học"),
    file: UploadFile = File(..., description="File PDF cần kiểm tra"),
    submission_id: int | None = Form(None, description="ID bài nộp (tuỳ chọn)"),
    topic_id: int | None = Form(None, description="ID đề tài (tuỳ chọn)"),
):
    file_extension = _get_file_extension(file.filename)
    if (
        file.content_type not in ALLOWED_CONTENT_TYPES
        and file_extension not in ALLOWED_FILE_EXTENSIONS
    ):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file PDF hoặc DOCX")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="File rỗng")

    try:
        full_text, sentences = preprocessing.extract_and_preprocess(BytesIO(file_bytes))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not sentences:
        raise HTTPException(status_code=400, detail="Không trích xuất được câu nào từ file")

    minhash_values = minhash.compute_minhash(full_text)

    conn = await get_conn()
    try:
        candidates = await minhash.find_candidates_by_minhash(
            conn=conn,
            minhash=minhash_values,
            subject_id=subject_id,
            threshold=MINHASH_THRESHOLD,
        )
    finally:
        await release_conn(conn)

    sentence_texts = [s.sentence_text for s in sentences]
    embeddings = embedding.embed_sentences(sentence_texts)

    check_result: CheckResponse = find_candidate(
        query_sentences=sentences,
        query_embeddings=embeddings,
        candidates=candidates,
    )

    return CheckReport(
        submission_id=submission_id,
        topic_id=topic_id,
        file_name=file.filename,
        plagiarism_score=round(check_result.plagiarism_ratio * 100, 4),
        status="checked",
        checked_at=datetime.now(timezone.utc),
        report=check_result,
    )
