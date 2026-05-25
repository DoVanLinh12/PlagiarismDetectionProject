from datetime import datetime, timezone
from io import BytesIO

from fastapi import APIRouter, File, Form, UploadFile

from app.core.database import get_conn, release_conn
from app.models.check import CheckReport, CheckResponse
from app.repositories import document_repo
from app.services import embedding, minhash, preprocessing
from app.services.checker import find_candidate

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"application/pdf"}
MINHASH_THRESHOLD = 0.05   # lọc thô: giữ tài liệu có Jaccard >= 5%


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
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file PDF")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="File rỗng")

    # 1. Tiền xử lý tài liệu đẩy lên
    full_text, sentences = preprocessing.extract_and_preprocess(BytesIO(file_bytes))

    if not sentences:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Không trích xuất được câu nào từ file")

    # 2. Tạo MinHash → lọc thô
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

    # 3. Embedding tài liệu đẩy lên
    sentence_texts = [s.sentence_text for s in sentences]
    embeddings = embedding.embed_sentences(sentence_texts)

    # 4. So khớp chi tiết với từng tài liệu tham chiếu
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