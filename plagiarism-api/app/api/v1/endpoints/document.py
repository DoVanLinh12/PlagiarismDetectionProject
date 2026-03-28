import uuid
from io import BytesIO

from fastapi import APIRouter, File, Form, UploadFile

from app.core.database import get_conn, release_conn
from app.models.document import DocumentUploadResponse, UploadBatchResponse
from app.repositories import document_repo, milvus_repo
from app.services import embedding, minhash, preprocessing, storage

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"application/pdf"}


@router.post(
    "/upload",
    response_model=UploadBatchResponse,
    summary="Upload danh sách PDF tài liệu tham chiếu",
)
async def upload_documents(
    subject_id: str = Form(..., description="ID môn học"),
    files: list[UploadFile] = File(..., description="Danh sách file PDF"),
):
    results: list[DocumentUploadResponse] = []
    errors: list[dict] = []

    for file in files:
        try:
            print(f"\n[UPLOAD] Bắt đầu xử lý: {file.filename}")

            if file.content_type not in ALLOWED_CONTENT_TYPES:
                errors.append({"file_name": file.filename, "error": f"Chỉ chấp nhận PDF, nhận: {file.content_type}"})
                continue

            file_bytes = await file.read()
            if len(file_bytes) == 0:
                errors.append({"file_name": file.filename, "error": "File rỗng"})
                continue

            print(f"[UPLOAD] File size: {len(file_bytes)} bytes")

            conn = await get_conn()
            try:
                exists = await document_repo.document_exists(conn, file.filename, subject_id)
                if exists:
                    errors.append({"file_name": file.filename, "error": "File đã tồn tại trong môn học này"})
                    continue

                # 1. Upload → MinIO
                file_id = str(uuid.uuid4())
                file_path = f"documents/{subject_id}/{file_id}/{file.filename}"
                storage.upload_file(file_bytes, file_path)
                print(f"[UPLOAD] MinIO OK: {file_path}")

                # 2. Lưu Postgres
                doc_id = await document_repo.insert_document(conn, file.filename, subject_id, file_path)
                print(f"[UPLOAD] Postgres OK: doc_id={doc_id}")

                # 3. Trích xuất text
                full_text, sentences = preprocessing.extract_and_preprocess(BytesIO(file_bytes))
                print(f"[UPLOAD] Preprocessing OK: {len(sentences)} câu, text_len={len(full_text)}")

                # 4. MinHash
                minhash_values = minhash.compute_minhash(full_text)
                await document_repo.update_minhash(conn, doc_id, minhash_values)
                print(f"[UPLOAD] MinHash OK: {len(minhash_values)} values")

                # 5. Embedding
                sentence_texts = [s.sentence_text for s in sentences]
                print(f"[UPLOAD] Embedding {len(sentence_texts)} câu...")
                embeddings = embedding.embed_sentences(sentence_texts)
                print(f"[UPLOAD] Embedding OK: {len(embeddings)} vectors, dim={len(embeddings[0]) if embeddings else 0}")

                # 6. Milvus
                print(f"[UPLOAD] Inserting vào Milvus...")
                count = milvus_repo.insert_sentences(
                    document_id=doc_id,
                    file_name=file.filename,
                    subject_id=subject_id,
                    sentences=sentences,
                    embeddings=embeddings,
                )
                print(f"[UPLOAD] Milvus OK: {count} câu đã insert")

                results.append(DocumentUploadResponse(
                    document_id=doc_id,
                    file_name=file.filename,
                    subject_id=subject_id,
                    file_path=file_path,
                    sentences_indexed=count,
                ))

            finally:
                await release_conn(conn)

        except Exception as e:
            import traceback
            print(f"[UPLOAD] LỖI: {e}")
            print(traceback.format_exc())
            errors.append({
                "file_name": file.filename or "unknown",
                "error": str(e),
            })

    return UploadBatchResponse(
        total=len(files),
        succeeded=len(results),
        failed=len(errors),
        documents=results,
        errors=errors,
    )