import uuid
import asyncpg
from app.services.minhash import jaccard_similarity


async def insert_document(
    conn: asyncpg.Connection,
    file_name: str,
    subject_id: str,
    file_path: str,
) -> str:
    doc_id = str(uuid.uuid4())
    await conn.execute(
        """
        INSERT INTO documents (id, file_name, subject_id, file_path, minhash, created_at)
        VALUES ($1, $2, $3, $4, NULL, NOW())
        """,
        doc_id, file_name, subject_id, file_path,
    )
    return doc_id


async def update_minhash(
    conn: asyncpg.Connection,
    doc_id: str,
    minhash: list[int],
):
    await conn.execute(
        "UPDATE documents SET minhash = $1 WHERE id = $2",
        minhash, doc_id,
    )


async def get_document_by_id(
    conn: asyncpg.Connection,
    doc_id: str,
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        "SELECT * FROM documents WHERE id = $1", doc_id
    )


async def document_exists(
    conn: asyncpg.Connection,
    file_name: str,
    subject_id: str,
) -> bool:
    row = await conn.fetchrow(
        "SELECT id FROM documents WHERE file_name = $1 AND subject_id = $2",
        file_name, subject_id,
    )
    return row is not None


async def find_candidates_by_minhash(
    conn: asyncpg.Connection,
    minhash: list[int],
    subject_id: str,
    threshold: float = 0.5,
) -> list[dict]:
    """
    Lọc thô: lấy các tài liệu tham chiếu cùng môn học,
    tính Jaccard similarity qua MinHash,
    trả về các tài liệu có similarity >= threshold.
    """
    rows = await conn.fetch(
        """
        SELECT id, file_name, subject_id, minhash
        FROM documents
        WHERE subject_id = $1
          AND minhash IS NOT NULL
        """,
        subject_id,
    )

    candidates = []
    for row in rows:
        sim = jaccard_similarity(minhash, list(row["minhash"]))
        if sim >= threshold:
            candidates.append({
                "document_id": str(row["id"]),
                "file_name":   row["file_name"],
                "subject_id":  row["subject_id"],
                "jaccard_similarity": round(sim, 4),
            })

    return sorted(candidates, key=lambda x: x["jaccard_similarity"], reverse=True)
