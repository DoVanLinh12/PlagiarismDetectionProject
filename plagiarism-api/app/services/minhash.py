from datasketch import MinHash
import asyncpg
NUM_PERM = 128
SHINGLE_SIZE = 3   


def compute_minhash(text: str) -> list[int]:
    """
    Tính MinHash từ text đã clean (chưa tách câu).
    Trả về mảng NUM_PERM số nguyên để lưu vào Postgres INTEGER[].
    """
    m = MinHash(num_perm=NUM_PERM)
    words = text.split()

    if len(words) < SHINGLE_SIZE:
        # Văn bản quá ngắn — dùng unigram thay shingle
        for word in words:
            m.update(word.encode("utf-8"))
    else:
        for i in range(len(words) - SHINGLE_SIZE + 1):
            shingle = " ".join(words[i:i + SHINGLE_SIZE])
            m.update(shingle.encode("utf-8"))

    return [int(v) for v in m.hashvalues]


def jaccard_similarity(minhash_a: list[int], minhash_b: list[int]) -> float:
    """
    Ước tính Jaccard similarity giữa 2 MinHash vector.
    Kết quả trong khoảng [0.0, 1.0].
    """
    if len(minhash_a) != len(minhash_b):
        raise ValueError("Hai MinHash phải có cùng số lượng permutations")
    matches = sum(a == b for a, b in zip(minhash_a, minhash_b))
    return matches / NUM_PERM

async def find_candidates_by_minhash(
    conn: asyncpg.Connection,
    minhash: list[int],
    subject_id: str,
    threshold: float,
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