from datasketch import MinHash

NUM_PERM = 128      # số hash functions — càng cao càng chính xác, tốn RAM hơn
SHINGLE_SIZE = 3    # 3-gram theo từ


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
