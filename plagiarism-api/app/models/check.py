from pydantic import BaseModel


class MatchedSentence(BaseModel):
    query_sentence_index: int
    query_sentence_text: str
    query_page: int
    ref_sentence_text: str
    ref_page: int
    similarity: float


class ReferenceMatch(BaseModel):
    document_id: str
    file_name: str
    subject_id: str
    jaccard_similarity: float        # minhash similarity (lọc thô)
    plagiarism_ratio: float          # số câu d đạo văn từ tài liệu này / tổng câu d
    plagiarized_count: int           # số câu d bị đạo văn từ tài liệu này
    matched_sentences: list[MatchedSentence]


class CheckResponse(BaseModel):
    total_sentences: int             # tổng số câu của tài liệu đẩy lên
    plagiarized_sentences: int       # số câu bị gán nhãn đạo văn
    plagiarism_ratio: float          # P = số câu đạo văn / tổng câu
    is_plagiarized: bool             # P > 80%
    sentence_labels: list[int]       # 0/1 cho từng câu của d
    references: list[ReferenceMatch] # chi tiết từng tài liệu tham chiếu
