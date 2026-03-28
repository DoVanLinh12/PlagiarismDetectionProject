import re
import unicodedata
from io import BytesIO
from dataclasses import dataclass

import pdfplumber
from underthesea import sent_tokenize


@dataclass
class SentenceRecord:
    page_number: int
    sentence_index: int         # thứ tự câu toàn file (bắt đầu từ 0)
    sentence_index_page: int    # thứ tự câu trong trang (bắt đầu từ 0)
    sentence_text: str
    bbox_x0: float
    bbox_y0: float
    bbox_x1: float
    bbox_y1: float


def clean_text(text: str) -> str:
    """
    - Chuẩn hóa unicode NFC (UTF-8)
    - Loại bỏ ký tự đặc biệt, chỉ giữ chữ/số/dấu câu cơ bản
    - Xóa khoảng trắng thừa, xuống dòng thừa
    - Chuyển về chữ thường
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s\.,;:!?\"'()\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _is_valid_sentence(sentence: str, min_words: int = 5) -> bool:
    return len(sentence.split()) >= min_words


def _compute_bbox_for_sentence(
    sent_words: list[str],
    page_words: list[dict],
    word_cursor: int,
    fallback_bbox: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], int]:
    """
    Map các từ trong câu với danh sách words của pdfplumber để lấy bbox.
    Trả về (bbox, new_cursor).
    """
    matched = []
    cursor = word_cursor

    for sw in sent_words:
        while cursor < len(page_words):
            pw_clean = clean_text(page_words[cursor]["text"])
            if pw_clean == sw or sw in pw_clean or pw_clean in sw:
                matched.append(page_words[cursor])
                cursor += 1
                break
            cursor += 1

    if matched:
        x0 = min(w["x0"]     for w in matched)
        y0 = min(w["top"]    for w in matched)
        x1 = max(w["x1"]     for w in matched)
        y1 = max(w["bottom"] for w in matched)
        return (round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)), cursor
    else:
        return fallback_bbox, cursor


def _process_page(
    page_words: list[dict],
    page_number: int,
    global_index: int,
) -> tuple[list[SentenceRecord], str, int]:
    """
    Xử lý 1 trang PDF:
    - Gộp words thành text
    - Clean text
    - Tách câu
    - Map bbox từng câu

    Trả về (records, cleaned_page_text, updated_global_index)
    """
    if not page_words:
        return [], "", global_index

    raw_text = " ".join(w["text"] for w in page_words)
    cleaned_text = clean_text(raw_text)

    # Bbox toàn trang làm fallback
    fallback_bbox = (
        round(min(w["x0"]     for w in page_words), 2),
        round(min(w["top"]    for w in page_words), 2),
        round(max(w["x1"]     for w in page_words), 2),
        round(max(w["bottom"] for w in page_words), 2),
    )

    raw_sentences = sent_tokenize(cleaned_text)
    valid_sentences = [s.strip() for s in raw_sentences if _is_valid_sentence(s)]

    records: list[SentenceRecord] = []
    word_cursor = 0

    for sent_idx, sent in enumerate(valid_sentences):
        sent_words = sent.split()
        bbox, word_cursor = _compute_bbox_for_sentence(
            sent_words, page_words, word_cursor, fallback_bbox
        )

        records.append(SentenceRecord(
            page_number=page_number,
            sentence_index=global_index,
            sentence_index_page=sent_idx,
            sentence_text=sent,
            bbox_x0=bbox[0],
            bbox_y0=bbox[1],
            bbox_x1=bbox[2],
            bbox_y1=bbox[3],
        ))
        global_index += 1

    return records, cleaned_text, global_index


def extract_and_preprocess(
    pdf_bytes: BytesIO,
) -> tuple[str, list[SentenceRecord]]:
    """
    Đầu vào : BytesIO của file PDF
    Đầu ra  :
        full_text_cleaned  — toàn bộ text đã clean (dùng cho MinHash)
        sentences          — list[SentenceRecord] (dùng cho embedding + Milvus)
    """
    all_sentences: list[SentenceRecord] = []
    full_text_parts: list[str] = []
    global_index = 0

    with pdfplumber.open(pdf_bytes) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_words = page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                extra_attrs=["size"],
            )

            records, cleaned_page_text, global_index = _process_page(
                page_words, page_num, global_index
            )

            if cleaned_page_text:
                full_text_parts.append(cleaned_page_text)
            all_sentences.extend(records)

    full_text_cleaned = " ".join(full_text_parts)
    return full_text_cleaned, all_sentences
