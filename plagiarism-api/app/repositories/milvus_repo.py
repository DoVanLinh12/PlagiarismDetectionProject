import uuid
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

from app.core.config import settings
from app.services.preprocessing import SentenceRecord

COLLECTION_NAME = "PlagiarismDetection"
DIM = 768


def connect_milvus():
    connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=str(settings.milvus_port),
    )


def create_collection_if_not_exists() -> Collection:
    connect_milvus()

    if utility.has_collection(COLLECTION_NAME):
        col = Collection(COLLECTION_NAME)
        col.load()
        return col

    fields = [
        FieldSchema(name="id",                  dtype=DataType.VARCHAR,      max_length=36, is_primary=True, auto_id=False),
        FieldSchema(name="document_id",         dtype=DataType.VARCHAR,      max_length=36),
        FieldSchema(name="file_name",           dtype=DataType.VARCHAR,      max_length=255),
        FieldSchema(name="subject_id",          dtype=DataType.VARCHAR,      max_length=100),
        FieldSchema(name="sentence_index",      dtype=DataType.INT64),
        FieldSchema(name="sentence_index_page", dtype=DataType.INT64),
        FieldSchema(name="page_number",         dtype=DataType.INT64),
        FieldSchema(name="sentence_text",       dtype=DataType.VARCHAR,      max_length=2000),
        FieldSchema(name="bbox_x0",             dtype=DataType.DOUBLE),
        FieldSchema(name="bbox_y0",             dtype=DataType.DOUBLE),
        FieldSchema(name="bbox_x1",             dtype=DataType.DOUBLE),
        FieldSchema(name="bbox_y1",             dtype=DataType.DOUBLE),
        FieldSchema(name="embedding",           dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]

    schema = CollectionSchema(fields, description="Plagiarism Detection")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    collection.create_index("embedding", {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    })
    collection.create_index(field_name="document_id", index_name="idx_document_id")
    collection.create_index(field_name="subject_id",  index_name="idx_subject_id")
    collection.load()
    return collection


def insert_sentences(
    document_id: str,
    file_name: str,
    subject_id: str,
    sentences: list[SentenceRecord],
    embeddings: list[list[float]],
) -> int:
    if not sentences:
        return 0

    collection = create_collection_if_not_exists()

    batch_size = 100
    total = 0
    for start in range(0, len(sentences), batch_size):
        batch_sents = sentences[start:start + batch_size]
        batch_embs  = embeddings[start:start + batch_size]

        data = [
            [str(uuid.uuid4())         for _ in batch_sents],
            [document_id]              * len(batch_sents),
            [file_name]                * len(batch_sents),
            [subject_id]               * len(batch_sents),
            [s.sentence_index          for s in batch_sents],
            [s.sentence_index_page     for s in batch_sents],
            [s.page_number             for s in batch_sents],
            [s.sentence_text           for s in batch_sents],
            [float(s.bbox_x0)          for s in batch_sents],
            [float(s.bbox_y0)          for s in batch_sents],
            [float(s.bbox_x1)          for s in batch_sents],
            [float(s.bbox_y1)          for s in batch_sents],
            batch_embs,
        ]

        collection.insert(data)
        total += len(batch_sents)

    collection.flush()
    return total