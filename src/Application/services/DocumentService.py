import datetime
import io
from typing import Literal
from minio.error import S3Error
from io import BytesIO
from src.Infrastructure.dbcontext.context import settings
from src.Infrastructure.minio.index import minio_client
from src.Infrastructure.embedding_model.index import local_embed_model
from src.Infrastructure.llm.index import local_llm
from src.Infrastructure.elastic_search.index import (
    create_elasticsearch_index,
    index,
)
from markitdown import MarkItDown
from loguru import logger
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    Response,
    AsyncStreamingResponse,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode

md = MarkItDown()
md_parser = MarkdownNodeParser()
sentence_splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,
)
pipeline = IngestionPipeline(
    transformations=[md_parser, sentence_splitter, local_embed_model]
)


class DocumentService:
    def __init__(self):
        pass

    def store_to_minio(
        self,
        file_size: int,
        file_stream: BytesIO,
        filename: str | None,
        file_content_type: Literal["application/pdf"],
    ) -> bool | S3Error:
        try:
            object_name = filename if filename is not None else "document"

            logger.info(f"Checking if file already exists in MinIO: {object_name}")
            try:
                minio_client.stat_object(
                    bucket_name=settings.MINIO_BUCKET_NAME,
                    object_name=object_name,
                )
                logger.warning(f"File already exists in MinIO: {object_name}")
                raise FileExistsError(
                    f"Document with filename '{object_name}' already exists"
                )
            except S3Error as err:
                if err.code != "NoSuchKey":
                    logger.error(f"Error checking file existence: {err}")
                    return err

            logger.info(
                f"Storing file to MinIO: {object_name}, size: {file_size} bytes"
            )
            minio_client.put_object(
                bucket_name=settings.MINIO_BUCKET_NAME,
                object_name=object_name,
                data=file_stream,
                length=file_size,
                content_type=file_content_type,
            )
            logger.info(f"File stored successfully: {object_name}")
            return True

        except S3Error as err:
            logger.error(f"Failed to store file to MinIO: {filename}, error: {err}")
            return err

    def parse_document(self, filename: str | None):
        markdown_text = None
        response = None
        try:
            logger.info(f"Parsing document: {filename}")
            response = minio_client.get_object(
                bucket_name=settings.MINIO_BUCKET_NAME,
                object_name=filename if filename is not None else "document",
            )
            pdf_bytes = response.read()
            pdf_buffer = io.BytesIO(pdf_bytes)
            result = md.convert(pdf_buffer, file_extension=".pdf")
            markdown_text = result.markdown
            doc = Document(
                text=markdown_text,
                metadata={
                    "filename": filename,
                    "author": "system",
                    "category": "markdown",
                    "parsed_at": datetime.datetime.now().isoformat(),
                    "domain": "electronic finance",
                    "classification": "open-source documnets",
                },
            )
            nodes = pipeline.run(documents=[doc], include_metadata=True)

            # Inspect nodes
            logger.info(f"Document parsed successfully: {filename}")
            logger.info(f"Total nodes extracted: {len(nodes)}")

            for i, node in enumerate(nodes):
                logger.info(f"Node {i}:")
                logger.info(f"  - Text preview: {node.get_content()[:200]}...")
                logger.info(f"  - Metadata: {node.metadata}")
                logger.info(f"  - Node ID: {node.node_id}")

            logger.info(f"Storing embeddings in Elasticsearch for: {filename}")
            index = create_elasticsearch_index(nodes)
            logger.info(
                f"Embeddings successfully stored in Elasticsearch for: {filename}"
            )
        except S3Error as err:
            logger.error(
                f"MinIO Error while parsing document: {filename}, error: {err}"
            )
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    async def query_pipeline(self, prompt: str) -> RESPONSE_TYPE:
        """Query the document index using Elasticsearch and Llama-8B."""
        try:
            logger.info(f"Querying with prompt: {prompt}")

            query_engine = index.as_query_engine(
                streaming=True,
                similarity_top_k=5,
                llm=local_llm,
                # vector_store_query_mode=VectorStoreQueryMode.HYBRID, paid feature
            )
            response = await query_engine.aquery(prompt)
            logger.info(f"Query response generated successfully")

            return response

        except Exception as e:
            logger.error(f"Error in query_pipeline: {e}")
            raise

    @staticmethod
    async def stream_response(response: RESPONSE_TYPE):
        if isinstance(response, AsyncStreamingResponse):
            async for token in response.async_response_gen():
                yield token
        elif isinstance(response, Response):
            yield response.response
        else:
            raise TypeError("Unknown response type")
