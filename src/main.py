from contextlib import asynccontextmanager
import datetime
from io import BytesIO
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from minio import S3Error
from src.Application.services.DocumentService import DocumentService, FileType
from src.Infrastructure.minio.index import minio_client
from src.Infrastructure.dbcontext.context import settings
from loguru import logger

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/{current_time}-log-veria.log"
logger.add(
    log_filename,
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info(
            f"Checking buckets: {settings.MINIO_BUCKET_NAME}, {settings.MINIO_BUCKET_NAME_MD}, {settings.MINIO_BUCKET_NAME_DOCX}, {settings.MINIO_BUCKET_NAME_DOCX_MD}, {settings.MINIO_BUCKET_NAME_PDF_MD}"
        )
        for bucket_name in [
            settings.MINIO_BUCKET_NAME,
            settings.MINIO_BUCKET_NAME_MD,
            settings.MINIO_BUCKET_NAME_DOCX,
            settings.MINIO_BUCKET_NAME_DOCX_MD,
            settings.MINIO_BUCKET_NAME_PDF_MD,
        ]:
            if not minio_client.bucket_exists(bucket_name):
                logger.info(f"Creating bucket: {bucket_name}")
                minio_client.make_bucket(bucket_name)
                logger.info(f"Bucket created successfully")
            else:
                logger.info(f"Bucket already exists: {bucket_name}")
    except S3Error as e:
        logger.error(f"Error connecting to MinIO: {e}")
        raise

    yield


app = FastAPI(lifespan=lifespan)


class PromptRequest(BaseModel):
    query: str


@app.get("/health")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "ok"}


@app.post("/upload-md")
async def upload_md(file: UploadFile = File(...)):
    logger.info(f"Upload markdown endpoint called with file: {file.filename}")
    document_service = DocumentService()
    if file.content_type != "text/markdown":
        logger.warning(f"Invalid file type: {file.content_type}")
        error = {"error": "Only markdown document files are allowed in this endpoint"}
        raise HTTPException(status_code=400, detail=str(error))
    logger.info(f"Reading file content: {file.filename}")
    file_content = await file.read()
    file_size = len(file_content)
    file_stream = BytesIO(file_content)
    file_content_type = file.content_type
    filename = file.filename
    logger.info(f"Storing file to MinIO: {filename}, size: {file_size} bytes")
    try:
        minio_store_result = document_service.store_to_minio(
            file_size=file_size,
            file_stream=file_stream,
            filename=filename,
            file_content_type=file_content_type,
            file_type=FileType.MD,
        )
    except FileExistsError as e:
        logger.warning(f"File already exists: {filename}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error storing file to MinIO: {filename}, error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if minio_store_result is True:
        logger.info(f"File uploaded successfully: {filename}")
        document_service.parse_document(filename, file_type=FileType.MD)
        return {
            "message": "Success, markdown file uploaded successfully",
            "filename": filename,
        }
    else:
        error_message = str(minio_store_result)
        logger.error(f"Failed to upload file: {filename}, error: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info(f"Upload PDF endpoint called with file: {file.filename}")
    document_service = DocumentService()
    if file.content_type != "application/pdf":
        logger.warning(f"Invalid file type: {file.content_type}")
        error = {"error": "Only PDF files are allowed in this endpoint"}
        raise HTTPException(status_code=400, detail=str(error))

    logger.info(f"Reading file content: {file.filename}")
    file_content = await file.read()
    file_size = len(file_content)
    file_stream = BytesIO(file_content)
    file_content_type = file.content_type
    filename = file.filename
    logger.info(f"Storing file to MinIO: {filename}, size: {file_size} bytes")
    try:
        minio_store_result = document_service.store_to_minio(
            file_size=file_size,
            file_stream=file_stream,
            filename=filename,
            file_content_type=file_content_type,
            file_type=FileType.PDF,
        )
    except FileExistsError as e:
        logger.warning(f"File already exists: {filename}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error storing file to MinIO: {filename}, error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if minio_store_result is True:
        logger.info(f"File uploaded successfully: {filename}")
        document_service.parse_document(filename, file_type=FileType.PDF)
        return {"message": "Success, PDF uploaded successfully", "filename": filename}
    else:
        error_message = str(minio_store_result)
        logger.error(f"Failed to upload file: {filename}, error: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/upload-docx")
async def upload_docx(file: UploadFile = File(...)):
    logger.info(f"Upload DOCX endpoint called with file: {file.filename}")
    document_service = DocumentService()
    allowed_content_types = {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "application/octet-stream",
    }
    filename = file.filename or ""
    if file.content_type not in allowed_content_types and not filename.lower().endswith(
        ".docx"
    ):
        logger.warning(f"Invalid file type: {file.content_type}")
        error = {"error": "Only DOCX files are allowed in this endpoint"}
        raise HTTPException(status_code=400, detail=str(error))

    logger.info(f"Reading file content: {filename}")
    file_content = await file.read()
    file_size = len(file_content)
    file_stream = BytesIO(file_content)
    file_content_type = file.content_type or "application/octet-stream"
    logger.info(f"Storing file to MinIO: {filename}, size: {file_size} bytes")
    try:
        minio_store_result = document_service.store_to_minio(
            file_size=file_size,
            file_stream=file_stream,
            filename=filename,
            file_content_type=file_content_type,
            file_type=FileType.DOCX,
        )
    except FileExistsError as e:
        logger.warning(f"File already exists: {filename}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error storing file to MinIO: {filename}, error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if minio_store_result is True:
        logger.info(f"File uploaded successfully: {filename}")
        document_service.parse_document(filename, file_type=FileType.DOCX)
        return {"message": "Success, DOCX uploaded successfully", "filename": filename}
    else:
        error_message = str(minio_store_result)
        logger.error(f"Failed to upload file: {filename}, error: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/prompt")
async def prompt(request: PromptRequest):
    logger.info(f"Prompt endpoint called with query: {request.query}")
    document_service = DocumentService()

    def generate_response():
        try:
            token_stream = document_service.query_pipeline(request.query)
            # print(list(token_stream))
            for token in token_stream:
                yield f"data: {token}\n\n"
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            yield f"data: Error: {str(e)}\n\n"

        logger.info(f"Query processed successfully")

    return StreamingResponse(generate_response(), media_type="text/event-stream")
