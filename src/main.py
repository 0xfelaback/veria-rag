# technical papers

from contextlib import asynccontextmanager
import datetime
from io import BytesIO
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from minio import S3Error
from src.Application.services.DocumentService import DocumentService
from src.Infrastructure.minio.index import minio_client
from src.Infrastructure.dbcontext.context import settings
from loguru import logger
import asyncio

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
        logger.info(f"Checking for bucket: {settings.MINIO_BUCKET_NAME}")
        if not minio_client.bucket_exists(settings.MINIO_BUCKET_NAME):
            logger.info(f"Creating bucket: {settings.MINIO_BUCKET_NAME}")
            minio_client.make_bucket(settings.MINIO_BUCKET_NAME)
            logger.info(f"Bucket created successfully")
        else:
            logger.info(f"Bucket already exists: {settings.MINIO_BUCKET_NAME}")
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


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info(f"Upload PDF endpoint called with file: {file.filename}")
    document_service = DocumentService()
    if file.content_type != "application/pdf":
        logger.warning(f"Invalid file type: {file.content_type}")
        return {"error": "Only PDF files are allowed"}, 400

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
        )
    except FileExistsError as e:
        logger.warning(f"File already exists: {filename}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error storing file to MinIO: {filename}, error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if minio_store_result is True:
        logger.info(f"File uploaded successfully: {filename}")
        document_service.parse_document(filename)
        return {"message": "Success, PDF uploaded successfully", "filename": filename}
    else:
        error_message = str(minio_store_result)
        logger.error(f"Failed to upload file: {filename}, error: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/prompt")
async def prompt(request: PromptRequest):
    logger.info(f"Prompt endpoint called with query: {request.query}")
    document_service = DocumentService()
    try:
        response = await document_service.query_pipeline(request.query)
        logger.info(f"Query processed successfully")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
