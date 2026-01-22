from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import time
from ml_model import neural_parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="Neural Invoice API",
    version="2.1",
    description="OCR + MuPDF invoice extraction service"
)

# Pydantic модель результата
class InvoiceData(BaseModel):
    invoice_number: Optional[str] = None
    date: Optional[str] = None
    total_amount: Optional[str] = None

    seller_name: Optional[str] = None
    seller_inn: Optional[str] = None
    buyer_name: Optional[str] = None
    bank_account: Optional[str] = None
    bik: Optional[str] = None

    # Новые/дополнительные реквизиты
    kpp: Optional[str] = None
    ogrn: Optional[str] = None
    bank_name: Optional[str] = None
    correspondent_account: Optional[str] = None  # к/с
    payment_purpose: Optional[str] = None
    total_vat: Optional[str] = None
    company_address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    contract_number: Optional[str] = None

    # Мета-информация об обработке
    processing_method: Optional[str] = None
    ml_enabled: Optional[bool] = None
    fields_found: Optional[int] = None
    fields_detected: Optional[int] = None
    text_quality: Optional[str] = None
    total_pages: Optional[int] = None
    processing_time_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None

# Health check
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ml_model": "loaded" if neural_parser.ml_model else "fallback",
        "device": str(neural_parser.device),
        "version": "2.1"
    }

# Main PDF parsing route
@app.post("/parse-invoice/", response_model=InvoiceData)
async def parse_invoice(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload PDF only")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(pdf_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    start = time.time()

    result = neural_parser.process_pdf(pdf_bytes)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    # дополнительная мета-информация
    result["processing_time_seconds"] = round(time.time() - start, 2)
    result["file_size_bytes"] = len(pdf_bytes)

    return result
