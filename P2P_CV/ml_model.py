import io
import re
import cv2
import fitz
import numpy as np
from PIL import Image
import logging

# Попытка импортировать torch, если есть
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# CNN MODEL (опционально — используется только если есть)
if TORCH_AVAILABLE:
    class InvoiceCNN(nn.Module):
        """Простая CNN для классификации областей (необязательно)."""
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.fc = nn.Sequential(
                nn.Linear(128 * 28 * 28, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.conv(x)
            x = x.flatten(1)
            return self.fc(x)
else:
    # Если torch нет, создаём заглушку
    class InvoiceCNN:
        def __init__(self, *args, **kwargs):
            pass

# IMAGE PREPROCESSING + OCR
class EnhancedOCR:
    """Улучшение изображения + OCR (Tesseract)."""

    def __init__(self):
        self.sharp_kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def enhance(self, pil_img: Image.Image) -> Image.Image:
        try:
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            img = cv2.filter2D(img, -1, self.sharp_kernel)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = self.clahe.apply(l)
            img = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return pil_img

    def extract_text(self, pil_img: Image.Image) -> str:
        import pytesseract
        try:
            return pytesseract.image_to_string(pil_img, lang="rus+eng")
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

# FIELD DETECTOR
class FieldDetector:
    """Обнаружение потенциальных полей документа через OpenCV."""

    def detect(self, pil_img: Image.Image) -> int:
        try:
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            count = 0
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if 50 < w < 600 and 20 < h < 200:
                    count += 1
            return count
        except Exception as e:
            logger.error(f"Field detection error: {e}")
            return 0

# MAIN PARSER ENGINE
class NeuralInvoiceParser:

    def __init__(self, model_path: str = None):
        self.device = "cpu"
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ocr = EnhancedOCR()
        self.detector = FieldDetector()

        # CNN model
        self.ml_model = None
        if TORCH_AVAILABLE:
            try:
                self.ml_model = InvoiceCNN()
                if model_path:
                    self.ml_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.ml_model.to(self.device).eval()
                logger.info("ML model loaded successfully")
            except Exception as e:
                logger.warning(f"ML model load failed: {e}")
                self.ml_model = None
        else:
            logger.info("Torch not available — ML model disabled")

        # Regex patterns
        self.patterns = {
            "invoice_number": r"(?:Счёт-оферта\s*№|Счет[- ]оферта\s*№|№)\s*([A-Za-z0-9\-\/]+)",
            "date": r"(?:от\s*)?(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
            "total_amount": r"(?:Итого\s*к оплате|Итого|Сумма|Всего|total)[^\d]{0,10}([\d\s,.]+)",
            "seller_inn": r"ИНН(?:\s*[:\-]?\s*|\/)\s*(\d{9,12})",
            "bik": r"(?:БИК|BIC)[\s:]+(\d{9})",
            "bank_account": r"(?:р\/с|рс|Р\/С|Расчётный счёт|account)[\s:]+(\d+)",
            "seller_name": r"(?:Продавец|Поставщик|Seller)[\s:]+([^\n]+)",
            "buyer_name": r"(?:Покупатель|Buyer)[\s:]+([^\n]+)",
            "kpp": r"КПП[:\s]+(\d{9})",
            "ogrn": r"ОГРН[:\s]+(\d{13})",
            "bank_name": r"(Банк\s+[A-Za-zА-Яа-я0-9\"'«»\s\.]+|АО\s+\"?[A-Za-zА-Яа-я0-9\s\.]+\"?(?:\s*г\.[A-Za-zА-Яа-я]+)?)",
            "payment_purpose": r"(?:Назначение платежа|Purpose)[\s:]+([^\n]+)",
            "correspondent_account": r"(?:к\/с|корр\.? счёт|корреспондентский счёт)[\s:]+(\d+)",
            "total_vat": r"(?:НДС|VAT)[^\d]{0,10}([\d\s,.]+)",
            "company_address": r"(?:Адрес|Address)[:\s]+([^\n]+)",
            "phone": r"(?:Тел\.?|Phone)[\s:]+([\+\d\-\s\(\)]+)",
            "email": r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
            "contract_number": r"(?:Договор(?:а)?(?:\s*поставки)?|Contract)[^\d]{0,10}(№\s*[A-Za-z0-9\-\/]+)",
        }

    # PDF → TEXT + IMAGES
    def extract_text_and_images(self, pdf_bytes: bytes):
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            return None, None, f"PDF parse error: {e}"

        all_text = ""
        images = []

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text()
            all_text += f"\n{text}"
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)

        return all_text, images, None

    # MAIN PROCESSING
    def process_pdf(self, pdf_bytes: bytes) -> dict:
        text, images, error = self.extract_text_and_images(pdf_bytes)
        if error:
            return {"error": error}

        if not text.strip():
            return {"error": "Не удалось извлечь текст из PDF"}

        ocr_text_total = ""
        fields_total = 0
        for img in images:
            enhanced = self.ocr.enhance(img)
            ocr_text_total += "\n" + self.ocr.extract_text(enhanced)
            fields_total += self.detector.detect(enhanced)

        full_text = text + "\n" + ocr_text_total

        data = {}
        for key, pattern in self.patterns.items():
            m = re.search(pattern, full_text, re.IGNORECASE)
            if m:
                data[key] = m.group(1).strip()

        data.update({
            "processing_method": "mupdf_ocr_hybrid",
            "ml_enabled": self.ml_model is not None,
            "fields_detected": fields_total,
            "fields_found": len(data),
            "text_quality": "good" if len(full_text) > 300 else "medium",
            "total_pages": len(images)
        })

        return data

# GLOBAL INSTANCE
neural_parser = NeuralInvoiceParser(model_path="cnn_invoice_model.pth")
