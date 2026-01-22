import os
import logging
import requests
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO")
)
logger = logging.getLogger(__name__)


class NeuralInvoiceBot:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN missing in .env")

        self.api_url = "http://api:8030/parse-invoice/"

        self.app = Application.builder().token(self.token).build()
        self.setup_handlers()

    def setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(MessageHandler(filters.Document.PDF, self.handle_pdf))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "<b>Neural Invoice Parser</b>\n"
            "Отправьте мне PDF — я извлеку данные",
            parse_mode="HTML"
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "<b>Команды:</b>\n"
            "/start — начать\n"
            "/help — помощь",
            parse_mode="HTML",
        )



    async def handle_pdf(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        doc = update.message.document
        await update.message.reply_text("⏳ Обработка...")

        try:
            file = await context.bot.get_file(doc.file_id)
            pdf_bytes = await file.download_as_bytearray()

            response = requests.post(
                self.api_url,
                files={"file": (doc.file_name, pdf_bytes, "application/pdf")},
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                await update.message.reply_text(
                    self.format_response(data),
                    parse_mode="HTML"
                )
            else:
                await update.message.reply_text(f"Ошибка: {response.text}")

        except Exception as e:
            logger.error(e)
            await update.message.reply_text(f"Ошибка обработки: {e}")

    def format_response(self, d: dict) -> str:
        text = "<b>Извлеченные данные:</b>\n\n"
        for key, label in {
            "invoice_number": "Номер",
            "date": "Дата",
            "total_amount": "Сумма",
            "seller_inn": "ИНН",
            "seller_name": "Поставщик",
            "buyer_name": "Покупатель",
            "bank_account": "Счет",
            "bik": "БИК"
        }.items():
            if key in d:
                text += f"<b>{label}:</b> <code>{d[key]}</code>\n"

        text += "\n<b>Метаданные:</b>\n"
        text += f"Время обработки: {d.get('processing_time_seconds')}s\n"
        text += f"Страниц: {d.get('total_pages')}\n"
        text += f"Качество текста: {d.get('text_quality')}\n"
        return text

    async def handle_text(self, update: Update, context):
        await update.message.reply_text("Отправьте PDF-файл")

    def run(self):
        self.app.run_polling()


def main():
    NeuralInvoiceBot().run()


if __name__ == "__main__":
    main()