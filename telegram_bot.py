import logging
import random
from chat import handle_message, suggest
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes, ConversationHandler

TOKEN = '6027508451:AAH6aoVs2gzVZ6m30PlBGGfzAte5k6V8QjE'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update : Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a greeting message to the user when they start the chatbot."""
    user = update.message.from_user
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Hi {user.first_name}, I'm Pulse, your personal health chatbot!")
    await context.bot.send_message(chat_id=update.effective_chat.id, text="How can I help you today? Please describe your symptoms.")
    context.user_data['prev_msg'] = 'start'
    context.user_data['prev_suggestions'] = ['fever', 'cough', 'headache', 'fatigue', 'nausea', 'dizziness', 'chills', 'sore throat']
    await suggest(update, context)

if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    message_handle = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    application.add_handler(message_handle)
    
    application.run_polling()