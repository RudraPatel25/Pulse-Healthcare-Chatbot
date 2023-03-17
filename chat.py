import random
import json
import logging
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes, ConversationHandler

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('health-data.json','r') as f:
    diseases = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
names = data["names"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Pulse"

"""
Use this function instead of handle_message function if you want to run chatbot on web:

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    
    name = names[predicted.item()]

    greeting_response = ['greeting', 'goodbye', 'thanks', 'about_you']

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.30:
        for disease in diseases['diseases']:
            if name == disease['name']:
                if name in greeting_response:
                    greeting_return = random.choice(disease['treatments'])
                    return greeting_return
                else:
                    disease_value = "You may diagonise with {name}.\n"
                    disease_name = name
                    disease_sentence = disease_value.format(name=disease_name)
                    treatment_sentence = "Treatment: " + random.choice(disease['treatments']) + "."
                    recommend_sentence = "Please remember this is just a recommendation, consult with a doctor for professional advice."
                    return '\n'.join([disease_sentence, treatment_sentence, recommend_sentence])

    return "Sorry, I couldn't find a match."

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
"""

async def suggest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'prev_suggestions' in context.user_data:
        prev_suggestions = context.user_data['prev_suggestions']
        message = "You can try these symptoms: " + ", ".join(prev_suggestions)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=message)
    return ConversationHandler.END

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user's messages and respond with appropriate output."""
    message = update.message.text
    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    
    name = names[predicted.item()]

    greeting_response = ['greeting', 'about_you']
    thanking_response = ['goodbye', 'thanks']

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.20:
        for disease in diseases['diseases']:
            if name == disease['name']:
                if name in greeting_response:
                    greeting_return = random.choice(disease['treatments'])
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=greeting_return)
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="How can I help you today? Please describe your symptoms.")
                    context.user_data['prev_msg'] = 'start'
                    context.user_data['prev_suggestions'] = ['fever', 'cough', 'headache', 'fatigue', 'nausea', 'dizziness', 'chills', 'sore throat']
                    await suggest(update, context)
                elif name in thanking_response:
                    thanking_return = random.choice(disease['treatments'])
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=thanking_return)
                else:
                    disease_value = "You may diagnose with {name}.\n"
                    disease_name = name
                    disease_sentence = disease_value.format(name=disease_name)
                    treatment_sentence = "Treatment: " + random.choice(disease['treatments']) + "."
                    recommend_sentence = "Please remember this is just a recommendation, consult with a doctor for professional advice."
                    await context.bot.send_message(chat_id=update.effective_chat.id, text='\n'.join([disease_sentence, treatment_sentence]))
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=recommend_sentence)

    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I couldn't find a match.")
        context.user_data['prev_msg'] = 'start'
        context.user_data['prev_suggestions'] = ['fever', 'cough', 'headache', 'fatigue', 'nausea', 'dizziness', 'chills', 'sore throat']
        await suggest(update, context)