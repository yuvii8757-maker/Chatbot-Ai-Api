import os
import random
import json
import torch

from dotenv import load_dotenv
from google import genai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# Create FastAPI app
app = FastAPI()


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
         "http://localhost:5173",
         "http://localhost:5174",
         "http://localhost:5175",
         "http://127.0.0.1:5173",
         "http://127.0.0.1:5174",
         "http://127.0.0.1:5175"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load intents
with open("intents.json", "r") as json_data:
    intents = json.load(json_data)


# Load trained model
FILE = "data.pth"

data = torch.load(FILE, weights_only=False)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]

all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]


# Create neural network
model = NeuralNet(input_size, hidden_size, output_size)

model.load_state_dict(model_state)
model.eval()


bot_name = "MyChatbot"


# Chat request format
class ChatRequest(BaseModel):
    message: str


# Home route
@app.get("/")
def home():
    return {
        "message": "MyChatbot API is running!"
    }


# Chat route
@app.post("/chat")
def chat(request: ChatRequest):

    sentence = request.message

    sentence_tokens = tokenize(sentence)

    X = bag_of_words(sentence_tokens, all_words)

    X = torch.from_numpy(X)

    output = model(X)

    _, predicted = torch.max(output, dim=0)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=0)

    probability = probs[predicted.item()]


    # First try the trained chatbot
    if probability.item() > 0.75:

        for intent in intents["intents"]:

            if tag == intent["tag"]:

                response = random.choice(
                    intent["responses"]
                )

                return {
                    "response": response
                }


    # If chatbot doesn't understand, use Gemini
    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=sentence
        )

        return {
            "response": response.text
        }


    except Exception as e:

        print("Gemini Error:", repr(e))

        return {
            "response": f"Gemini Error: {str(e)}"
        }