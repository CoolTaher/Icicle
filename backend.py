from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import streamlit as st
import pyvips

@dataclass
class Message:
    actor: str
    payload: str

conversation_history = []

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [Message(actor="ai", payload="Hi! How can I help you?")]

initialize_session_state()

def chat_record(speaker, message):
    st.session_state["messages"].append(Message(actor=speaker, payload=message))

@st.cache_resource
def model_loading():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "vikhyatk/moondream2"
    revision="2025-01-09"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer, device

def generate_answer(model, tokenizer, device, image_path, question):
    image = Image.open(image_path)
    
    enc_image = model.encode_image(image)
    
    if isinstance(enc_image, torch.Tensor):
        enc_image = enc_image.to(device)
    
    answer = model.answer_question(enc_image, question, tokenizer)
    
    if isinstance(answer, torch.Tensor):
        answer = answer.cpu().tolist()

    return answer

def process_gif(model, tokenizer, device, gif_path, question):
    gif = Image.open(gif_path)
    frame_count = 0
    answers = []

    while True:
        try:
            gif.seek(frame_count)
            if frame_count % 10 == 0 and gif.getbbox():  # Non-empty frame check
                frame = gif.convert('RGB')
                enc_image = model.encode_image(frame)

                if isinstance(enc_image, torch.Tensor):
                    enc_image = enc_image.to(device)
                
                answer = model.answer_question(enc_image, question, tokenizer)
                
                if isinstance(answer, list):
                    answer = tokenizer.decode(answer, skip_special_tokens=True)
                
                answers.append(answer)

            frame_count += 1
        except EOFError:
            break

    return answers[0] if answers else "No answer could be generated from the GIF."
