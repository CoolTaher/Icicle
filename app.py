from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import backend
from backend import *

def chat_UI(model, tokenizer, device, uploaded_file):
    st.snow()
    st.header('Welcome to ***Icicle*** :ice_cube: ')
    st.divider()

    if uploaded_file is not None:
        if uploaded_file.type == "image/gif":
            process_input(model, tokenizer, device, uploaded_file, is_gif=True)
        else:
            process_input(model, tokenizer, device, uploaded_file, is_gif=False)


def process_input(model, tokenizer, device, uploaded_file, is_gif=False):
    st.subheader("Uploaded Image:")
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    conversation = st.session_state.get("conversation", [])

    user_input = st.session_state.get("user_input", "")

    user_input = st.chat_input("Say something")

    if user_input:
        chat_record("user", user_input)

        if is_gif:
            response = process_gif(model, tokenizer, device, uploaded_file, user_input)
        else:
            response = generate_answer(model, tokenizer, device, uploaded_file, user_input)

        chat_record("ai", response)

        conversation.append({'speaker': 'ai', 'message': response})

        st.session_state["conversation"] = conversation

    for i, chat in enumerate(conversation):
        if chat['speaker'] == 'ai':
            st.text_area(f"AI Response {i+1}", value=chat['message'],max_chars=None, key=None)

def side_bar():
    with st.sidebar:
        logo_image = "logo.png"
        with stylable_container(
            key='Logo_Image',
            css_styles="""
            div[data-testid="stImage"] > img {
                border-radius:50%;
                width:70%;
                margin: -4em 0em 0em 2.5em;
            }
            """,
        ):
            st.image(logo_image)
        st.markdown('<p class="info-text">A GenAI Vision Language Model</p>', unsafe_allow_html=True)
        st.markdown('<p class="upload-text">Upload an Image and Ask!</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an Image file : ", type=["jpg", "jpeg", "png", "gif"])
    return uploaded_file

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def footer():
    st.markdown('<div class="footer"><p>Developed with ❤ by <a style="display: block; text-align: center;" href="linkedin.com/in/taher-p-821817214" target="_blank">Taher !</a></p></div>',unsafe_allow_html=True)


def main():
    local_css("styles.css")
    initialize_session_state()
    model, tokenizer, device = model_loading()
    uploaded_image = side_bar()
    chat_UI(model, tokenizer, device, uploaded_image)
    footer()

if __name__=='__main__':
    main()