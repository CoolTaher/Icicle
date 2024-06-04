# Icicle

Welcome to **Icicle**, a GenAI Vision Language Model application. This project leverages the power of the `moondream2` model provided by [vikhyatk](https://huggingface.co/vikhyatk/moondream2). 

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Credits](#credits)

## Introduction

**Icicle** is a Vision Language Model application built with Streamlit. It allows users to upload images and interact with the model by asking questions related to the uploaded image. The model can handle static images as well as GIFs.

## Features

- Image and GIF processing
- Natural language interaction with images
- Powered by the `moondream2` model from [vikhyatk](https://huggingface.co/vikhyatk/moondream2)
- Streamlit-based user interface
- Local and remote deployment capabilities

## Live Demo

You can try out the model directly on Hugging Face Spaces -> https://huggingface.co/spaces/CoolT/Icicle .

## Requirements
Python 3.x
Streamlit
Transformers
Pillow
Streamlit Extras
localtunnel (for remote access)

## Installation

To run this project, follow these steps:

### Clone the Repository

git clone https://github.com/CoolTaher/Icicle.git
cd Icicle
streamlit run app.py

## Performance Note
For better performance, it is recommended to use a local GPU. If you do not have access to a local GPU, you can use Google Colab's T4 inference. This will significantly speed up the processing time.

## Credits
The model moondream2 used in this project is provided by vikhyatk.

