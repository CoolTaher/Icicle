# Icicle 🧊

Welcome to **Icicle**, a GenAI Vision Language Model application. This project leverages the power of the `moondream2` model provided by [vikhyatk](https://huggingface.co/vikhyatk/moondream2). 

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Usage](#usage)
- [Requirements](#requirements)
- [Credits](#credits)

## Introduction

**Icicle** is a Vision Language Model application built with Streamlit. It allows users to upload images and interact with the model by asking questions related to the uploaded image. The model can handle static images as well as GIFs.

## Features

- Image and GIF processing
- Natural language interaction with images
- Streamlit-based user interface
- Local and remote deployment capabilities

## Usage

- Educational Tool: Helps students and educators by providing detailed descriptions and explanations of images.
- Content Creation: Assists bloggers and content creators in generating captions and narratives for images.
- Accessibility: Provides visually impaired users with textual descriptions of images.
- Social Media Management: Generates engaging content and responses based on images for social media posts.
- Research Assistance: Aids researchers in analyzing visual data and generating related insights.

## Live Demo

You can try out the model directly on Hugging Face Spaces -> https://huggingface.co/spaces/CoolT/Icicle .

## Requirements
- Python 3.x
- Streamlit
- Transformers
- Pillow
- Streamlit Extras
- localtunnel (for remote access)

## Performance Note
For better performance, it is recommended to use a local GPU. If you do not have access to a local GPU, you can use Google Colab's T4 inference. This will significantly speed up the processing time.

## Credits
The model `moondream2` used in this project is provided by [vikhyatk](https://huggingface.co/vikhyatk/moondream2).

