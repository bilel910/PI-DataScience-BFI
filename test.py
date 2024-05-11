from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
import os 
import sys
from flask import jsonify
import csv 
from pdf2image import convert_from_path
import os


def pdf_to_images(pdf_path):
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    output_folder = 'C:/Users/kortb/Desktop/BFI/PdfToImg'  # Replace with the folder where you want to save the images
    os.makedirs(output_folder, exist_ok=True)
    

    # Save images
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i + 1}.png')
        image.save(image_path, 'PNG')


path="C:/Users/kortb/Desktop/BFI/DATA/ABC_EFD311213.pdf"
pdf_to_images(path)