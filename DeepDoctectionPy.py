# %%
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# import pytesseract
# from pytesseract import Output

from ultralyticsplus import YOLO, render_result
from PIL import Image
import deepdoctection as dd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
# from paddleocr import PaddleOCR, draw_ocr 
from matplotlib import pyplot as plt 
import cv2 
from pathlib import Path
import os
from IPython.core.display import HTML
from bs4 import BeautifulSoup


# %%
def DeepDoctectionPy(image_path):

    config_overwrite = ["LANGUAGE='fra'"]

    analyzer = dd.get_dd_analyzer(config_overwrite)


# %%

    df = analyzer.analyze(path=image_path)

    df.reset_state()

    # %%

    # %%
    doc=iter(df)


    # %%
    page = next(doc)

    table = page.tables[0]
    table.get_attribute_names()

    # %%
    print(f" number of rows: {table.number_of_rows} \n number of columns: {table.number_of_columns} \n reading order: {table.reading_order}")


    # %%
    html_content = HTML(page.tables[0].html)

    # plt.imshow(html_content); plt.axis('off'); plt.show()

    tables = pd.read_html(page.tables[0].html)

    # for table in tables:
    #  print(table, '\n\n')
    
    for i, table in enumerate(tables, start=1):
        file_name = f'table_{i}.csv'
        table.to_csv(file_name)
    # %%
    return file_name

# %%






DeepDoctectionPy("1")