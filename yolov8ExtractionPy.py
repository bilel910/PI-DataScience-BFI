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
import os
import json
import re
from transformers import AutoModelForObjectDetection
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import io
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
# %%


# %%

image = 'DATA2/ABC_EFD3112_2005_page_1.png'

img = Image.open(image)
img


model = YOLO('keremberke/yolov8m-table-extraction')

# set model parameters
model.overrides['conf'] = 0.25  
model.overrides['iou'] = 0.45  
model.overrides['agnostic_nms'] = False  
model.overrides['max_det'] = 1000 

# %%
results = model.predict(img)

print('Boxes: ', results[0].boxes)
render = render_result(model=model, image=img, result=results[0])
render

# %%
for box in results[0].boxes.data.numpy():
    print("Box shape:", box.shape)

# %%
# # Assuming results is a list of detected objects, each containing bounding box information

# # Iterate over all detected objects
# for box in results[0].boxes.data.numpy():
#     # Extract bounding box coordinates
#     x1, y1, x2, y2 = map(int, box[:4])

#     # Crop the table region from the original image
#     cropped_image = img[y1:y2, x1:x2]

#     # Convert the cropped region to an Image object
#     cropped_image = Image.fromarray(cropped_image)

#     # Process the cropped table region (e.g., perform OCR, further analysis, etc.)
#     # ...

#     # Save or display the cropped table region
#     cropped_image 


# %%
x1, y1, x2, y2, _, _ = tuple(int(item) for item in results[0].boxes.data.numpy()[2])
img = np.array(Image.open(image))
#cropping
cropped_image = img[y1:y2, x1:x2]
cropped_image = Image.fromarray(cropped_image)
cropped_image

# %%
# ext_df = pytesseract.image_to_data(cropped_image, output_type=Output.DATAFRAME, config="--psm 6 --oem 3")
# ext_df.head()

# %%
results[0].boxes[2]

# %%
original_image = cv2.imread(image)

box = results[0].boxes[2].xyxy  
print(box)

x1, y1, x2, y2 = map(int, box[0])

cropped_image = original_image[y1:y2, x1:x2]

cv2.imwrite('cropped_table.jpeg', cropped_image)

# %%


os.environ['USE_TF'] = '1'
#os.environ['USE_TORCH'] = '1'




doc = DocumentFile.from_images("cropped_table.jpeg")
print(f"Number of pages: {len(doc)}")

# %%
# predictor = PaddleOCR(use_gpu=False,lang='french')
# result = predictor.ocr('cropped_table.jpeg', cls=True)

predictor = ocr_predictor(pretrained=True)

result = predictor(doc)

# %%
result

# %%
json_export = result.export()
json_export

# %%


# Instantiate a pretrained model
#
# JSON export
json_export = result.export()

def remove_fields(obj, fields):
    if isinstance(obj, list):
        for item in obj:
            remove_fields(item, fields)
    elif isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in fields:
                del obj[key]
            else:
                remove_fields(obj[key], fields)

fields_to_remove = ['confidence', 'page_idx', 'dimensions', 'orientation', 'language', 'artefacts']

remove_fields(json_export, fields_to_remove)

for page in json_export['pages']:
    for block in page['blocks']:
        if 'geometry' in block:
            del block['geometry']
        for line in block.get('lines', []):
            if 'geometry' in line:
                del line['geometry']

modified_json = json.dumps(json_export, separators=(',', ':'),ensure_ascii=False)

print(modified_json)

output_file_path = "OCR_Result2.json"

with open(output_file_path, "w",encoding="utf-8") as output_file:
    output_file.write(modified_json)

print(f"Modified JSON data saved to {output_file_path}")

# %%

def is_valid_format(date_str):
    # Define the regular expression pattern for the format "DD-MMM-YYYY"
    pattern = r'^\d{2}-\w{3}-\d{4}$'
    
    # Check if the date string matches the pattern
    if re.match(pattern, date_str):
        return True
    else:
        return False

# %%

new_file_path = "new_Ocr2.json"
concatenated_values = []

for page in json_export['pages']:
    for block in page['blocks']:
        lines = block["lines"]
        num_lines = len(lines)
        if num_lines > 1:
            for i in range(num_lines - 1):     
                concatenated_value = ""
                current_line = lines[i]
                next_line = lines[i + 1]
                for word in current_line["words"]:
                    concatenated_value += " " + word["value"]
                for word in next_line["words"]:
                    concatenated_value += " " + word["value"]
                concatenated_values.append(concatenated_value.strip()) 
        else:
            for i in range(num_lines):  
                concatenated_value = ""
                current_line = lines[i]
                for word in current_line["words"]:
                    if is_valid_format(word["value"]):
                      concatenated_values.append(word["value"])
                    else:
                      concatenated_value += " " + word["value"]
                    
                concatenated_values.append(concatenated_value.strip())

# Prepare the JSON data
new_json_data = {
    "concatenated_values": concatenated_values
}

# Convert JSON data to a string
new_json_string = json.dumps(new_json_data, indent=4, ensure_ascii=False)

# Write the JSON string to a file
with open(new_file_path, "w", encoding="utf-8") as output_file:
    output_file.write(new_json_string)

print("New JSON file created successfully:", new_file_path)


# %%

synthetic_pages = result.synthesize()
plt.imshow(synthetic_pages[0]); plt.axis('off'); plt.show()


# %%
with open("OCR_Result.txt", "r") as f:
    data = json.load(f)

# %%
def extract_values(page_data):
    values = []
    for block in page_data["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                values.append(word["value"])
    return values

all_values = []
for page in data["pages"]:
    page_values = extract_values(page)
    all_values.extend(page_values)

# %%
model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
print(model.config.id2label)

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("")

# %%
# let's load an example image
file_path = 'cropped_table.jpeg'
image = Image.open(file_path).convert("RGB")
image

# %%

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

pixel_values = detection_transform(image).unsqueeze(0)
pixel_values = pixel_values.to(device)
print(pixel_values.shape)

# %%

with torch.no_grad():
  outputs = model(pixel_values)

print(outputs.logits.shape)

# %%
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# update id2label to include "no object"
id2label = model.config.id2label
id2label[len(model.config.id2label)] = "no object"


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

objects = outputs_to_objects(outputs, image.size, id2label)
print(objects)

# %%
img

# %%


def fig2img(fig):
    
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def visualize_detected_tables(img, det_tables, out_path=None):
    plt.imshow(img, interpolation="lanczos")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        else:
            continue

        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor='none',facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0,
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                                label='Table', hatch='//////', alpha=0.3),
                        Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                                label='Table (rotated)', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
                    fontsize=10, ncol=2)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')

    if out_path is not None:
      plt.savefig(out_path, bbox_inches='tight', dpi=150)

    return fig

fig = visualize_detected_tables(img, objects)
visualized_image = fig2img(fig)

# %%
def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if  (token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0]-bbox[0],
                             token['bbox'][1]-bbox[1],
                             token['bbox'][2]-bbox[0],
                             token['bbox'][3]-bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0]-bbox[3]-1,
                        bbox[0],
                        cropped_img.size[0]-bbox[1]-1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops

print(objects)

# to view corp tables
tokens = []
detection_class_thresholds = {
    "table": 0.5,
    "table rotated": 0.5,
    "no object": 10
}
crop_padding = 10

tables_crops = objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=0)
cropped_table = tables_crops[0]['image'].convert("RGB")
cropped_table

# %%

structure_model = AutoModelForObjectDetection.from_pretrained("bilguun/table-transformer-structure-recognition")
structure_model.to(device)
print("")

structure_model.config.id2label

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

pixel_values = structure_transform(cropped_table).unsqueeze(0)
pixel_values = pixel_values.to(device)
print(pixel_values.shape)

with torch.no_grad():
  outputs = structure_model(pixel_values)

structure_id2label = structure_model.config.id2label
structure_id2label[len(structure_id2label)] = "no object"

cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)
print(cells)

# %%
def get_cell_coordinates_by_row(table_data):
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        row_cells.sort(key=lambda x: x['column'][0])

        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates

cell_coordinates = get_cell_coordinates_by_row(cells)

# %%
cell_coordinates

# %%
# num_rows = len(cell_coordinates)

num_columns = len(cell_coordinates[0]['cells'])

# print("Number of rows:", num_rows)
print("Number of columns:", num_columns)

# %%


file_path = 'cropped_table.jpeg'
image = Image.open(file_path).convert("RGB")


image_processor = AutoImageProcessor.from_pretrained("bilguun/table-transformer-structure-recognition")
model = AutoModelForObjectDetection.from_pretrained("bilguun/table-transformer-structure-recognition")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)
num_rows=0
# convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
    0
]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    if(model.config.id2label[label.item()])=="table row":
        num_rows+=1
print(num_rows)

# %%
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)

# %%
df = pd.DataFrame(np.nan, index=np.arange(num_rows), columns=np.arange(num_columns))

# %%


with open('new_Ocr2.json', 'r', encoding="utf-8") as file:
    json_data = json.load(file)


values = json_data['concatenated_values']

data = [values[i:i+num_columns] for i in range(0, len(values), num_columns)]

df = pd.DataFrame(data)
print(df)



# %%


