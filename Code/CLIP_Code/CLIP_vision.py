from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel
import os
model = CLIPVisionModel.from_pretrained("../../CLIP")
processor = AutoProcessor.from_pretrained("../../CLIP")
image_dir="../../dataset/MARS/images/Q10008133"
images=os.listdir(image_dir)
image_list=[]
for path in images:
    image_path=image_dir+"/"+path
    image_list.append(Image.open(image_path))
print(image_list)
inputs = processor(images=image_list, return_tensors="pt",return_dict=True)

outputs = model(**inputs)
for key in outputs:
    print(key,outputs[key].size())

