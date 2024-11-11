import chromadb
import torch
from torch import nn
import math
import json
import os
from transformers import AutoProcessor,CLIPTokenizer
from PIL import Image

device=torch.device("cuda")
load_steps=1000
text_encoder=torch.load("./CLIP_model2/0_"+str(load_steps)+"_text_encoder.pt")
image_encoder=torch.load("./CLIP_model2/0_"+str(load_steps)+"_image_encoder.pt")
concat_model=torch.load("./CLIP_model2/0_"+str(load_steps)+"_pin_model.pt")
chroma_client=chromadb.PersistentClient(path="./Chroma_base/"+str(load_steps)+"/")     ###存储实体的embedding
collection=chroma_client.get_or_create_collection("entity_emb")
image_encoder.to(device)
text_encoder.to(device)
concat_model.to(device)
def get_test_data():
    file_dir="../../dataset/MARS/"
    entities=[]
    with open(file_dir+"analogy_entities.txt") as f:
        for line in f:
            entities.append(line.strip())
    with open(file_dir+"test.json","r") as f:
        dataset=[]
        for line in f.readlines():
            line=json.loads(line)
            dataset.append(line)
    return dataset,entities
def get_image_emb(batch_image,image_encoder,image_processor):
    image_encoder.eval()
    batch_image_embed = torch.tensor([])
    switch = 0
    for images in batch_image:
        inputs = image_processor(images=images, return_tensors="pt", return_dict=True)
        inputs.to(device)
        outputs = image_encoder(**inputs)
        emb = torch.mean(outputs.image_embeds, dim=0)
        emb = emb.unsqueeze(0)
        if switch == 0:
            batch_image_embed = emb
            switch = 1
        else:
            batch_image_embed = torch.cat((batch_image_embed, emb), dim=0)

    batch_emb = torch.mean(batch_image_embed, dim=0)
    return batch_emb

def divide_sample(dataset):
    data = []
    label = []
    for sample in dataset:
        data_dict = {}
        label_dict = {}
        data_dict["example"] = sample["example"]
        data_dict["question"] = sample["question"]
        label_dict["answer"] = sample["answer"]
        data.append(data_dict)
        label.append(label_dict)
    return data,label
def get_pil_image(image_path):
    images=os.listdir(image_path)
    image_list=[]
    for image in images:
        image_path_temp=os.path.join(image_path,image)
        fp=open(image_path_temp,'rb')
        image=Image.open(fp)
        image_list.append(image.copy())
        image.close()

    return image_list
def get_image(image_path,entities):
    #image_example_heads = []
    #image_example_tails = []
    #image_ques = []
    #image_answers = []
    images=[]
    num=0
    for entity in entities:
        num+=1
        if num%200==0:
            print(num)
        image_path_entity = os.path.join(image_path, entity)

        image_pil = get_pil_image(image_path_entity)
        images.append(image_pil)
    return images
def get_text_emb(text,text_encoder,tokenizer):
    text_encoder.eval()

    text_id=[tokenizer.encode(text)]
    text_id = torch.tensor(text_id).to(device)
    emb=text_encoder(text_id).pooler_output
    emb=emb.squeeze(0)
    return emb

def get_text(text_path,entities,id2text,id2des):
    texts=[]
    for entity in entities:
        text_entity=id2text[entity]
        des_entity=id2des[entity]
        if len(text_entity)<len(des_entity)-3:
            texts.append(text_entity+des_entity)
        else:
            texts.append(text_entity)
    return texts
test_dataset,entities=get_test_data()
id2text={}
id2des={}
with open("../../dataset/MarKG/entity2text.txt","r") as f:
    for line in f:
        id_name=line.split("\t")
        id2text[id_name[0]]=id_name[1]
    f.close()
with open("../../dataset/MarKG/entity2textlong.txt","r") as f:
    for line in f:
        id_name=line.split("\t")
        id2des[id_name[0]]=id_name[1]
data,label=divide_sample(test_dataset)                       ###获取数据


exists=0

if exists==0:
    images=get_image("../../dataset/MARS/images/",entities)
    texts=get_text("../../dataset/MARKG/entity2text.txt",entities,id2text,id2des)
    image_processor=AutoProcessor.from_pretrained("../../CLIP/")
    tokenizer=CLIPTokenizer.from_pretrained("../../CLIP/")
    num=0
    for entity,image_entity,text_entity in zip(entities,images,texts):
        num+=1
        if num%100==0:
            print(num)
        image_emb=get_image_emb(image_entity,image_encoder,image_processor)
        text_emb=get_text_emb(text_entity,text_encoder,tokenizer)
        image_emb=image_emb.unsqueeze(0)
        text_emb=text_emb.unsqueeze(0)
        entity_emb=concat_model(image_emb,text_emb)
        entity_emb=entity_emb.tolist()
        collection.add(embeddings=entity_emb,documents=[entity],ids=[entity])   ###推理获取实体embedding存储
#docs=collection.get(ids=["Q7242",""],include=["documents","embeddings"])
num=0
pdist=nn.PairwiseDistance(p=2)
res=[0,0,0,0]
for sample,label in zip(data,label):
    head_id=sample["example"][0]
    tail_id=sample["example"][1]
    question=sample["question"]
    answer=label["answer"]
    head_emb=collection.get(ids=[head_id],include=["embeddings"])["embeddings"]
    tail_emb=collection.get(ids=[tail_id],include=["embeddings"])["embeddings"]
    question_emb=collection.get(ids=[question],include=["embeddings"])["embeddings"]
    pred_emb=tail_emb+question_emb-head_emb
    #answer_emb=collection.get(ids=[answer],include=["embeddings"])["embeddings"]
    results=collection.query(query_embeddings=pred_emb,n_results=60)    ###查询与q+t-h最近的60个实体，

    results_id=results["documents"][0]
    results_dis=results["distances"]

    num+=1
    switch=0
    for i in range(len(results_id)):
        
        if results_id[i]== answer:
            switch=1
            if i<=20:
                res[0]+=1
            if i>20 and i<40:
                res[1]+=1
            if i>40 and i<=60:
                res[2]+=1
    if switch==0:
        res[3]+=1                                ##查看标签在top几
print(res)
            
    



