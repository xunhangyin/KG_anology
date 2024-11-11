import torch
import transformers
import json
from model_trainer import ModelTrainer
import argparse
from get_parse import parser_add_main_args
import os
def get_dataset_json(file_path):                 ###读取每个sample,h,t,q,a
    with open(file_path, 'r',encoding="utf-8") as f:
        dataset=[]
        for line in f.readlines():
            line=json.loads(line)
            dataset.append(line)
        return dataset
def get_des_entity_dict(file_path,file_path_long):      ###获取每个实体的文字描述，以字典形式存储,形式为{id:desc}，如{Q33012:this is an animal}（CLIP粗排没有用到）
    des_entity_dict={}
    with open(file_path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.split("\t")
            entity,name=line[0],line[1]
            if entity not in des_entity_dict:
                des_entity_dict[entity]=name
    with open(file_path_long,"r",encoding="utf-8") as f:
        for line in f:
            line=line.split("\t")
            entity,des=line[0],line[1]
            if entity in des_entity_dict:
                des_entity_dict[entity]=des_entity_dict[entity]+" "+des
    return des_entity_dict
def get_des_rel_dict(file_path,file_path_long):     ###获取关系描述（CLIP粗排没有用到）
    des_rel_dict={}
    with open(file_path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.split("\t")
            rel,name=line[0],line[1]
            if rel not in des_rel_dict:
                des_rel_dict[rel]=name
    with open(file_path_long,"r",encoding="utf-8") as f:
        for line in f:
            line=line.split("\t")
            rel,des=line[0],line[1]
            if rel in des_rel_dict:
                des_rel_dict[rel]=des_rel_dict[rel]+des
    return des_rel_dict
def get_entities(file_path):          ##获取所有类比推理相关实体的列表[Q3011,Q332423,.....]
    entities=[]
    with open(file_path,"r",encoding="utf-8") as f:
        for line in f:
            entities.append(line.strip())
    return entities
parser=argparse.ArgumentParser()
parser_add_main_args(parser)
args = parser.parse_args()
dataset_train_json=get_dataset_json('../../dataset/MARS/train.json')
dataset_val_json=get_dataset_json('../../dataset/MARS/dev.json')
dataset_test_json=get_dataset_json('../../dataset/MARS/test.json')
des_entity_dict=get_des_entity_dict("../../dataset/MarKG/entity2text.txt","../../dataset/MarKG/entity2textlong.txt")
rel_entity_dict=get_des_rel_dict("../../dataset/MarKG/relation2text.txt","../../dataset/MarKG/relation2textlong.txt")
entities=get_entities("../../dataset/MARS/analogy_entities.txt")
trainer=ModelTrainer("../../CLIP/",args.output_dir,args.lr,args.batch_size,args.k_shot)      
trainer.train(dataset_train_json,des_entity_dict,rel_entity_dict,entities)     ###在model trainer类中训练


