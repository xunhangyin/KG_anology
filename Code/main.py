import torch
import transformers
import json
def get_dataset_json(file_path):
    with open(file_path, 'r',encoding="utf-8") as f:
        dataset=[]
        for line in f.readlines():
            line=json.loads(line)
            dataset.append(line)
        return dataset
def get_des_entity_dict(file_path,file_path_long):
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
def get_des_rel_dict(file_path,file_path_long):
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
dataset_train_json=get_dataset_json('../dataset/MARS/train.json')
dataset_val_json=get_dataset_json('../dataset/MARS/dev.json')
dataset_test_json=get_dataset_json('../dataset/MARS/test.json')
des_entity_dict=get_des_entity_dict("../dataset/MarKG/entity2text.txt","../dataset/MarKG/entity2textlong.txt")
rel_entity_dict=get_des_rel_dict("../dataset/MarKG/relation2text.txt","../dataset/MarKG/relation2textlong.txt")



