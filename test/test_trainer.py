import sys
import unittest
sys.path.append('../')
from Code.model_trainer import ModelTrainer
from Code.main import get_des_entity_dict,get_des_rel_dict
import json
class TestCase(unittest.TestCase):
    def setUp(self):
        self.pretrained_llama="../llama3"
        self.pretrained_clip="../CLIP"
        self.output_dir="../output_model"
        self.entity_des=get_des_entity_dict("../dataset/MarKG/entity2text.txt","../dataset/MarKG/entity2textlong.txt")
        self.relation_des=get_des_rel_dict("../dataset/MarKG/relation2text.txt","../dataset/MarKG/relation2textlong.txt")
    def get_data(self):
        with open("./test.json", 'r', encoding="utf-8") as f:
            dataset = []
            for line in f.readlines():
                line = json.loads(line)
                dataset.append(line)
        return dataset
    def test_trainer(self):
        dataset=self.get_data()
        trainer=ModelTrainer(self.pretrained_llama, self.pretrained_clip, self.output_dir)
        trainer.train(dataset,self.entity_des,self.relation_des)



if __name__ =="__main__":
    unittest.main()