from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
from PIL import Image
import os
import random
from transformers import AutoProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer
from concat_model import concat_model_linear
from torch import nn
import datetime

device = torch.device("cuda:0")


class ModelTrainer(object):
    def __init__(self, pretrained_clip, output_dir, learning_rate, batch_size, k_shot, num_epochs=5,
                 accumulate_steps=4, checkpoint_steps=1000, tempature=0.8):
        self.pretrained_clip = pretrained_clip
        self.output_dir = output_dir
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.accu_steps = accumulate_steps
        self.checkpoint_steps = checkpoint_steps
        self.tempeature = tempature
        self.weight_decay = 1e-4
        self.k_shot = k_shot

    def divide_sample(self, dataset):
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
        return data, label

    def get_pil_image(self, image_path):
        images = os.listdir(image_path)
        image_list = []
        for image in images:
            image_path_temp = os.path.join(image_path, image)
            fp = open(image_path_temp, 'rb')
            image = Image.open(fp)
            image_list.append(image)

        return image_list

    def get_image(self, image_path, data_batch, label_batch):
        image_example_heads = []
        image_example_tails = []
        image_ques = []
        image_answers = []
        for data, label in zip(data_batch, label_batch):
            image_example_head_id = data["example"][0]
            image_example_tail_id = data["example"][1]
            image_question_id = data["question"]
            image_answer_id = label["answer"]
            image_example_head_path = os.path.join(image_path, image_example_head_id)
            image_example_tail_path = os.path.join(image_path, image_example_tail_id)
            image_question_path = os.path.join(image_path, image_question_id)
            image_answer_path = os.path.join(image_path, image_answer_id)
            image_example_head = self.get_pil_image(image_example_head_path)
            image_example_tail = self.get_pil_image(image_example_tail_path)
            image_question = self.get_pil_image(image_question_path)
            image_answer = self.get_pil_image(image_answer_path)
            image_example_heads.append(image_example_head)
            image_example_tails.append(image_example_tail)
            image_ques.append(image_question)
            image_answers.append(image_answer)
        return image_example_heads, image_example_tails, image_ques, image_answers

    def get_text(self, text_path, data_batch, label_batch, id2text, id2des):
        text_example_heads = []
        text_example_tails = []
        text_ques = []
        text_answers = []
        for data, label in zip(data_batch, label_batch):
            text_example_head_id = data["example"][0]
            text_example_tail_id = data["example"][1]
            text_question_id = data["question"]
            text_answer_id = label["answer"]
            text_example_head = id2text[text_example_head_id]
            text_example_tail = id2text[text_example_tail_id]
            text_question = id2text[text_question_id]
            text_answer = id2text[text_answer_id]
            text_example_head_des = id2des[text_example_head_id]
            text_example_tail_des = id2des[text_example_tail_id]
            text_question_des = id2des[text_question_id]
            text_answer_des = id2des[text_answer_id]
            if len(text_example_head_des) < len(text_example_head) + 3:
                text_example_heads.append(text_example_head)
            else:
                text_example_heads.append(text_example_head + text_example_head_des)
            if len(text_example_tail_des) < len(text_example_tail) + 3:
                text_example_tails.append(text_example_tail)
            else:
                text_example_tails.append(text_example_tail + text_example_tail_des)
            if len(text_question_des) < len(text_question) + 3:
                text_ques.append(text_question)
            else:
                text_ques.append(text_question + text_question_des)
            if len(text_answer_des) < len(text_answer) + 3:
                text_answers.append(text_answer)
            else:
                text_answers.append(text_answer + text_answer_des)
        return text_example_heads, text_example_tails, text_ques, text_answers

    def image_to_emb(self, batch_image, image_encoder, image_processor):
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
        return batch_image_embed

    def text_to_emb(self, batch_text, text_encoder, tokenizer):
        batch_max_length = 0
        text_ids = []
        attention_mask = []
        for text in batch_text:
            text_id = tokenizer.encode(text)
            attention_mask.append([1 for _ in range(len(text_id))])
            batch_max_length = max(batch_max_length, len(text_id))
            text_ids.append(text_id)
        for i in range(len(text_ids)):
            while len(text_ids[i]) < batch_max_length:
                text_ids[i].append(49407)
                attention_mask[i].append(0)
        text_ids = torch.tensor(text_ids).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)
        emb = text_encoder(input_ids=text_ids, attention_mask=attention_mask, return_dict=True).pooler_output
        return emb

    def get_entity_image_emb(self, k_samples, image_path, image_encoder, image_processor):
        sample_images = []
        sample_texts = []
        for sample in k_samples:
            image_path_temp = os.path.join(image_path, sample)
            sample_images.append(self.get_pil_image(image_path_temp))
        sample_emb = self.image_to_emb(sample_images, image_encoder, image_processor)
        return sample_emb

    def get_entity_text_emb(self, k_samples, id2text, id2des, text_encoder, tokenizer):
        sample_texts = []
        for sample in k_samples:
            name = id2text[sample]
            des = id2des[sample]
            if len(des) < len(name) + 2:
                sample_texts.append(id2text[sample])
            else:
                sample_texts.append(id2text[sample] + "\n" + id2des[sample])
        sample_emb = self.text_to_emb(sample_texts, text_encoder, tokenizer)
        return sample_emb

    def get_entities_emb(self, batch_k_samples, image_path, id2text, id2des, image_encoder, text_encoder,
                         image_processor, tokenizer, pin_model):
        num = 0
        for k_samples in batch_k_samples:
            k_image_embeds = self.get_entity_image_emb(k_samples, image_path, image_encoder, image_processor)
            k_text_embeds = self.get_entity_text_emb(k_samples, id2text, id2des, text_encoder, tokenizer)
            k_sample_embed = pin_model(k_image_embeds, k_text_embeds)
            k_sample_embed = k_sample_embed.unsqueeze(0)

            if num == 0:
                batch_entities_embed = k_sample_embed
            else:
                batch_entities_embed = torch.cat((batch_entities_embed, k_sample_embed), dim=0)
            num += 1
        return batch_entities_embed

    def get_dist(self, rel_ht, rel_qa):
        pdist = nn.PairwiseDistance(p=2)
        return torch.exp(-pdist(rel_ht, rel_qa))

    def compute_loss(self, batch_ans_pred, batch_ans_embed, batch_k_embeds):
        fenzi = 0
        fenmu = 0
        loss = 0
        for i in range(len(batch_ans_pred)):
            fenzi = 0
            fenmu = 0
            fenzi = self.get_dist(batch_ans_pred[i], batch_ans_embed[i])
            fenmu += self.get_dist(batch_ans_pred[i], batch_ans_embed[i])

            for j in range(batch_k_embeds.size()[1]):
                fenmu += self.get_dist(batch_ans_pred[i], batch_k_embeds[i][j])
            loss += -torch.log(fenzi / fenmu)
        return loss / len(batch_ans_pred)

    def get_k_shot(self, label_batch, entities):
        sample_total = []
        for label in label_batch:
            entities_ans = entities.copy()
            entities_ans.remove(label["answer"])
            k_shot_sel = random.choices(entities_ans, k=self.k_shot)
            sample_total.append(k_shot_sel)
        return sample_total

    def get_data(self, batch_idx, total_samples, train_data, train_label):
        batch_sample_ids = [_ for _ in range(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size)]
        batch_sample_ids_sel = random.sample(batch_sample_ids, self.batch_size // 2)
        total_sample_ids = random.sample(total_samples, 2 * self.batch_size)
        total_sample_ids_sel = []
        for sample_id in total_sample_ids:
            if sample_id not in batch_sample_ids:
                total_sample_ids_sel.append(sample_id)
        data_batch = []
        label_batch = []
        for batch_id, total_id in zip(batch_sample_ids_sel, total_sample_ids_sel):
            data_batch.append(train_data[batch_id])
            data_batch.append(train_data[total_id])
            label_batch.append(train_label[batch_id])
            label_batch.append(train_label[total_id])
        return data_batch, label_batch

    def train(self, dataset_train_json, entity_des_dict, rel_des_dict, entities, dataset_val_json=None,
              dataset_test_json=None):
        train_data, train_label = self.divide_sample(dataset_train_json)
        pin_model = concat_model_linear(512)
        pin_model = pin_model.to(device)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.pretrained_clip).to(device)
        text_encoder = CLIPTextModel.from_pretrained(self.pretrained_clip).to(device)

        processor = AutoProcessor.from_pretrained(self.pretrained_clip)
        tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_clip)
        id2text = {}
        id2des = {}
        with open("../../dataset/MarKG/entity2text.txt", "r") as f:
            for line in f:
                id_name = line.split("\t")
                id2text[id_name[0]] = id_name[1]
            f.close()
        with open("../../dataset/MarKG/entity2textlong.txt", "r") as f:
            for line in f:
                id_name = line.split("\t")
                id2des[id_name[0]] = id_name[1]
        if dataset_val_json != None:
            val_data, val_label = self.divide_sample(dataset_val_json)
            test_data, test_label = self.divide_sample(dataset_test_json)
        total_samples = [_ for _ in range(len(train_data))]

        optimizer_image = torch.optim.Adam(image_encoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_text = torch.optim.Adam(text_encoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_pin = torch.optim.Adam(pin_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_avg = 0
        for epoch in range(self.num_epochs):
            for batch_idx in range(math.ceil(len(train_data) / self.batch_size)):
                data_batch, label_batch = self.get_data(batch_idx, total_samples, train_data, train_label)

                timea = datetime.datetime.now()
                batch_image_example_heads, batch_image_example_tails, \
                    batch_image_ques, batch_image_answers = self.get_image("../../dataset/MARS/images/", data_batch,
                                                                           label_batch)
                batch_text_example_heads, batch_text_example_talis, \
                    batch_text_ques, batch_text_answers = self.get_text("../../dataset/MarKG/entity2text.txt",
                                                                        data_batch, label_batch, id2text, id2des)

                batch_image_emb_head = self.image_to_emb(batch_image_example_heads, image_encoder, processor)
                batch_image_emb_tail = self.image_to_emb(batch_image_example_tails, image_encoder, processor)
                batch_image_emb_ques = self.image_to_emb(batch_image_ques, image_encoder, processor)
                batch_image_emb_ans = self.image_to_emb(batch_image_answers, image_encoder, processor)
                batch_text_emb_head = self.text_to_emb(batch_text_example_heads, text_encoder, tokenizer)
                batch_text_emb_tail = self.text_to_emb(batch_text_example_talis, text_encoder, tokenizer)
                batch_text_emb_ques = self.text_to_emb(batch_text_ques, text_encoder, tokenizer)
                batch_text_emb_ans = self.text_to_emb(batch_text_answers, text_encoder, tokenizer)
                batch_emb_head = pin_model(batch_image_emb_head, batch_text_emb_head)
                batch_emb_tail = pin_model(batch_image_emb_tail, batch_text_emb_tail)
                batch_emb_ques = pin_model(batch_image_emb_ques, batch_text_emb_ques)
                batch_emb_ans = pin_model(batch_image_emb_ans, batch_text_emb_ans)
                batch_ans_pred = batch_emb_ques + batch_emb_tail - batch_emb_head
                k_samples = self.get_k_shot(label_batch, entities)
                batch_k_embeds = self.get_entities_emb(k_samples, "../../dataset/MARS/images/", id2text, id2des,
                                                       image_encoder, text_encoder, processor, tokenizer, pin_model)

                loss = self.compute_loss(batch_ans_pred, batch_emb_ans, batch_k_embeds)
                loss.backward()
                # print(str(epoch)+" "+str(batch_idx)+"/"+str(math.ceil(len(train_data)/self.batch_size))+"  loss:"+str(loss.item()))
                optimizer_pin.step()
                optimizer_image.step()
                optimizer_text.step()
                optimizer_text.zero_grad()
                optimizer_image.zero_grad()
                optimizer_pin.zero_grad()
                timeb = datetime.datetime.now()
                loss_avg += loss.item()
                if batch_idx % 100 == 0 and batch_idx != 0:
                    print(str(epoch) + "  " + str(batch_idx) + "/" + str(
                        math.ceil(len(train_data) / self.batch_size)) + "   " + str(loss_avg / 100))
                    loss_avg = 0
                if batch_idx % 1500 == 0 and batch_idx != 0:
                    torch.save(image_encoder, self.output_dir + str(epoch) + "_" + str(batch_idx) + "_image_encoder.pt")
                    torch.save(text_encoder, self.output_dir + str(epoch) + "_" + str(batch_idx) + "_text_encoder.pt")
                    torch.save(pin_model, self.output_dir + str(epoch) + "_" + str(batch_idx) + "_pin_model.pt")
                    print("model_saved")
















