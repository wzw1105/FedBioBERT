import datetime
import logging
import os
from logging import handlers
from pathlib import Path

import torch
import time
from datasets import load_dataset
from torch.utils.data import DataLoader
import tqdm
from transformers import RobertaTokenizer, RobertaConfig
from torch.optim import AdamW

from Roberta.model_roberta import RobertaForMaskedLM
from RobertDataset.RobertaDataset import RobertaDataset


# device = torch.device('cpu')
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = RobertaTokenizer.from_pretrained('../TokenizerModel', max_len=512)

tokens = tokenizer("what are you doing?", max_length=512, padding='max_length', truncation=True)
print(tokens['input_ids'])

# define model
config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,  # we align this to the tokenizer vocab_size
    max_position_embeddings=514,
    hidden_size=768,  # 隐藏单元
    num_attention_heads=12,
    num_hidden_layers=12,  # 表示多少个encoder
    type_vocab_size=1
)

model = RobertaForMaskedLM(config).to(device)

#model = RobertaForMaskedLM.from_pretrained("/mnt/7T/wangzw/trained_models/SingleModel/round3/").to(device)
model.train()
optim = AdamW(model.parameters(), lr=0.00004, betas=(0.9,0.999), weight_decay=0.01, eps=1e-6)
#print(stat(model, (1, 16, 512)))

epochs = 40
batch_size = 16
save_steps = 2
train_length = 10

checkpoint_save_dir = '/mnt/7T/wangzw/trained_models/SingleModel'

if not os.path.exists(f'{checkpoint_save_dir}/loss'):
    os.system(f'mkdir {checkpoint_save_dir}/loss')
os.system(f'rm -rf {checkpoint_save_dir}/loss/*')

tot_time = 0

for epoch in range(epochs):
    file_name = '../generate-data/pubmed-wiki.txt'
    train_dataset = RobertaDataset(file_name, tokenizer=tokenizer)
    train_data_loader = DataLoader(train_dataset, num_workers=2, batch_size=batch_size, shuffle=True)
    data_iter = tqdm.tqdm(train_data_loader,
                          desc="Epoch-%d" % (epoch),
                          total=train_length,
                          bar_format="{l_bar}{bar}{r_bar}")

    if epoch % save_steps == 0:
        print('Model saved ! ')
        save_dir = f'{checkpoint_save_dir}/round{epoch}'
        if not os.path.exists(save_dir):
            os.system(f'mkdir {save_dir}')
        model.save_pretrained(save_dir)
        #model.save_pretrained('./RobertaSingleModel')

    loss_log = []
    time_log = []
    index = 0

    for data in data_iter:
        print(data['bert_input'].size())
        batch_start = time.time()
        data = {key: value.to(device) for key, value in data.items()}
        #print(data['bert_input'])
        # print(data['bert_input'].max(), data['bert_input'].min(), data['bert_label'].max(), data['bert_label'].min())
        outputs = model(data['bert_input'], attention_mask=data['segment_label'], labels=data['bert_label'])

        optim.zero_grad()
        loss=outputs[0]
        loss.backward()
        optim.step()

        tot_time += (time.time() - batch_start)
        loss_log.append(loss.item())
        time_log.append(tot_time)

        #data_iter.set_description(f'Epoch {epoch}')
        data_iter.set_postfix(loss=loss.item())

        index += 1
        if index % train_length == 0:
            break

    with open(f"{checkpoint_save_dir}/loss/loss-log.txt", 'a') as w:
        for time_, data in zip(time_log, loss_log):
            w.write("%.4f %.8f\n" % (time_, data))
        w.close()