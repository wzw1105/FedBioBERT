import random

import torch
from datasets import Dataset


class RobertaDataset(Dataset):
    def __init__(self, corpus_path,  tokenizer, seq_len=512, encoding="utf-8"):
        self.seq_len = seq_len

        self.corpus_path = corpus_path
        self.encoding = encoding
        self.tokenizer = tokenizer

        with open(corpus_path, "r", encoding=encoding) as f:
            self.lines = [line for line in f]
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        bert_input, bert_label, segment_label = self.random_word(self.lines[item])
        output = {
            "bert_input": bert_input,
            "bert_label": bert_label,
            "segment_label": segment_label,
        }
        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):

        tokens = self.tokenizer(sentence, max_length=self.seq_len, padding='max_length', truncation=True)
        input = []
        output_label = []

        for i, token in enumerate(tokens['input_ids']):
            prob = random.random()
            if prob < 0.15 and token > 4:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    input.append(self.tokenizer.mask_token_id)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    input.append(random.randrange(self.tokenizer.vocab_size - 5) + 5)

                # 10% randomly change token to current token
                else:
                    input.append(token)
                output_label.append(token)

            else:
                input.append(token)
                output_label.append(token)

        return input, output_label, tokens['attention_mask']