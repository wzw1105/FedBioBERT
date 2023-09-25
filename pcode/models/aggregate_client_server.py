import torch
from transformers import RobertaConfig, RobertaTokenizer

from roberta_client.roberta_client import RobertaClient
from roberta_server.roberta_server import RobertaServer
from roberta_whole.roberta_whole import RobertaForMaskedLM

tokenizer = RobertaTokenizer.from_pretrained('../BertSingle/RobertaSingleModel', max_len=512)

config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,  # we align this to the tokenizer vocab_size
    max_position_embeddings=514,
    hidden_size=512,  # 隐藏单元
    num_attention_heads=4,
    num_hidden_layers=8,  # 表示多少个encoder
    type_vocab_size=1
)
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

embedding_state_key = ['roberta.embeddings.word_embeddings.weight', 'roberta.embeddings.position_embeddings.weight', 'roberta.embeddings.token_type_embeddings.weight', 'roberta.embeddings.LayerNorm.weight', 'roberta.embeddings.LayerNorm.bias']
lm_head_state_key = ['roberta.pooler.dense.weight', 'roberta.pooler.dense.bias', 'lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.decoder.bias']

def aggregate_client_server(client_dict, server_dict):
    num_client_encoder, num_server_client = 0, 0
    for key, value in client_dict.items():
        if key[0:21] == 'roberta.encoder.layer':
            num_client_encoder = max(num_server_client, int(key.split('.')[3]))
    for key, value in server_dict.items():
        if key[0:21] == 'roberta.encoder.layer':
            num_server_client = max(num_server_client, int(key.split('.')[3]))
    config.num_hidden_layers = num_server_client + num_client_encoder + 2
    model = RobertaForMaskedLM(config).to(device)
    result_dict = model.state_dict()
    # set embedding parameters
    for key in embedding_state_key:
        if key not in result_dict:
            print("found error")
            return
        result_dict[key] = client_dict[key]

    # set lm_head parameters
    for key in lm_head_state_key:
        if key not in result_dict:
            print("found error")
            return
        result_dict[key] = server_dict[key]
    # set encoder parameters
    for key, value in client_dict.items():
        if key[0:21] == 'roberta.encoder.layer':
            result_dict[key] = value

    for key, value in server_dict.items():
        if key[0:21] == 'roberta.encoder.layer':
            sp = key.split('.')
            sp[3] = str(int(sp[3]) + num_client_encoder + 1)
            key = '.'.join(sp)
            print(key)
            result_dict[key] = value
    model.load_state_dict(result_dict)
    return model


model_client = RobertaClient.from_pretrained('./tmp_model').to(device)
model_server = RobertaServer.from_pretrained('./tmp_model2').to(device)


client_state = model_client.state_dict()
server_state = model_server.state_dict()
model = aggregate_client_server(client_state, server_state)
model.save_pretrained('./aggregate_model')

print(model.state_dict()['roberta.encoder.layer.1.attention.self.query.bias'])
print()
print(server_state['roberta.encoder.layer.0.attention.self.query.bias'])
'''

for k, v in client_state.items():
    print(k)

print()
for k, v in server_state.items():
    print(k)

print()
for k, v in whole_state.items():
    print(k)
'''
