import torch
from transformers import BertTokenizer, BertModel


class BertPokerValueModel(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super(BertPokerValueModel, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, 1)
        self.tanh = torch.nn.Tanh()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.tanh(linear_output)

        return final_layer

    def load_from_checkpoint(self, model_path):
        print(f"Loading model from {model_path}")
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def get_tokenizer(self):
        return self.tokenizer
