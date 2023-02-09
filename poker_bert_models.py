import torch
import math
from transformers import BertTokenizer, BertModel


"""
Input is sentance describing poker hand state. Output is tanh of learned multiplayer of 
Hero input money in pot.
"""
class BertPokerValueModel(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super(BertPokerValueModel, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, 1)


    def forward(self, tokenized_input_data, device):
        
        mask = tokenized_input_data["attention_mask"].to(device)
        input_id = tokenized_input_data["input_ids"].squeeze(1).to(device)

        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )

        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        
        # Add tanh for classification
        #final_layer = torch.tanh(linear_output)
        final_layer = linear_output

        return final_layer


    """
    For input sentance Eg.: "Hero is Small Blind. Small Blind gets King of Hearts and Seven of Diamonds. Small Blind check/calls. Big Blind raises. Small Blind check/calls. Flop is Ace of Diamonds, King of Spades and Ten of Hearts. Big Blind check/calls. Small Blind check/calls. Turn is Nine of Diamonds. Big Blind check/calls. Small Blind check/calls. "
    Returns multiplier of Hero's invested money in pot.
    """
    def run_inference(self, sentance, device):
        with torch.no_grad():
            input_data = self.tokenize(sentance)
            
            output = self.forward(input_data, device)
            return output.item()
        

    def load_from_checkpoint(self, model_path):
        print(f"Loading model from {model_path}")
        self.load_state_dict(torch.load(model_path))
        self.eval()


    def tokenize(self, text):

        return self.tokenizer(
            text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
