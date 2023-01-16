from poker_bert_models import BertPokerValueModel


class BpBBot():
    def __init__(self, model_path):
        self.model = BertPokerValueModel()
        self.model.load_from_checkpoint(model_path)


    """
    Hand is dictionary
    { 'actions': 'r200c/cc/cc/cr200', 'client_pos': 0, 'hole_cards': ['Ac', '9d'], 'board': [] }
    """
    def play_hand(hand):
    
