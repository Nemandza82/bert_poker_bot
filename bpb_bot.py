import time
import torch
from poker_bert_models import BertPokerValueModel
from bpb_common import ACTION_CC, ACTION_RAISE, STACK_SIZE, BIG_BLIND_SIZE


class BpBBot():
    def __init__(self, model_path, device_string):
        self.model = BertPokerValueModel()
        self.model.load_from_checkpoint(model_path)

        self.device = torch.device(device_string)

    """
    Parsed action {'street': 3, 'hero': 'Big Blind', 'sentance': 'Hero is Big Blind. Big Blind gets Ace of 
    Spades and Seven of Clubs. Big Blind check/calls. Small Blind raises. Big Blind check/calls. Flop is 
    Eight of Spades, Four of Diamonds and Two of Diamonds. Small Blind raises. Big Blind check/calls.
     Turn is King of Clubs. Small Blind raises. Big Blind check/calls. River is Six of Spades. Small Blind raises. ',
      'street_last_bet_to': 1800, 'total_last_bet_to': 3600, 'last_bet_size': 1800, 'last_bettor': 'Small Blind'}
    """
    def play_hand(self, parsed_action):

        sentance_cc = parsed_action['sentance'] + f"{parsed_action['hero']} {ACTION_CC}."
        sentance_raise = parsed_action['sentance'] + f"{parsed_action['hero']} {ACTION_RAISE}."

        print(f"sentance_cc '{sentance_cc}'")
        print(f"sentance_raise '{sentance_raise}'")

        start = time.time()
        
        mult_cc = self.model.run_inference(sentance_cc, self.device)
        mult_raise = self.model.run_inference(sentance_raise, self.device)

        duration = time.time() - start
        
        print(f"BPB Inference in {duration:.2f}. mult_cc: {mult_cc:.2f}. mult_raise: {mult_raise:.2f}")

        # If bot cc and raise are less than 0 than fold
        if mult_raise < 0 and mult_cc < 0:
            return "f"


        # ----------------------- Evaluate raising first -------------------------
        print("Evaluate raising first")
        total_last_bet_to = parsed_action['total_last_bet_to']

        # Remaining
        remaining = STACK_SIZE - total_last_bet_to

        print(f"Remaining is {remaining}")

        # Now consider full raise
        raise_size = total_last_bet_to
        raise_size = min(raise_size, remaining)

        raise_odds_limiter = raise_size / (total_last_bet_to + raise_size)

        print(f"Evaluate full raise size {raise_size}.")
        print(f"mult_raise {mult_raise} vs raise_odds_limiter {raise_odds_limiter}.")

        # Calculate raise odds
        if mult_raise > raise_odds_limiter:
            return f"r{raise_size}"

        # Try to find smaller raise size
        if mult_raise > 0:
            print(f"Try to find smaller raise size")
            min_bet_size = max(BIG_BLIND_SIZE, parsed_action['last_bet_size'])

            # Can always go all-in
            if min_bet_size > remaining:
                min_bet_size = remaining

            calc_bet_size = round((total_last_bet_to * mult_raise) / (1 - mult_raise))

            print(f"Calc bet size is {calc_bet_size}")

            if calc_bet_size >= min_bet_size:
                return f"r{calc_bet_size}"

            print(f"Smaller than min_bet_size {min_bet_size}")


        # ----------------------- Evaluate check call now -------------------------
        print("Evaluate check/call now")

        if parsed_action['last_bettor'] == "":
            print("If not forced to put any money in just check")
            return "k"

        if mult_cc < 0:
            print("We need to put money in but we have negative mult_cc {mult_cc}: fold")
            return "f"

        call_size = parsed_action["last_bet_size"]
        print(f"We need to call {call_size}")

        call_odds_limiter = call_size / total_last_bet_to

        print(f"mult_cc {mult_cc} vs call_odds_limiter {call_odds_limiter}.")

        # Calculate odds
        if mult_cc > call_odds_limiter:
            print("Positive odds for calling")
            return "c"
        
        print("Only we can do is fold")
        return "f"



    
