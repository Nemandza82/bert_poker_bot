import time
import torch
import os
import random
from loguru import logger
from poker_bert_models import BertPokerValueModel
from bpb_common import ACTION_CC, ACTION_RAISE, STACK_SIZE, BIG_BLIND_SIZE, calc_raise_size
from bpb_common import parse_action

RAISE_LIMIT_REDUCIONT = 2
CALL_LIMIT_REDUCIONT = 4


class BpBBot():
    def __init__(self, model_path, device_string):
        self.model = BertPokerValueModel()

        if os.path.isfile(model_path):
            self.model.load_from_checkpoint(model_path)
        else:
            logger.warning(f"Did not laod model {model_path}. File does not exist.")

        self.device = torch.device(device_string)
        self.model = self.model.to(self.device)

    
    def simple_logic_max(self, mult_raise, mult_cc):

        if mult_raise <= 0 and mult_cc <= 0:
            logger.info("Both mult_raise and mult_cc are < 0. Fold.")
            return "f"

        if mult_raise >= mult_cc:
            logger.info("mult_raise >= mult_cc -> Raise.")
            return "b"
        else:
            logger.info("mult_raise < mult_cc -> Call.")
            return "c"


    # Against random bot +900 
    # Against calling bot +170
    # Against Slumbot -167
    def simple_logic_raise_first(self, mult_raise, mult_cc):

        if mult_raise > 0:
            logger.info("mult_raise > 0 -> Raise.")
            return "b"

        if mult_cc > 0:
            logger.info("mult_cc > 0 -> Call.")
            return "c"

        logger.info("Both mult_raise and mult_cc are < 0. Fold.")
        return "f"

    # Against random bot +723 (10,000 hands)
    # Against random bot +26 (8,200 hands)
    # Against Slumbot -200 (800)
    def simple_logic_random(self, mult_raise, mult_cc):

        if mult_raise <= 0 and mult_cc <= 0:
            logger.info("Both mult_raise and mult_cc are < 0. Fold.")
            return "f"

        if mult_raise > 0 and mult_cc <= 0:
            logger.info("mult_raise >= 0 and mult_cc < 0 -> Raise.")
            return "b"

        if mult_raise <= 0 and mult_cc > 0:
            logger.info("mult_raise < 0 and mult_cc > 0 -> Call.")
            return "c"

        # Both mult_raise and mult_cc > 0
        logger.info("Both mult_raise and mult_cc > 0 -> Randomizing")

        if random.uniform(0, mult_raise + mult_cc) <= mult_raise:
            logger.info("Randomized raise")
            return "b"

        else:
            logger.info("Randomized call")
            return "c"


    def ev_logic(self, parsed_action, mult_raise, mult_cc):
         # ----------------------- Evaluate raising first -------------------------
        logger.info("Evaluate raising first")

        # Now consider full raise
        raise_size, remaining = calc_raise_size(parsed_action)
        raise_odds_limiter = raise_size / (parsed_action['total_last_bet_to'] + raise_size)

        # HACk ----------
        raise_odds_limiter /= RAISE_LIMIT_REDUCIONT
        logger.info(f"HACk: reducing raise_odds_limiter by {RAISE_LIMIT_REDUCIONT} {raise_odds_limiter}")

        logger.info(f"Evaluate full raise size {raise_size}.")
        logger.info(f"mult_raise {mult_raise:.2f} vs raise_odds_limiter {raise_odds_limiter}.")

        # Calculate raise odds
        #if mult_raise >= raise_odds_limiter:
        if mult_raise >= 0:
            #logger.info(f"mult_raise is > raise_odds_limiter")
            logger.info(f"mult_raise is > 0")

            

            return f"b{parsed_action['street_last_bet_to'] + raise_size}"

        # Try to find smaller raise size
        #if mult_raise > 0:
        #    logger.info(f"Try to find smaller raise size")

        #    if parsed_action['last_bet_size'] > 0:
        #        min_bet_size = parsed_action['last_bet_size']
                
                # Make sure minimum opening bet is the size of the big blind.
        #        if min_bet_size < BIG_BLIND_SIZE:
        #            min_bet_size = BIG_BLIND_SIZE
        #    else:
        #        min_bet_size = BIG_BLIND_SIZE
                
            # Can always go all-in
        #    if min_bet_size > remaining:
        #        min_bet_size = remaining

        #    calc_bet_size = round((parsed_action['total_last_bet_to'] * mult_raise) / (1 - mult_raise))

        #    logger.info(f"Calc bet size is {calc_bet_size}")

        #    if calc_bet_size >= min_bet_size:
        #        if remaining == 0:
        #            logger.info("There is no money to raise: Just call")
        #            return "c"

        #        return f"b{parsed_action['street_last_bet_to'] + calc_bet_size}"

        #    logger.info(f"Smaller than min_bet_size {min_bet_size}")


        # ----------------------- Evaluate check call now -------------------------
        logger.info("Evaluate check/call now")

        if parsed_action['last_bettor'] == "":
            logger.info("If not forced to put any money in just check")
            return "k"

        if mult_cc < 0:
            logger.info(f"We need to put money in but we have negative mult_cc {mult_cc:.2f}: fold")
            return "f"

        call_size = parsed_action["last_bet_size"]
        logger.info(f"We need to call {call_size}")

        call_odds_limiter = call_size / parsed_action['total_last_bet_to']

        # HACk ----------
        call_odds_limiter /= CALL_LIMIT_REDUCIONT
        logger.info(f"HACk: reducing call_odds_limiter by {CALL_LIMIT_REDUCIONT}: {call_odds_limiter}")
        logger.info(f"mult_cc {mult_cc:.2f} vs call_odds_limiter {call_odds_limiter}.")

        # Calculate odds
        #if mult_cc >= call_odds_limiter:
        if mult_cc >= 0:
            logger.info("Positive odds for calling")
            return "c"
        
        logger.info("Only we can do is fold")
        return "f"


    """
    Parsed action {'street': 3, 'hero': 'Big Blind', 'sentance': 'Hero is Big Blind. Big Blind gets Ace of 
    Spades and Seven of Clubs. Big Blind check/calls. Small Blind raises. Big Blind check/calls. Flop is 
    Eight of Spades, Four of Diamonds and Two of Diamonds. Small Blind raises. Big Blind check/calls.
     Turn is King of Clubs. Small Blind raises. Big Blind check/calls. River is Six of Spades. Small Blind raises. ',
      'street_last_bet_to': 1800, 'total_last_bet_to': 3600, 'last_bet_size': 1800, 'last_bettor': 'Small Blind'}
    """
    def next(self, state):

        parsed_action = parse_action(state)

        sentance_cc = parsed_action['sentance'] + f"{parsed_action['hero']} {ACTION_CC}."
        sentance_raise = parsed_action['sentance'] + f"{parsed_action['hero']} {ACTION_RAISE}."

        start = time.time()
        
        mult_cc = self.model.run_inference(sentance_cc, self.device)
        mult_raise = self.model.run_inference(sentance_raise, self.device)

        duration = time.time() - start
        
        logger.info(f"BPB Inference in {duration:.2f}s. mult_cc: {mult_cc:.2f}. mult_raise: {mult_raise:.2f}")

        #act = self.simple_logic_max(mult_raise, mult_cc)
        #act = self.simple_logic_raise_first(mult_raise, mult_cc)
        act = self.simple_logic_random(mult_raise, mult_cc)
        #act = self.ev_logic(parsed_action, mult_raise, mult_cc)

        if act == "f":
            if parsed_action['last_bettor'] == "":
                return "k"
            else:
                return "f"

        elif act in ["c", "k", "cc", "ck"]:
            if parsed_action['last_bettor'] == "":
                return "k"
            else:
                return "c"

        elif act in ["b", "r"]:

            raise_size, remaining = calc_raise_size(parsed_action)

            if raise_size == 0:
                logger.info("There is no money to raise: Just call")
                return "c"

            return f"b{parsed_action['street_last_bet_to'] + raise_size}"

    def hand_finished(self, state):
        x = 1



    
