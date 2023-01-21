import time
import random
from loguru import logger
from treys import Card, Evaluator,  Deck
from bpb_common import SMALL_BLIND_SIZE, BIG_BLIND_SIZE, STACK_SIZE, ACTION_CC, ACTION_RAISE, ACTION_FOLD
from bpb_common import PLAYER_SB_STRING, PLAYER_BB_STRING
from bpb_common import parse_action, calc_raise_size
from bpb_common import STREET_PRE_FLOP, STREET_FLOP, STREET_TURN, STREET_RIVER, STREET_SHOWDOWN

#Card.print_pretty_cards(hand)
#evaluator = Evaluator()
#print(evaluator.evaluate(board, hand))


class PokerGym():

    def __init__(self, bot1, bot2, bot1_name, bot2_name):

        self.bots = [bot1, bot2]
        self.bot_names = [bot1_name, bot2_name]
        self.positions = [PLAYER_BB_STRING, PLAYER_SB_STRING]
        self.winnings = [0, 0]
        self.session_num_hands = 0

    """
    {'action': 'b200', 'client_pos': 0, 'hole_cards': ['9h', '4s'], 'board': []}

    {'action': 'b200f', 'client_pos': 0, 'hole_cards': ['Td', '2c'], 'board': [], 
     'bot_hole_cards': ['8s', '8d'], 'winnings': -100, 'won_pot': -300, 'session_num_hands': 1, 
     'baseline_winnings': -500, 'session_total': -100, 'session_baseline_total': -500}
    """
    def get_state(self, player_id, hole_cards, board, actions):
        res = {}

        res["action"] = actions
        res["client_pos"] = 0 if self.positions[player_id] == PLAYER_BB_STRING else 1
        res["hole_cards"] = [Card.int_to_str(card) for card in hole_cards[player_id]]
        res["board"] = [Card.int_to_str(card) for card in board]

        return res


    def player_id(position_string):
        pid = 0

        # Find bot who plays
        while pid < len(self.positions):
            if self.positions[pid] == position_string:
                break

            pid += 1

        return pid


    def play_action(self, player_id, hole_cards, board, actions):

        state = self.get_state(player_id, hole_cards, board, actions)
        logger.info(f"state {state}")

        incr = bots[player_id].next(state)
        return incr


    def other_player(self, next_to_play):
        return PLAYER_SB_STRING if next_to_play == PLAYER_BB_STRING else PLAYER_BB_STRING
        

    def finish_hand(self, won_id, hole_cards, board, actions, stacks, pot):

        hand_winnings = [0, 0]

        # If someone won, else 0, 0 (Split pot)
        if won_id >= 0:
            lost_id = (won_id + 1) % 2
            hand_winnings = [0, 0]

            hand_winnings[won_id] = stacks[won_id] + pot - STACK_SIZE
            hand_winnings[lost_id] = stacks[lost_id] - pot - STACK_SIZE

        # Inform each bot of new state
        for i in range(len(self.bots)):
            self.winnings[i] += hand_winnings[i]

            state = self.get_state(i, hole_cards, board, actions)

            state["winnings"] = hand_winnings[i]
            state["session_total"] = self.winnings[i]
            state["session_num_hands"] = self.session_num_hands

            self.bots[i].hand_finished(state)

        logger.info(f"Hand finished.")
        logger.info(f"{self.bot_names[0]}: {self.winnings[0]}.")
        logger.info(f"{self.bot_names[1]}: {self.winnings[1]}. ")


    def folds(self, player_id, hole_cards, board, actions, stacks, pot):

        won_id = (player_id + 1) % 2
        self.finish_hand(won_id, hole_cards, board, actions, stacks, pot)


    def play_hand(self):

        # Create and shuffle deck
        deck = Deck()

        stacks = [STACK_SIZE, STACK_SIZE]
        stacks_at_start_of_street = stacks.copy()
        hole_cards = [deck.draw(2), deck.draw(2)]
        street = STREET_PRE_FLOP
        actions = ""
        board = []

        # At beggining of hand take the blinds from players
        stacks[self.player_id(PLAYER_SB_STRING)] -= SMALL_BLIND_SIZE
        stacks[self.player_id(PLAYER_BB_STRING)] -= BIG_BLIND_SIZE

        # Put that money in pot
        pot = SMALL_BLIND_SIZE + BIG_BLIND_SIZE

        # Small blind plays first at pre flop
        next_to_play = PLAYER_SB_STRING

        # Play 4 streets
        while street <= STREET_RIVER:

            # At the beggining of street draw cards
            if street == STREET_FLOP:
                drawn_cards = deck.draw(3)
                board += drawn_cards
                logger.info(f"Flop: {Card.ints_to_pretty_str(drawn_cards)}")
                
            elif street == STREET_TURN:
                drawn_cards += deck.draw(1)
                board += drawn_cards
                logger.info(f"Turn: {Card.ints_to_pretty_str(drawn_cards)}")

            elif street == STREET_RIVER:
                drawn_cards += deck.draw(1)
                board += drawn_cards
                logger.info(f"River: {Card.ints_to_pretty_str(drawn_cards)}")
            
            # If there is no money to bet just go to next street
            # If there is money to bet in stacks then play the street
            while stacks[0] + stacks[1] > 0:

                player_id = self.player_id(next_to_play)
                other_bot_id = (player_id + 1) % 2

                # Ammount to call
                ammount_to_call = stacks[player_id] - stacks[other_bot_id]

                incr = self.play_action(self, player_id, hole_cards, board, actions)
                logger.info(f"{next_to_play} {incr}")

                if incr == "f":
                    if ammount_to_call == 0:
                        logger.warning(f"Illegal fold. Try again.")
                        time.sleep(1)
                        continue

                    actions += "f"
                    self.folds(player_id, hole_cards, board, actions, stacks, pot)

                    # Finish hand
                    return

                elif incr == "c" or incr == "k":

                    # append action
                    actions += incr

                    # Reduce the stack of bot
                    stacks[player_id] -= ammount_to_call

                    # Increese the pot
                    pot += ammount_to_call

                    if ammount_to_call > 0:
                        # Go to next street. In next street BB is playing first
                        break

                    else:
                        # This player checks. Other player is asked.
                        next_to_play = self.other_player(next_to_play)

                else:

                    # Must be raise
                    if not isinstance(incr, str):
                        logger.warning(f"Returned action {incr} is not string. Try again.")
                        time.sleep(1)
                        continue

                    if len(incr) < 2:
                        logger.warning(f"Returned action must be raise. Try again.")
                        time.sleep(1)
                        continue

                    if incr[0] != "r":
                        logger.warning(f"Returned action must be raise. Try again.")
                        time.sleep(1)
                        continue

                    try:
                        raise_ammount = int(incr[1:-1])

                    except:
                        logger.warning(f"Cannot parse raise ammount. Try again.")
                        time.sleep(1)
                        continue

                    if raise_ammount > stacks_at_start_of_street[player_id]:
                        logger.warning(f"Max to raise is {stacks_at_start_of_street[player_id]}. Try again.")
                        time.sleep(1)
                        continue

                    # Finally raise
                    actions += incr
                    stacks[player_id] = stacks_at_start_of_street[player_id] - raise_ammount
                    pot = 2*STACK_SIZE - stacks[player_id] - stacks[(player_id + 1) % 2]

                    # This player raises. Other player is asked.
                    next_to_play = self.other_player(next_to_play)

                    logger.info(f"stacks {stacks}. pot {pot}")

                    # raise_ammount in beagining of each street is "raise by". (eg r100) player adds 100 to pot
                    # in same street re-rease is "raise to". Eg r100r300

            # Incr street
            street += 1
            next_to_play = PLAYER_BB_STRING
            stacks_at_start_of_street = stacks.copy()

            if street <= STREET_RIVER:
                actions += "/"


    def play_hands(self, num_hands):

        # Set initial positions and reset winning
        self.positions = [PLAYER_BB_STRING, PLAYER_SB_STRING]
        self.winnings = [0, 0]
        self.session_num_hands = 0

        while self.session_num_hands < num_hands:
            self.session_num_hands += 1
            self.play_hand()

            # Switch positions between hands
            tmp = self.positions[0]
            self.positions[0] = self.positions[1]
            self.positions[1] = tmp


class RandomBot():
    def __init__(self):
        self.bla = 1

    def next(state):
        parsed_action = parse_action(state)

        raise_size, remaining = calc_raise_size(parsed_action)

        if parsed_action['last_bettor'] == "":
            # Not forced action: can k or b
            if random.randint(0, 100) < 70:
                return f"b{parsed_action['street_last_bet_to'] + raise_size}"
            else:
                return "k"
        else:
            # Forced action: can c, b, or f
            if random.randint(0, 100) < 70:
                if raise_size > 0:
                    return f"b{parsed_action['street_last_bet_to'] + raise_size}"
                else:
                    return "c"
            else:
                return "f"

        return "f"

    def hand_finished(state):
        self.bla = 1


class CallingBot():
    def __init__(self):
        self.bla = 1

    def next(self, state):
        parsed_action = parse_action(state)

        if parsed_action['last_bettor'] == "":
            return "k"
        else:
            return "c"

    def hand_finished(self, state):
        self.bla = 1


class FoldingBot():
    def __init__(self):
        self.bla = 1

    def next(self, state):
        parsed_action = parse_action(state)

        if parsed_action['last_bettor'] == "":
            return "k"
        else:
            return "f"

    def hand_finished(self, state):
        self.bla = 1


class RandomBot():
    def __init__(self):
        self.bla = 1

    def next(self, state):
        parsed_action = parse_action(state)

        raise_size, remaining = calc_raise_size(parsed_action)

        if parsed_action['last_bettor'] == "":
            # Not forced action: can k or b
            if random.randint(0, 100) < 70:
                return f"b{parsed_action['street_last_bet_to'] + raise_size}"
            else:
                return "k"
        else:
            # Forced action: can c, b, or f
            if random.randint(0, 100) < 70:
                if raise_size > 0:
                    return f"b{parsed_action['street_last_bet_to'] + raise_size}"
                else:
                    return "c"
            else:
                return "f"

        return "f"

    def hand_finished(self, state):
        self.bla = 1


if __name__ == "__main__":
    deck = Deck()

    print(deck.draw())