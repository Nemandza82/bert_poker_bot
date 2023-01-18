import os
import random
import math
from bpb_common import SMALL_BLIND_SIZE, ACTION_CC, ACTION_RAISE, ACTION_FOLD, PLAYER_SB_STRING, PLAYER_BB_STRING
from bpb_common import card_to_sentance



# Parses multiple cards. Returns list as result.
# Example input: Qs2d
# Example output ["Queen of Spades", "Two of Diamonds"]
def parse_cards(cards_str):
    i = 0
    cards = []

    while i < len(cards_str):
        card = card_to_sentance(cards_str[i : i + 2])

        cards.append(card)
        i += 2

    return cards


# Parsing actions in one street. For instance input is "cr1219f".
# Last action must be c or f.
def parse_actions(street_actions_string):
    i = 0
    street_actions = []

    while i < len(street_actions_string):

        # Parse fold action
        if street_actions_string[i] == "f":
            street_actions.append({"type": ACTION_FOLD})
            i += 1
            continue

        # Parse check/call actions
        if street_actions_string[i] == "c":
            street_actions.append({"type": ACTION_CC})
            i += 1
            continue

        # Parse raise action
        if street_actions_string[i] == "r":
            i += 1
            r_start = i

            # Now parsing raise ammount. Advance until we get to next action. 
            # Raise ammount can not be last since after raise must be another action
            while street_actions_string[i] not in ["c", "r", "f"]:
                i += 1

            # Raise ammount is now between r_start and i
            r_amount = street_actions_string[r_start:i]

            # We normalize raise ammount by small blind size
            amount = round(float(r_amount) / SMALL_BLIND_SIZE, 2)
            street_actions.append({"type": ACTION_RAISE, "amount": amount})
            continue

    return street_actions


def get_player(street, action_idx):
    action_idx = action_idx % 2

    if street == 0:
        # In pre-flop in heads-up small_blind plays first
        return PLAYER_SB_STRING if action_idx == 0 else PLAYER_BB_STRING
    else:
        # In post-flop in heads-up big_blind plays first
        return PLAYER_SB_STRING if action_idx == 1 else PLAYER_BB_STRING


# Input is list of cards. Eg: ["Queen of Spades", "Two of Diamonds"]
# Output is human readable sentance: "Queen of Spades and Two of Diamonds"
# If there are more than 2 cards it ads commas.
def print_cards(cards):
    text = ""
    card_i = 0

    while card_i < len(cards):
        text += f"{cards[card_i]}"

        if card_i < len(cards) - 2:
            text += ", "
        elif card_i < len(cards) - 1:
            text += " and "

        card_i += 1

    return text


# STATE:4:r200c/cc/r466c/cr1219f:KdJd|7sAs/JcTs7c/Ah/4d:-466|466:Hero|Villian
# Money is mormalized so that sb is 1
class AcpcLogHand:
    def __init__(self, line):
        self.valid = True
        self.actions = []

        # We first split by : to get sections
        spliited = line.split(":")

        try:
            # First must be string "STATE"
            assert spliited[0] == "STATE"

            # Action strings are separated by "/"
            action_strings = spliited[2].split("/")

            # street_actions_string can be "cr1219f"
            for street_actions_string in action_strings:
                street_actions = parse_actions(street_actions_string)
                self.actions.append(street_actions)

            # print(self.actions)

            card_strings = spliited[3].split("/")

            # card_strings[0] are hand cards. Eg: "KdJd|7sAs"
            hand_strings = card_strings[0].split("|")

            self.big_blind_card_string = hand_strings[0]
            self.small_blind_card_string = hand_strings[1]

            self.big_blind_cards = parse_cards(hand_strings[0])
            self.small_blind_cards = parse_cards(hand_strings[1])

            # print(f"Big blind has: {self.big_blind_cards}")
            # print(f"Small blind has: {self.small_blind_cards}")

            self.common_cards = []
            i = 1

            # Parses commonn cards. They are separated in card_strings
            while i < len(card_strings):
                self.common_cards.append(parse_cards(card_strings[i]))
                i += 1

            # print(f"Common cards {self.common_cards}")

            # results splitted by "|" eg: "-466|466"
            results = spliited[4].split("|")

            # Normalize result by SMALL_BLIND_SIZE
            self.big_blind_res = round(float(results[0]) / SMALL_BLIND_SIZE, 2)
            self.small_blind_res = round(float(results[1]) / SMALL_BLIND_SIZE, 2)

            # print(self.big_blind_res)
            # print(self.small_blind_res)

        except:
            self.valid = False

    """
    Return 4 values: 
        hero_res: Multipluyer for winning eg.: 3.25
        villian_cards: Eg.: TsAh
        text: Eg.: "Hero is Big Blind. Big Blind gets Three of Diamonds and Ace of Hearts. Small Blind raises. Big Blind check/calls. "
    """
    def print_hand(self, max_street, max_action_in_max_street):
        street = 0
        text = ""

        hero = get_player(max_street, max_action_in_max_street)

        if hero == PLAYER_SB_STRING:
            hero_cards = self.small_blind_cards
            hero_res = self.small_blind_res
            villian_cards = self.big_blind_card_string
        else:
            hero_cards = self.big_blind_cards
            hero_res = self.big_blind_res
            villian_cards = self.small_blind_card_string

        text += f"Hero is {hero}. "
        text += f"{hero} gets {print_cards(hero_cards)}. "

        # sb and bb are normalized to 1 and 2
        sb_pot = 1
        bb_pot = 2
        last_action_type = ACTION_FOLD

        while street <= max_street:
            if street > 0:
                if street == 1:
                    text += "Flop is "
                elif street == 2:
                    text += "Turn is "
                elif street == 3:
                    text += "River is "

                text += print_cards(self.common_cards[street - 1])
                text += ". "

            if street < max_street:
                max_action = len(self.actions[street]) - 1
            else:
                max_action = max_action_in_max_street

            action_id = 0

            while action_id <= max_action:
                player = get_player(street, action_id)
                last_action = self.actions[street][action_id]
                last_action_type = last_action["type"]

                # update pots
                if last_action_type == ACTION_CC:
                    if player == PLAYER_SB_STRING:
                        sb_pot = bb_pot
                    else:
                        bb_pot = sb_pot
                elif last_action_type == ACTION_RAISE:
                    # It is raise to
                    if player == PLAYER_SB_STRING:
                        sb_pot = last_action["amount"]
                    else:
                        bb_pot = last_action["amount"]

                # amount = "" if last_action_type != ACTION_RAISE else last_action['amount']

                text += f"{player} {last_action_type}. "
                action_id += 1

            street += 1

        hero_pot = sb_pot if player == PLAYER_SB_STRING else bb_pot

        # We are learning how much hero pot is multiplied
        hero_res = hero_res / hero_pot

        # Get in (-1, 1) range to be better for learning
        # (We are doing that now during training) -> Commented out
        # hero_res = math.tanh(hero_res)

        # Round to two decimal places to take less space
        hero_res = round(hero_res, 2)

        # print(f"{hero_res}, {text}")
        # print(f"Player {player} pot: {hero_pot}")

        # We are not learning what happens when someone folds (we know that)
        if last_action_type == ACTION_FOLD:
            return None, None, None

        # print(text)
        # print(f"{hero} earns {hero_res}")

        return hero_res, villian_cards, text

    def print_hands(self):
        street = 0
        results = []

        while street < len(self.actions):
            action_in_street = 0

            while action_in_street < len(self.actions[street]):
                res, villian_cards, printed_hand = self.print_hand(
                    street, action_in_street
                )

                if not res is None:
                    results.append((res, villian_cards, street, printed_hand))

                action_in_street += 1

            street += 1

        return results


def parse_acpc_log_file(filename, train_f, val_f, test_f):
    total_examples = 0

    with open(filename) as f:
        lines = f.readlines()

        for line in lines:
            rnd = random.randint(0, 99)

            if rnd < 90:
                dst_file = train_f
            elif rnd < 95:
                dst_file = val_f
            else:
                dst_file = test_f

            hand = AcpcLogHand(line)

            if not hand.valid:
                continue

            samples = hand.print_hands()

            for sample in samples:
                res, villian_cards, street, printed_hand = sample
                dst_file.write(f"{res};{villian_cards};{street};{printed_hand}\n")
                total_examples += 1

    return total_examples


def convert_logs_to_dataset(root_folder):
    total_examples = 0

    with open("acpc_train.txt", "w") as train_f:
        with open("acpc_val.txt", "w") as val_f:
            with open("acpc_test.txt", "w") as test_f:
                train_f.write("score;villian_cards;street;text\n")
                val_f.write("score;villian_cards;street;text\n")
                test_f.write("score;villian_cards;street;text\n")

                for filename in os.listdir(root_folder):
                    if filename.endswith(".log"):
                        print(f"Parsing {root_folder}{filename} ")

                        examples = parse_acpc_log_file(
                            f"{root_folder}{filename}", train_f, val_f, test_f
                        )
                        total_examples += examples

                        print(f"Total examples {total_examples}")
                        # print(filename)

                        total_examples_limit = 40000000

                        if total_examples > total_examples_limit:
                            print(f"Total examples excedded {total_examples_limit}. Stopping")
                            break


if __name__ == "__main__"
    # parse_acpc_log_file("./data/acpc2017/PokerBot5.PokerCNN.1.0.log")
    convert_logs_to_dataset("./data/acpc2017/")
