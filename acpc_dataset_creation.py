import os
import random
import math

small_blind_size = 50
action_cc = "check/calls"
action_raise = "raises"
action_fold = "folds"

player_small_blind = "Small Blind"
player_big_blind = "Big Blind"

def parse_card(card_str):
    assert len(card_str) == 2
    rank = ""
    suite = ""
    
    if card_str[0] == "2":
        rank = "Two"
    elif card_str[0] == "3":
        rank = "Three"
    elif card_str[0] == "4":
        rank = "Four"
    elif card_str[0] == "5":
        rank = "Five"
    elif card_str[0] == "6":
        rank = "Six"
    elif card_str[0] == "7":
        rank = "Seven"
    elif card_str[0] == "8":
        rank = "Eight"
    elif card_str[0] == "9":
        rank = "Nine"
    elif card_str[0] == "T":
        rank = "Ten"
    elif card_str[0] == "J":
        rank = "Jack"
    elif card_str[0] == "Q":
        rank = "Queen"
    elif card_str[0] == "K":
        rank = "King"
    elif card_str[0] == "A":
        rank = "Ace"

    if card_str[1] == "s":
        suite = "Spades"
    elif card_str[1] == "c":
        suite = "Clubs"
    elif card_str[1] == "h":
        suite = "Hearts"
    elif card_str[1] == "d":
        suite = "Diamonds"

    return f"{rank} of {suite}"


def parse_cards(cards_str):
    i = 0
    cards = []

    while i < len(cards_str):
        card = parse_card(cards_str[i:i+2])

        cards.append(card)
        i += 2

    return cards


def parse_actions(street_actions_string):
    i = 0
    street_actions = []

    while i < len(street_actions_string):
        if street_actions_string[i] == "f":
            street_actions.append({ "type": action_fold })
            i += 1
            continue

        if street_actions_string[i] == "c":
            street_actions.append({ "type": action_cc })
            i += 1
            continue

        if street_actions_string[i] == "r":
            i += 1
            r_start = i

            while street_actions_string[i] not in ["c", "r", "f"]:
                i += 1

            r_amount = street_actions_string[r_start:i]

            amount = round(float(r_amount)/small_blind_size, 2)
            street_actions.append({ "type": action_raise, "amount": amount })
            continue
        
    return street_actions


def get_player(street, action):
    action = action % 2

    if street == 0:    
        return player_small_blind if action == 0 else player_big_blind
    else:
        return player_small_blind if action == 1 else player_big_blind


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
class Hand():

    def __init__(self, line):
        self.valid = True
        self.actions = []

        #print(f"Parsing {line}")
        spliited = line.split(":")

        try:
            assert spliited[0] == "STATE"
            action_strings = spliited[2].split("/")

            for street_actions_string in action_strings:
                street_actions = parse_actions(street_actions_string)
                self.actions.append(street_actions)

            #print(self.actions)

            card_strings = spliited[3].split("/")
            hand_strings = card_strings[0].split("|")

            self.big_blind_card_string = hand_strings[0]
            self.small_blind_card_string = hand_strings[1]

            self.big_blind_cards = parse_cards(hand_strings[0])
            self.small_blind_cards = parse_cards(hand_strings[1])

            #print(f"Big blind has: {self.big_blind_cards}")
            #print(f"Small blind has: {self.small_blind_cards}")

            self.common_cards = []
            i = 1

            while i < len(card_strings):
                self.common_cards.append(parse_cards(card_strings[i]))
                i += 1

            #print(f"Common cards {self.common_cards}")

            results = spliited[4].split("|")
            self.big_blind_res = round(float(results[0]) / small_blind_size, 2)
            self.small_blind_res = round(float(results[1]) / small_blind_size)

            #print(self.big_blind_res)
            #print(self.small_blind_res)

        except:
            self.valid = False

    def print_hand(self, max_street, max_action_in_max_street):
        street = 0
        text = ""

        hero = get_player(max_street, max_action_in_max_street)

        if hero == player_small_blind:
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
        last_action_type = action_fold
        
        while street <= max_street:
            if street > 0:
                if street == 1:
                    text += "Flop is "
                elif street == 2:
                    text += "Turn is "
                elif street == 3:
                    text += "River is "

                text += print_cards(self.common_cards[street-1])
                text += ". "

            if street < max_street:
                max_action = len(self.actions[street]) - 1
            else:
                max_action = max_action_in_max_street

            action_id = 0

            while action_id <= max_action:
                player = get_player(street, action_id)
                last_action = self.actions[street][action_id]
                last_action_type = last_action['type']

                # update pots
                if last_action_type == action_cc:
                    if player == player_small_blind:
                        sb_pot = bb_pot
                    else:
                        bb_pot = sb_pot
                elif last_action_type == action_raise:
                    # It is raise to
                    if player == player_small_blind:
                        sb_pot = last_action['amount']
                    else:
                        bb_pot = last_action['amount']

                #amount = "" if last_action_type != action_raise else last_action['amount']

                text += f"{player} {last_action_type}. "
                action_id += 1

            street += 1

        hero_pot = sb_pot if player == player_small_blind else bb_pot

        # We are learning how much hero pot is multiplied
        hero_res = hero_res/hero_pot

        # Get in (-1, 1) range to be better for learning
        # (We are doing that now during training) -> Commented out
        # hero_res = math.tanh(hero_res)

        # Round to two decimal places to take less space
        hero_res = round(hero_res, 2)

        #print(f"{hero_res}, {text}")
        #print(f"Player {player} pot: {hero_pot}")

        # We are not learning what happens when someone folds (we know that)
        if last_action_type == action_fold:
            return None, None, None

        #print(text)
        #print(f"{hero} earns {hero_res}")

        return hero_res, villian_cards, text
            

    def print_hands(self):
        street = 0
        results = []

        while street < len(self.actions):
            action_in_street = 0

            while action_in_street < len(self.actions[street]):
                res, villian_cards, printed_hand = self.print_hand(street, action_in_street)

                if not res is None:
                    results.append((res, villian_cards, printed_hand))

                action_in_street += 1
            
            street += 1

        return results


def parse_acpc_file(filename, train_f, val_f, test_f):
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

            hand = Hand(line)

            if not hand.valid:
                continue

            samples = hand.print_hands()

            for sample in samples:
                res, villian_cards, printed_hand = sample
                dst_file.write(f"{res}; {villian_cards}; {printed_hand}\n")
                total_examples += 1

    return total_examples


def convert_logs_to_dataset(root_folder):
    total_examples = 0

    with open('acpc_train.txt', 'w') as train_f:
        with open('acpc_val.txt', 'w') as val_f:
            with open('acpc_test.txt', 'w') as test_f:
                train_f.write("score;villian_cards;text\n")
                val_f.write("score;villian_cards;text\n")
                test_f.write("score;villian_cards;text\n")

                for filename in os.listdir(root_folder):
                    if filename.endswith(".log"):
                        print(f"Parsing {root_folder}{filename} ")
                        
                        examples = parse_acpc_file(f"{root_folder}{filename}", train_f, val_f, test_f)
                        total_examples += examples
                        
                        print(f"Total examples {total_examples}")
                        #print(filename)

                        if total_examples > 40000000:
                            break


#parse_acpc_file("./data/acpc2017/PokerBot5.PokerCNN.1.0.log")
convert_logs_to_dataset('./data/acpc2017/')

