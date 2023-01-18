SMALL_BLIND_SIZE = 50
BIG_BLIND_SIZE = 2 * SMALL_BLIND_SIZE
STACK_SIZE = 20000

ACTION_CC = "check/calls"
ACTION_RAISE = "raises"
ACTION_FOLD = "folds"

PLAYER_SB_STRING = "Small Blind"
PLAYER_BB_STRING = "Big Blind"


# For input "Qs" returns "Queen of Spades"
def card_to_sentance(card_str):
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


def calc_raise_size(parsed_action):
    total_last_bet_to = parsed_action['total_last_bet_to']

    # Remaining
    remaining = STACK_SIZE - total_last_bet_to

    raise_size = total_last_bet_to
    raise_size = min(raise_size, remaining)

    return raise_size