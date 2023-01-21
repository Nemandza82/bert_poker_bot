SMALL_BLIND_SIZE = 50
BIG_BLIND_SIZE = 2 * SMALL_BLIND_SIZE
STACK_SIZE = 20000

ACTION_CC = "check/calls"
ACTION_RAISE = "raises"
ACTION_FOLD = "folds"

PLAYER_SB_STRING = "Small Blind"
PLAYER_BB_STRING = "Big Blind"

NUM_STREETS = 4
STREET_PRE_FLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3
STREET_SHOWDOWN = 4

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

    return raise_size, remaining


def pos_string(pos):
    if pos == 0:
        return PLAYER_BB_STRING

    if pos == 1:
        return PLAYER_SB_STRING

    return ""


def street_to_sentance(street, board):

    if street == 1:
        return f"Flop is {card_to_sentance(board[0])}, {card_to_sentance(board[1])} and {card_to_sentance(board[2])}. "
    
    if street == 2:
        return f"Turn is {card_to_sentance(board[3])}. "

    if street == 3:
        return f"River is {card_to_sentance(board[4])}. "


"""
'action': 'b200', 'hero_pos': 0, 'hole_cards': ['Ac', '9d'], 'board': [], 
"""
def parse_action(state):
    """
    Returns a dict with information about the action passed in.
    Returns a key "error" if there was a problem parsing the action.
    pos is returned as -1 if the hand is over; otherwise the position of the player next to act.
    street_last_bet_to only counts chips bet on this street, total_last_bet_to counts all
      chips put into the pot.
    Handles action with or without a final '/'; e.g., "ck" or "ck/".
    """

    action = state["action"]
    hero_pos = state["client_pos"]
    hole_cards = state["hole_cards"]
    board = state["board"]

    street = 0
    street_last_bet_to = BIG_BLIND_SIZE
    total_last_bet_to = BIG_BLIND_SIZE
    last_bet_size = BIG_BLIND_SIZE - SMALL_BLIND_SIZE
    last_bettor = 0   
    pos = 1
    sentance = f"Hero is {pos_string(hero_pos)}. "
    sentance += f"{pos_string(hero_pos)} gets {card_to_sentance(hole_cards[0])} and {card_to_sentance(hole_cards[1])}. "
    
    check_or_call_ends_street = False
    i = 0
    
    while i < len(action):
        if len(action) == 0:
            break

        if street >= NUM_STREETS:
            return {'error': 'Unexpected error'}
    
        c = action[i]
        i += 1
        
        if c == 'k':
            if last_bet_size > 0:
                return {'error': 'Illegal check'}

            sentance += f"{pos_string(pos)} {ACTION_CC}. " 
            
            if check_or_call_ends_street:
	            # After a check that ends a pre-river street, expect either a '/' or end of string.
                if street < NUM_STREETS - 1 and i < len(action):
                    if action[i] != '/':
                        return {'error': 'Missing slash'}
                    i += 1
    
                if street == NUM_STREETS - 1:
	                # Reached showdown
                    pos = -1
                else:
                    pos = 0
                    street += 1
                    sentance += street_to_sentance(street, board)
    
                street_last_bet_to = 0
                check_or_call_ends_street = False
            else:
                pos = (pos + 1) % 2
                check_or_call_ends_street = True

        elif c == 'c':
            if last_bet_size == 0:
                return {'error': 'Illegal call'}

            sentance += f"{pos_string(pos)} {ACTION_CC}. " 
    
            if total_last_bet_to == STACK_SIZE:
	            # Call of an all-in bet
    	        # Either allow no slashes, or slashes terminating all streets prior to the river.
                if i != len(action):
                    for st1 in range(street, NUM_STREETS - 1):
                        if i == len(action):
                            return {'error': 'Missing slash (end of string)'}
                        else:
                            c = action[i]
                            i += 1
                            if c != '/':
                                return {'error': 'Missing slash'}
                
                if i != len(action):
                    return {'error': 'Extra characters at end of action'}
                
                # Roll out to river
                while street != NUM_STREETS - 1:
                    street += 1
                    sentance += street_to_sentance(street, board)

                pos = -1
                last_bet_size = 0

                # Its all in.. break..
                break
            
            if check_or_call_ends_street:
	            # After a call that ends a pre-river street, expect either a '/' or end of string.
                if street < NUM_STREETS - 1 and i < len(action):
                    if action[i] != '/':
                        return {'error': 'Missing slash'}
                    i += 1
                
                if street == NUM_STREETS - 1:
	                # Reached showdown
                    pos = -1
                else:
                    pos = 0
                    street += 1
                    sentance += street_to_sentance(street, board)
                
                street_last_bet_to = 0
                check_or_call_ends_street = False
            else:
                pos = (pos + 1) % 2
                check_or_call_ends_street = True
            
            last_bet_size = 0
            last_bettor = -1

        elif c == 'f':
            if last_bet_size == 0:
                return {'error', 'Illegal fold'}
            
            if i != len(action):
                return {'error': 'Extra characters at end of action'}

            sentance += f"{pos_string(pos)} {ACTION_FOLD}. " 

            pos = -1
            break

        elif c == 'b':
            j = i
            
            while i < len(action) and action[i] >= '0' and action[i] <= '9':
                i += 1
            
            if i == j:
                return {'error': 'Missing bet size'}
            
            try:
                new_street_last_bet_to = int(action[j:i])
            except (TypeError, ValueError):
                return {'error': 'Bet size not an integer'}
            
            new_last_bet_size = new_street_last_bet_to - street_last_bet_to
            sentance += f"{pos_string(pos)} {ACTION_RAISE}. "
            
            last_bet_size = new_last_bet_size
            street_last_bet_to = new_street_last_bet_to
            total_last_bet_to += last_bet_size
            last_bettor = pos
            pos = (pos + 1) % 2
            check_or_call_ends_street = True
        else:
            return {'error': 'Unexpected character in action'}

    return {
        'street': street,
        'hero': pos_string(pos),
        'sentance': sentance,
        'street_last_bet_to': street_last_bet_to,
        'total_last_bet_to': total_last_bet_to,
        'last_bet_size': last_bet_size,
        'last_bettor': pos_string(last_bettor),
    }
