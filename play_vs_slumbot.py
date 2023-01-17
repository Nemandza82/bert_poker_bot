# The API utilizes HTTP POST requests.  Requests and responses have a JSON body.
# There are two endpoints:
#   /api/new_hand
#   /api/act
# To initiate a new hand, send a request to /api/new_hand.  To take an action, send a
# request to /api/act.
#
# The body of a sample request to /api/new_hand:
#   {"token": "a2f42f44-7ff6-40dd-906b-4c2f03fcee57"}
# The body of a sample request to /api/act:
#   {"token": "a2f42f44-7ff6-40dd-906b-4c2f03fcee57", "incr": "c"}
#
# A sample response from /api/new_hand or /api/act:
#   {'old_action': '', 'action': 'b200', 'client_pos': 0, 'hole_cards': ['Ac', '9d'], 'board': [], 'token': 'a2f42f44-7ff6-40dd-906b-4c2f03fcee57'}
#
# Note that if the bot is first to act, then the response to /api/new_hand will contain the
# bot's initial action.
#
# A token should be passed into every request.  With the exception that on the initial request to
# /api/new_hand, the token may be missing.  But all subsequent requests should contain a token.
# The token can in theory change over the course of a session (usually only if there is a long
# pause) so always check if there is a new token in a response and use it going forward.
#
# Sample action that you might get in a response looks like this:
#   b200c/kk/kk/kb200
# An all-in can contain streets with no action.  For example:
#   b20000c///
#
# Slumbot plays with blinds of 50 and 100 and a stack size of 200 BB (20,000 chips).  The stacks
# reset after each hand.

import requests
import sys
import argparse
from loguru import logger
from datetime import datetime
from bpb_common import SMALL_BLIND_SIZE, STACK_SIZE, ACTION_CC, ACTION_RAISE, ACTION_FOLD, PLAYER_SB_STRING, PLAYER_BB_STRING
from bpb_common import card_to_sentance
from bpb_bot import BpBBot


host = 'slumbot.com'
BOT_CHECKPOINT = "./models/bert_train_002m_val_0641.zip"
#BOT_DEVICE = "cpu"
BOT_DEVICE = "cuda:0"

NUM_STREETS = 4
BIG_BLIND_SIZE = 2 * SMALL_BLIND_SIZE


def pos_string(pos):
    if pos == 0:
        return PLAYER_SB_STRING

    if pos == 1:
        return PLAYER_BB_STRING

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
def ParseAction(action, hero_pos, hole_cards, board):
    """
    Returns a dict with information about the action passed in.
    Returns a key "error" if there was a problem parsing the action.
    pos is returned as -1 if the hand is over; otherwise the position of the player next to act.
    street_last_bet_to only counts chips bet on this street, total_last_bet_to counts all
      chips put into the pot.
    Handles action with or without a final '/'; e.g., "ck" or "ck/".
    """
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
            
            # Validate that the bet is legal
            remaining = STACK_SIZE - street_last_bet_to
            
            if last_bet_size > 0:
                min_bet_size = last_bet_size
	            
                # Make sure minimum opening bet is the size of the big blind.
                if min_bet_size < BIG_BLIND:
                    min_bet_size = BIG_BLIND
            else:
                min_bet_size = BIG_BLIND
            
            # Can always go all-in
            if min_bet_size > remaining:
                min_bet_size = remaining
            
            if new_last_bet_size < min_bet_size:
                return {'error': 'Bet too small'}
            
            max_bet_size = remaining
            
            if new_last_bet_size > max_bet_size:
                return {'error': 'Bet too big'}

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


def NewHand(token):
    data = {}
    
    if token:
        data['token'] = token
    
    # Use verify=false to avoid SSL Error
    # If porting this code to another language, make sure that the Content-Type header is
    # set to application/json.
    response = requests.post(f'https://{host}/api/new_hand', headers={}, json=data)
    success = getattr(response, 'status_code') == 200
    
    if not success:
        logger.error('Status code: %s' % repr(response.status_code))
        
        try:
            logger.error('Error response: %s' % repr(response.json()))
        except ValueError:
            pass
        
        sys.exit(-1)

    try:
        r = response.json()
    except ValueError:
        logger.error('Could not get JSON from response')
        sys.exit(-1)

    if 'error_msg' in r:
        logger.error('Error: %s' % r['error_msg'])
        sys.exit(-1)
        
    return r


def Act(token, action):
    data = {'token': token, 'incr': action}
    
    # Use verify=false to avoid SSL Error
    # If porting this code to another language, make sure that the Content-Type header is
    # set to application/json.
    response = requests.post(f'https://{host}/api/act', headers={}, json=data)
    success = getattr(response, 'status_code') == 200
    
    if not success:
        logger.error('Status code: %s' % repr(response.status_code))
        
        try:
            logger.error('Error response: %s' % repr(response.json()))
        except ValueError:
            pass
        
        sys.exit(-1)

    try:
        r = response.json()
    except ValueError:
        logger.error('Could not get JSON from response')
        sys.exit(-1)

    if 'error_msg' in r:
        logger.error('Error: %s' % r['error_msg'])
        sys.exit(-1)
        
    return r
    
def PlayHand(token, bot):
    r = NewHand(token)
    
    # We may get a new token back from /api/new_hand
    new_token = r.get('token')
    
    if new_token:
        token = new_token
    
    logger.info('Token: %s' % token)
    
    while True:
        logger.info('-----------------')
        logger.info(repr(r))
        
        action = r.get('action')
        client_pos = r.get('client_pos')
        hole_cards = r.get('hole_cards')
        board = r.get('board')
        winnings = r.get('winnings')
        logger.info('Action: %s' % action)
        
        if client_pos:
            logger.info('Client pos: %i' % client_pos)

        logger.info('Client hole cards: %s' % repr(hole_cards))
        logger.info('Board: %s' % repr(board))
        
        if winnings is not None:
            logger.info('Hand winnings: %i' % winnings)
            return (token, winnings)
        
        # Need to check or call
        parsed_action = ParseAction(action, client_pos, hole_cards, board)

        logger.info(f"Parsed action {parsed_action}")

        if 'error' in parsed_action:
            logger.error('Error parsing action %s: %s' % (action, parsed_action['error']))
            sys.exit(-1)
        
        # This sample program implements a naive strategy of "always check or call".
        incr = bot.play_hand(parsed_action)
        
        logger.info('Sending incremental action: %s' % incr)
        r = Act(token, incr)

    # Should never get here

        
def Login(username, password):
    data = {"username": username, "password": password}
    
    # If porting this code to another language, make sure that the Content-Type header is
    # set to application/json.
    response = requests.post(f'https://{host}/api/login', json=data)
    success = getattr(response, 'status_code') == 200
    
    if not success:
        logger.error('Status code: %s' % repr(response.status_code))
        
        try:
            logger.error('Error response: %s' % repr(response.json()))
        except ValueError:
            pass
        
        sys.exit(-1)

    try:
        r = response.json()
    except ValueError:
        logger.error('Could not get JSON from response')
        sys.exit(-1)

    if 'error_msg' in r:
        logger.error('Error: %s' % r['error_msg'])
        sys.exit(-1)
        
    token = r.get('token')
    
    if not token:
        logger.error('Did not get token in response to /api/login')
        sys.exit(-1)
    
    return token


#  python play_vs_slumbot.py --username ngrujic@gmail.com --password dzohusafet
def main():
    parser = argparse.ArgumentParser(description='Slumbot API example')
    parser.add_argument('--username', type=str)
    parser.add_argument('--password', type=str)
    args = parser.parse_args()
    username = args.username
    password = args.password
    
    if username and password:
        token = Login(username, password)
    else:
        token = None

    # To avoid SSLError:
    #   import urllib3
    #   urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    num_hands = 10000
    winnings = 0

    # Create Bot
    bot = BpBBot(BOT_CHECKPOINT, BOT_DEVICE) 
    
    for h in range(num_hands):
        (token, hand_winnings) = PlayHand(token, bot)
        winnings += hand_winnings
    
    logger.info('Total winnings: %i' % winnings)

    
if __name__ == '__main__':
    date = datetime.now().strftime("%m-%d-%Y_%H:%M")
    logger.add(f"./logs/play_log_{date}.txt")

    main()