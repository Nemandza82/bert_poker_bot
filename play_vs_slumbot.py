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

# Random bot results
# session_total = -2054750 - 665350 - 1081350 - 198700 = 4000150
# number of hands = 10009 + 4524 + 4648 + 1576 = 20757
# -198 bb per 100 hands

# Calling bot results:
# session_total = -310200 -1012050 -19000 = 1341250
# number of hands = 2451 + 6132 + 405 = 8988
# -149 bb per 100 hands

import requests
import sys
import argparse
import random
import time
import traceback
from loguru import logger
from datetime import datetime
from bpb_common import parse_action
from bpb_bot import BpBBot
from poker_gym import RandomBot, CallingBot, FoldingBot


host = 'slumbot.com'
BOT_CHECKPOINT = "./models/bert_train_002m_val_0644.zip"
BOT_DEVICE = "cpu"
#BOT_DEVICE = "cuda:0"


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
        state = {}

        state["action"] = action
        state["client_pos"] = client_pos
        state["hole_cards"] = hole_cards
        state["board"] = board

        parsed_action = parse_action(state)
        logger.info(f"Parsed action {parsed_action}")

        if 'error' in parsed_action:
            logger.error('Error parsing action %s: %s' % (action, parsed_action['error']))
            sys.exit(-1)
        
        # Get action from bot
        incr = bot.next(state)
        
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
    num_hands = 10010
    winnings = 0

    # Create Bot
    #bot = BpBBot(BOT_CHECKPOINT, BOT_DEVICE) 
    #bot = RandomBot()
    bot = CallingBot()
    #bot = FoldingBot()
    
    for h in range(num_hands):
        logger.info(f"Playing hand {h} -------------------------------------------------")
        (token, hand_winnings) = PlayHand(token, bot)
        winnings += hand_winnings
    
    logger.info('Total winnings: %i' % winnings)

    
if __name__ == '__main__':

    while True:
        try:
            date = datetime.now().strftime("%m-%d-%Y_%H:%M")
            logger.add(f"./logs/play_log_{date}.txt")
            main()

        except Exception as e:
            traceback.print_exc()
            logger.warning(f"Exception happened. Sleeping and trying again.")
            time.sleep(60)
