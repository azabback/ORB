import pandas as pd
import numpy as np
import os
import random
import time

from mistralai import Mistral
import google.generativeai as genai
import cohere

white_card_dict = pd.read_pickle('/Users/amara/Documents/CAH_AI/white_cards.pkl')
black_card_dict = pd.read_pickle('/Users/amara/Documents/CAH_AI/black_cards.pkl')

class CardDeck():
    def __init__(self,color):
        self.color = color
        self.available_cards = []
        self.discard = []

    def draw_card(self):
        drawn_card = self.available_cards.pop(0)
        self.available_cards = self.available_cards[1:]
        random.shuffle(self.available_cards)
        return drawn_card
    
    def start_game(self):
        if self.color == 'white':
            card_dict = white_card_dict
        else:
            card_dict = black_card_dict
        all_cards = list(card_dict.keys())
        random.shuffle(all_cards)
        self.available_cards = all_cards

    def shuffle_in_discards(self):
        discards = self.discard
        random.shuffle(discards)
        self.available_cards.append(discards)
        self.discard = []


class GamePlay():
    def __init__(self,players):
        self.all_players = players
        self.whiteDeck = CardDeck('white')
        self.whiteDeck.start_game()
        self.blackDeck = CardDeck('black')
        self.blackDeck.start_game()
        for p in self.all_players:
            p.game = self
        self.round_count = 0
        score_dict = {p.name:0 for p in players}
        self.scores = score_dict
        self.judge_index = 0
        self.round_judge = self.all_players[self.judge_index]
        self.round_players = [i for i in self.all_players if i != self.round_judge]
        self.round_black_card = ''

    def select_black_card(self):
        card = self.blackDeck.draw_card()
        self.round_black_card = card
    
    def collect_cards(self):
        # self.select_black_card()
        prompt_ = self.round_black_card
        prompt_card = get_card_text(prompt_,'black')
        option_dict = {}
        option_num = 1

        temp_players = [i for i in self.round_players]
        random.shuffle(temp_players)

        for p in temp_players:
            in_dict = {'player_obj':p,
                       'player':p.name}
            selection = p.select_cards(prompt_card)
            in_dict['card'] = selection.split(', ')

            option_dict[f'OPTION {option_num}'] = in_dict
            option_num += 1
            print(f'Player {p.name} has submitted their choice!')
            time.sleep(random.choice(list(range(0,3))))

        prompt_l = prompt_card.split(' ')
        num_blanks = prompt_l.count('[BLANK]')

        for k,v in option_dict.items():
            if num_blanks == 0:
                full_submission = f'{prompt_card} -- {str.upper(get_card_text(v["card"][0],"white"))}'
            else:
                for c in v['card']:
                    resp_l = [i for i in prompt_l]
                    blank_ind = resp_l.index('[BLANK]')
                    resp_l[blank_ind] = str.upper(get_card_text(c,'white'))
                full_submission = ' '.join(resp_l)

            option_dict[k]['full_submission'] = full_submission

        return option_dict

    def get_judgement(self,option_dict):
        judge = self.round_judge
        winner = judge.judge(get_card_text(self.round_black_card,'black'),option_dict)
        self.scores[winner] += 1

    def next_round(self):
        self.round_count += 1
        self.select_black_card()
        if self.round_count == 1:
            self.whiteDeck.start_game()
            self.blackDeck.start_game()
            for p in self.all_players:
                p.replenish_white_cards(7)
        print(f'ROUND {self.round_count}:\n')
        time.sleep(1)
        print(f'Current scores are: ')
        for player,score in self.scores.items():
            print(f'\t{player}: {score}')
            time.sleep(1)
        time.sleep(1)
        print('\n')
        print(f"This round's judge is {self.round_judge.name}")
        time.sleep(1)
        prompt_text = get_card_text(self.round_black_card,'black')
        time.sleep(1)
        print(f'\nThe prompt is "{prompt_text}"')
        time.sleep(1.5)
        print(f'\nGood luck!\n\n')


    def play(self):
        while(max(list(self.scores.values())) < 5):
            self.next_round()
            submissions = self.collect_cards()
            self.get_judgement(submissions)

            self.judge_index += 1
            if self.judge_index == len(self.all_players):
                self.judge_index = 0
            self.round_judge = self.all_players[self.judge_index]

            self.round_players = [i for i in self.all_players if i != self.round_judge]

            time.sleep(7)

        game_winner = {k:v for k,v in self.scores.items() if v == 5}
        winner = list(game_winner.keys())[0]
        print(f'{winner} wins!')


def get_card_text(card_id,color):
    # print(card_id)
    if color == 'white':
        return white_card_dict[card_id]
    else:
        return black_card_dict[card_id]
    

whiteDeck = CardDeck('white')
blackDeck = CardDeck('black')


class Player:
    def __init__(self,api_key,name,game=None):
        self.api_key = api_key
        self.cards = []
        self.game = game
        self.name = name

    def prompt_ai(self,message):
        pass

    def draw_card(self,deck: CardDeck):
        current_cards = self.cards
        current_cards.append(deck.draw_card())
    
    def replenish_white_cards(self,num_cards):
        while len(self.cards) < num_cards:
            self.draw_card(self.game.whiteDeck)

    def turn_in_card(self,selected_card):
        if ',' in selected_card:
            selected_cards = selected_card.split(', ')
            for selected_card in selected_cards:
                selected_index = list(self.cards).index(selected_card)
                self.cards.pop(selected_index)
                self.replenish_white_cards(7)
        else:
            selected_index = list(self.cards).index(selected_card)
            self.cards.pop(selected_index)
            self.replenish_white_cards(7)

    def display_white_cards(self):
        card_num = 1
        # print(self.cards)
        for c in list(self.cards):
            print(f'Card {card_num}: {c}\n\t{get_card_text(c,"white")}')
            card_num += 1

    def select_cards(self,prompt):
        message = f'Here is the prompt for this round: \n\t"{prompt}".\n'
        prompt_l = prompt.split(' ')
        num_blanks = prompt_l.count('[BLANK]')
        if num_blanks == 0:
            message += f'You need to choose one card that is the best response to the prompt.'
        elif num_blanks == 1:
            message += f'You need to choose one card to replace the "[BLANK]" in the prompt.'
        else:
            message += f'You need to choose two cards to replace each "[BLANK]" in the prompt; the first card you select should stand for the first "[BLANK]".'

        message += f'\nYou should make your selection based on what would be the funniest, silliest, or raunchiest response, or which response would make the most sense.'
        message += '\nHere are the cards you have to choose from: '
        option_count = 1
        for c in list(self.cards):
            message += f'\nOPTION {option_count}:'
            message += f'\n\tcard id: {c}'
            message += f'\n\tcard text: {get_card_text(c,"white")}'
            option_count += 1
        
        message += '\nWhich would you like to select? Please only respond with the id of the card for which you would like to select, based on the card text.'
        # print(message)
        choices = list(self.cards)

        selected_card = self.prompt_ai(message,choices)
        if selected_card[-5:] != '.jpeg':
            selected_card = f'{selected_card}.jpeg'
        # print(get_card_text(selected_card,'white'))
        self.turn_in_card(selected_card)

        return selected_card

    def judge(self,prompt,options):
        """
        options is a dict:
        [option_count]: {"player": player who submitted,
                        "card": card(s) of the submission}
        """
        message = f'You are the judge for this round! Here is the prompt: {prompt}\n'
        print_m = message
        # print(print_m)
        time.sleep(1)
        print_m = ''
        prompt_l = prompt.split(' ')
        num_blanks = prompt_l.count('[BLANK]')
        if num_blanks == 0:
            message += f'The other players of the game have all selected a card that they think is the best response to the prompt.'
            # print_m += f'The other players of the game have all selected a card that they think is the best response to the prompt.'
        elif num_blanks == 1:
            message += f'The other players of the game have all selected a card that they think is the best choice to replace the "[BLANK]" in the prompt.'
            # print_m += f'The other players of the game have all selected a card that they think is the best choice to replace the "[BLANK]" in the prompt.'
        else:
            message += f'The other players of the game have all selected two cards that they think are the best choices to replace each "[BLANK]" in the prompt. The first card they turned in goes for the first "[BLANK]".'
            # print_m += f'The other players of the game have all selected two cards that they think are the best choices to replace each "[BLANK]" in the prompt. The first card they turned in goes for the first "[BLANK]".'
        # print(print_m)
        time.sleep(1)
        print_m = ''
        message += f'\nYour job is to choose the best submission. What makes a submission the best is up to you -- the funniest, what makes the most sense, the raunchiest, etc. \n'
        # print_m += f'\nYour job is to choose the best submission. What makes a submission the best is up to you -- the funniest, what makes the most sense, the raunchiest, etc. \n'
        message += 'The submissions are listed below. They have been combined with the prompt to allow you to understand each submission in context.'
        # print_m += 'The submissions are listed below. They have been combined with the prompt to allow you to understand each submission in context.'

        # print(print_m)
        time.sleep(1)
        print_m = ''
        for k,v in options.items():
            message += f'\n{k}'
            print_m += f'\n{k}'
            message += f'\n\t{v["full_submission"]}'
            print_m += f'\n\t{v["full_submission"]}'
            print(print_m)
            time.sleep(1)
            print_m = ''


        message += "\n\nWhich submission do you choose? Please only reply with 'OPTION' and the number of the option, as listed above."
        print_m += "\n\nWhich submission do you choose? Please only reply with 'OPTION' and the number of the option, as listed above."
        print(print_m)
        time.sleep(3)
        choices = list(options.keys())

        selected_option = self.prompt_ai(message,choices)

        print('\n\n')
        # print(options[selected_option][])
        winning_sub = options[selected_option]['full_submission']
        print(f'The winning submission is: {winning_sub}')

        winning_player = {k:v for k,v in options.items() if k == selected_option}
        winning_player = winning_player[selected_option]['player']

        print(f"{winning_player}'s submission was selected! Well done, {winning_player}!")
        return winning_player
    


class MistralPlayer(Player):
    def prompt_ai(self,message,choices=None):
        model = "mistral-large-latest"

        client = Mistral(api_key=self.api_key)

        chat_response = client.chat.complete(
            model= model,
            messages = [
                {
                    "role": "user",
                    "content": message,
                },
            ]
        )
        return chat_response.choices[0].message.content


class GeminiPlayer(Player):
    def prompt_ai(self,message,choices=None):
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(message)
        # print(response.text)
        selected_card = response.text
        selected_card = selected_card.strip('\n')
        # if selected_card[-5:] != '.jpeg':
        #     selected_card = f'{selected_card}.jpeg'
        return selected_card


class CoherePlayer(Player):
    def prompt_ai(self,message,choices=None):
        co = cohere.ClientV2(self.api_key)
        response = co.chat(
            model="command-r-plus", 
            messages=[{"role": "user", "content": message}]
        )

        selected_card = dict(dict(dict(response)['message'])['content'][0])['text']
        return selected_card


class RandomPlayer(Player):
    def prompt_ai(self,message,choices=None):
        selected_card = random.choice(choices)
        return selected_card
    


def main():
    Mistral1 = MistralPlayer('EnR9bolLjxLoo1eVHoyY1lv2ztSjLWFO',"Mistral1")
    Mistral2 = MistralPlayer('EnR9bolLjxLoo1eVHoyY1lv2ztSjLWFO',"Mistral2")
    Gemini1 = GeminiPlayer('AIzaSyC6AWqSlK1JWRIe79kwR48-WOr9WKplzy8','Gemini1')
    Gemini2 = GeminiPlayer('AIzaSyC6AWqSlK1JWRIe79kwR48-WOr9WKplzy8','Gemini2')
    Gemini3 = GeminiPlayer('AIzaSyC6AWqSlK1JWRIe79kwR48-WOr9WKplzy8','Gemini3')
    # CohereP = CoherePlayer('w1FP9IRl1UevS6LO5A2jRCIMO0KvYiPQNvmm5Z3x','CohereP')
    # Random1 = RandomPlayer(None,'Random1')
    # Random2 = RandomPlayer(None,'Random2')
    # Random3 = RandomPlayer(None,'Random3')

    # game = GamePlay([MistralP,GeminiP,CohereP,Random1,Random2,Random3])
    # game = GamePlay([Mistral1,Mistral2,Gemini1,Gemini2,CohereP])
    game = GamePlay([Mistral1,Mistral2,Gemini1,Gemini2,Gemini3])
    game.play()



if __name__ == "__main__":
    main()