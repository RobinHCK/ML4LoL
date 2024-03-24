import pandas as pd
import importlib

# the array of champions and the dataset are imported
from utils.constants import championsList
from utils.constants import champion_id_map

import json
import os
import csv
import time

from utils.tools import showCurrentTime

from config import pro_patch, pro, ranked, patch

import numpy as np
from collections import Counter

##########################################################################################
# SIMPLE FEATURES (number of games, number of wins, number of losses)
##########################################################################################

def get_specific_champion_features_champion_only(data, champion_name, role=None, game_row_id=None):

    champion_columns = np.array([
        'blueTopChampion', 'blueJungleChampion', 'blueMidChampion', 'blueCarryChampion', 'blueSupportChampion',
        'redTopChampion', 'redJungleChampion', 'redMidChampion', 'redCarryChampion', 'redSupportChampion'
    ])
    
    if role is not None:
        champion_columns = np.array(['blue' + role + 'Champion', 'red' + role + 'Champion'])

    def get_champion_stats(side_prefix, game_row_id=game_row_id):
        champion_cols = [col for col in champion_columns if col.startswith(side_prefix)]
        
        if game_row_id is not None:
            relevant_rows = np.where(
                np.any(np.isin(data[champion_cols].values, champion_name), axis=1) & (data.index != game_row_id)
            )
        else:
            relevant_rows = np.where(np.any(np.isin(data[champion_cols].values, champion_name), axis=1))

        wins_losses = Counter(data.loc[relevant_rows, 'winner'])

        return len(relevant_rows[0]), wins_losses[1], wins_losses[0]

    bluesideGamesAmount, bluesideWinsAmount, bluesideLossesAmount = get_champion_stats('blue')
    redsideGamesAmount, redsideLossesAmount, redsideWinsAmount = get_champion_stats('red')

    globalGamesAmount = bluesideGamesAmount + redsideGamesAmount
    globalWinsAmount = bluesideWinsAmount + redsideWinsAmount

    return (
        bluesideGamesAmount, bluesideWinsAmount, bluesideLossesAmount,
        redsideGamesAmount, redsideWinsAmount, redsideLossesAmount,
        globalGamesAmount, globalWinsAmount, (globalGamesAmount - globalWinsAmount)
    )


####################################
# DIRECT OPPONENTS
####################################

# function to get amount of games, amount of wins and amount of losses for a champion against direct specific champ
def get_specific_champion_features_against_direct_specific_champ_champion_only(data, champion_name, opponent_name, role=None, game_row_id=None):
    champion_name_columns = np.array([
        'blueTopChampion', 'blueJungleChampion', 'blueMidChampion', 'blueCarryChampion', 'blueSupportChampion',
        'redTopChampion', 'redJungleChampion', 'redMidChampion', 'redCarryChampion', 'redSupportChampion'
    ])

    if role is not None:
        champion_name_columns = np.array(['blue'+role+'Champion', 'red'+role+'Champion'])

    # Check if the main champion is present in any of the champ columns
    champ_in_dataset = np.any(np.isin(data[champion_name_columns].values, champion_name), axis=1)
    if not champ_in_dataset.any():
        return tuple([0] * 9)

    # if opponent was not encountered in dataset
    opponent_in_dataset = np.any(np.isin(data[champion_name_columns].values, opponent_name), axis=1)
    if not opponent_in_dataset.any():
        return tuple([0] * 9)

    def get_champ_stats_against_opponent(side_prefix, game_row_id=game_row_id):
        champion_cols = [col for col in champion_name_columns if col.startswith(side_prefix)]

        opponent_prefix = "red" if side_prefix == "blue" else "blue"
        opponent_cols = np.array([col.replace(side_prefix, opponent_prefix) for col in champion_cols])

        if game_row_id is not None:
            relevant_rows = np.where(
            (np.any(np.isin(data[champion_cols].values, champion_name), axis=1)) &
            (np.any(np.isin(data[opponent_cols].values, opponent_name), axis=1)) &
            (data.index != game_row_id) 
        )
        else:
            relevant_rows = np.where(
                (np.any(np.isin(data[champion_cols].values, champion_name), axis=1)) &
                (np.any(np.isin(data[opponent_cols].values, opponent_name), axis=1))
            )

        wins_losses = Counter(data.loc[relevant_rows, 'winner'])

        return len(relevant_rows[0]), wins_losses[1], wins_losses[0]

    bluesideplayerGamesAmountWithChamp, bluesideplayerWinsAmountWithChamp, bluesideplayerLossesAmountWithChamp = get_champ_stats_against_opponent('blue')
    redsideplayerGamesAmountWithChamp, redsideplayerLossesAmountWithChamp, redsideplayerWinsAmountWithChamp = get_champ_stats_against_opponent('red')

    globalplayerGamesAmountWithChamp = bluesideplayerGamesAmountWithChamp + redsideplayerGamesAmountWithChamp
    globalplayerWinsAmountWithChamp = bluesideplayerWinsAmountWithChamp + redsideplayerWinsAmountWithChamp

    return (
        bluesideplayerGamesAmountWithChamp, bluesideplayerWinsAmountWithChamp, bluesideplayerLossesAmountWithChamp,
        redsideplayerGamesAmountWithChamp, redsideplayerWinsAmountWithChamp, redsideplayerLossesAmountWithChamp,
        globalplayerGamesAmountWithChamp, globalplayerWinsAmountWithChamp, (globalplayerGamesAmountWithChamp - globalplayerWinsAmountWithChamp)
    )

####################################
# PLAYER WINRATE WITH CHAMPIONS
####################################

# function to get feature for a player and champion
def get_specific_player_features_on_specific_champion(data, player_id, champion_name, role=None, game_row_id=None):

    champion_name_columns = np.array([
        'blueTopChampion', 'blueJungleChampion', 'blueMidChampion', 'blueCarryChampion', 'blueSupportChampion',
        'redTopChampion', 'redJungleChampion', 'redMidChampion', 'redCarryChampion', 'redSupportChampion'
    ])

    if role!=None:
        champion_name_columns = np.array(['blue' + role + 'Champion', 'red' + role + 'Champion'])

    player_columns = np.array([col.replace('Champion', 'PlayerPuuid') for col in champion_name_columns])

    # Check if the player_id is present in any of the player columns
    player_in_dataset = np.any(np.isin(data[player_columns].values, player_id), axis=1)
    if not player_in_dataset.any():
        return tuple([0] * 9)

    # if champion was not played in dataset
    champion_in_dataset = np.any(np.isin(data[champion_name_columns].values, champion_name), axis=1)
    if not champion_in_dataset.any():
        return tuple([0] * 9)

    def get_player_stats_with_champ(side_prefix, game_row_id = game_row_id):
        champion_cols = [col for col in champion_name_columns if col.startswith(side_prefix)]
        player_cols = [col for col in player_columns if col.startswith(side_prefix)]
        
        if game_row_id is not None:

            player_rows = np.where(np.any(np.isin(data[player_cols].values, player_id), axis=1) &
                          np.any(np.isin(data[champion_cols].values, champion_name), axis=1) &
                          (data.index != game_row_id)
            )
        else:

            player_rows = np.where(np.any(np.isin(data[player_cols].values, player_id), axis=1) &
                          np.any(np.isin(data[champion_cols].values, champion_name), axis=1)
            )

        wins_losses = Counter(data.loc[player_rows, 'winner'])

        # games amount, wins of blue side, wins of red side (so depending on side of player, the variable used to get these returns will change its order)
        return len(player_rows[0]), wins_losses[1], wins_losses[0]

    bluesideplayerGamesAmountWithChamp, bluesideplayerWinsAmountWithChamp, bluesideplayerLossesAmountWithChamp = get_player_stats_with_champ('blue')
    redsideplayerGamesAmountWithChamp, redsideplayerLossesAmountWithChamp, redsideplayerWinsAmountWithChamp = get_player_stats_with_champ('red')

    globalplayerGamesAmountWithChamp = bluesideplayerGamesAmountWithChamp + redsideplayerGamesAmountWithChamp
    globalplayerWinsAmountWithChamp = bluesideplayerWinsAmountWithChamp + redsideplayerWinsAmountWithChamp

    return (
        bluesideplayerGamesAmountWithChamp, bluesideplayerWinsAmountWithChamp, bluesideplayerLossesAmountWithChamp,
        redsideplayerGamesAmountWithChamp, redsideplayerWinsAmountWithChamp, redsideplayerLossesAmountWithChamp,
        globalplayerGamesAmountWithChamp, globalplayerWinsAmountWithChamp, (globalplayerGamesAmountWithChamp - globalplayerWinsAmountWithChamp)
    )

############################
# CREATION OF THE FEATURES #
############################

from utils.constants import championsList
from utils.constants import champion_id_map

# The format of the combined features that will be used for the models when training and testing
features_list_format = []
features_list_format.extend(championsList)
features_list_format.extend(
    [
        'blueTopChampionWins', 'blueTopChampionLosses', 'blueJungleChampionWins', 'blueJungleChampionLosses', 'blueMidChampionWins', 'blueMidChampionLosses', 'blueCarryChampionWins', 'blueCarryChampionLosses', 'blueSupportChampionWins', 'blueSupportChampionLosses',

        'redTopChampionWins', 'redTopChampionLosses', 'redJungleChampionWins', 'redJungleChampionLosses', 'redMidChampionWins', 'redMidChampionLosses', 'redCarryChampionWins', 'redCarryChampionLosses', 'redSupportChampionWins', 'redSupportChampionLosses',


        'blueTopChampionWinsOfPlayer', 'blueTopChampionLossesOfPlayer', 'blueJungleChampionWinsOfPlayer', 'blueJungleChampionLossesOfPlayer', 'blueMidChampionWinsOfPlayer', 'blueMidChampionLossesOfPlayer', 'blueCarryChampionWinsOfPlayer', 'blueCarryChampionLossesOfPlayer', 'blueSupportChampionWinsOfPlayer', 'blueSupportChampionLossesOfPlayer',

        'redTopChampionWinsOfPlayer', 'redTopChampionLossesOfPlayer', 'redJungleChampionWinsOfPlayer', 'redJungleChampionLossesOfPlayer', 'redMidChampionWinsOfPlayer', 'redMidChampionLossesOfPlayer', 'redCarryChampionWinsOfPlayer', 'redCarryChampionLossesOfPlayer', 'redSupportChampionWinsOfPlayer', 'redSupportChampionLossesOfPlayer',


        'blueTopChampionWinsAgainstRedTop', 'blueTopChampionLossesAgainstRedTop', 'blueJungleChampionWinsAgainstRedJungle', 'blueJungleChampionLossesAgainstRedJungle', 'blueMidChampionWinsAgainstRedMid', 'blueMidChampionLossesAgainstRedMid', 'blueCarryChampionWinsAgainstRedCarry', 'blueCarryChampionLossesAgainstRedCarry', 'blueSupportChampionWinsAgainstRedSupport', 'blueSupportChampionLossesAgainstRedSupport',

        'winner'
    ])

def get_champion_role(column):
    champion_keyword_index = column.find('Champion')
    if column.startswith('red'):
        return column[3:champion_keyword_index]
    else:
        return column[4:champion_keyword_index]

def process_champion_features(train_fold, champion, role, color, game_row_id=None):
    champ_features = get_specific_champion_features_champion_only(train_fold, champion, role, game_row_id)
    nb_wins = champ_features[7]
    nb_losses = champ_features[8]
    return nb_wins, nb_losses

def process_player_features(train_fold, player_id, champion, role, color, game_row_id=None):
    player_features = get_specific_player_features_on_specific_champion(train_fold, player_id, champion, role, game_row_id)
    nb_wins = player_features[7]
    nb_losses = player_features[8]
    return nb_wins, nb_losses

# no need for color here, it's blue vs red (so blue wins and blue losses)
def process_champion_features_against_opponent(train_fold, champion, opponent, role, game_row_id=None):
    champ_features = get_specific_champion_features_against_direct_specific_champ_champion_only(train_fold, champion, opponent, role, game_row_id)
    nb_wins = champ_features[7]
    nb_losses = champ_features[8]
    return nb_wins, nb_losses

def create_features_list(dataset_type:None, fold_type='train'):

    champion_col = ['blueTopChampion', 'blueJungleChampion', 'blueMidChampion', 'blueCarryChampion', 'blueSupportChampion',
                        'redTopChampion', 'redJungleChampion', 'redMidChampion', 'redCarryChampion', 'redSupportChampion']

    # number of folds from cross-validation
    n_folds = 5
    for fold_id in range(1, n_folds + 1):

        input_folder = f"data/{dataset_type}/fold{fold_id}" if dataset_type in ["ranked", "pro"] else None

        if not input_folder:
            print(showCurrentTime() + " CAREFUL! The dataset type is not valid !")
            return
        
        input_path = f"{input_folder}/{fold_type}/raw_data.csv"

        data = pd.read_csv(input_path)
        data['winner'] = np.where(data['winner'] == 'Blue Team', 1, 0)

        train_fold = input_folder + '/train/raw_data.csv'
        # Load the train dataset
        train_data = pd.read_csv(train_fold)
        # blue is 1, else (red) it's 0
        train_data['winner'] = np.where(train_data['winner'] == 'Blue Team', 1, 0)

        features_of_all_games = []

        # variable to check if stat was already computed, to prevent loops from being done again
        # these variables will store the stat to prevent the reuse of loop

        # store variables for champ stats
        champ_stat_already_done = {}
        champ_stat_already_done["blue"] = {}
        champ_stat_already_done["blue"]["Top"] = {}
        champ_stat_already_done["blue"]["Jungle"] = {}
        champ_stat_already_done["blue"]["Mid"] = {}
        champ_stat_already_done["blue"]["Carry"] = {}
        champ_stat_already_done["blue"]["Support"] = {}
        champ_stat_already_done["red"] = {}
        champ_stat_already_done["red"]["Top"] = {}
        champ_stat_already_done["red"]["Jungle"] = {}
        champ_stat_already_done["red"]["Mid"] = {}
        champ_stat_already_done["red"]["Carry"] = {}
        champ_stat_already_done["red"]["Support"] = {}

        # player key will be added before champion key
        player_stat_already_done = {}
        player_stat_already_done["blue"] = {}
        player_stat_already_done["blue"]["Top"] = {}
        player_stat_already_done["blue"]["Jungle"] = {}
        player_stat_already_done["blue"]["Mid"] = {}
        player_stat_already_done["blue"]["Carry"] = {}
        player_stat_already_done["blue"]["Support"] = {}
        player_stat_already_done["red"] = {}
        player_stat_already_done["red"]["Top"] = {}
        player_stat_already_done["red"]["Jungle"] = {}
        player_stat_already_done["red"]["Mid"] = {}
        player_stat_already_done["red"]["Carry"] = {}
        player_stat_already_done["red"]["Support"] = {}

        # after role key, it will be champ played, then opponent fought
        confrontation_stat_already_done = {}
        confrontation_stat_already_done["blue"] = {}
        confrontation_stat_already_done["blue"]["Top"] = {}
        confrontation_stat_already_done["blue"]["Jungle"] = {}
        confrontation_stat_already_done["blue"]["Mid"] = {}
        confrontation_stat_already_done["blue"]["Carry"] = {}
        confrontation_stat_already_done["blue"]["Support"] = {}
        confrontation_stat_already_done["red"] = {}
        confrontation_stat_already_done["red"]["Top"] = {}
        confrontation_stat_already_done["red"]["Jungle"] = {}
        confrontation_stat_already_done["red"]["Mid"] = {}
        confrontation_stat_already_done["red"]["Carry"] = {}
        confrontation_stat_already_done["red"]["Support"] = {}

        players_not_in_train = []
        champions_not_in_train = []

        print(f"---\nFor {dataset_type}_split {fold_id} based on {fold_type}")
        start_time = time.time()
        # for each game
        for index, row in data.iterrows():
            start_time_game = time.time()

            features_of_game = []

            draft_champs = [0] * len(championsList)

            # for champion at color and role
            champ_features = []
            player_features = []
            confrontation_features = []
            # champion -> first color, then role

            # in case we are working on the train as input, working again on the train may cause problem, indeed we must ignore the game processed for its stats
            game_row_id = None
            if train_fold == input_path:
                game_row_id = index

            for column in champion_col:
                champion = row[column]
                # if champion is valid (in list)
                if champion in championsList:

                    champ_id = champion_id_map[champion]
                    # if false then red
                    is_blue_side = column.startswith("blue")
                    if(is_blue_side):
                        draft_champs[champ_id] = 1
                        color = "blue"
                    else:
                        draft_champs[champ_id] = -1
                        color = "red"

                    role = get_champion_role(column)

                    ###############
                    # champ stats #
                    ###############
                    if champion not in champ_stat_already_done[color][role]:
                        nb_wins, nb_losses = process_champion_features(train_data, champion, role, color, game_row_id)
                        if game_row_id == None:
                            champ_stat_already_done[color][role][champion] = [nb_wins, nb_losses]
                    else:
                        nb_wins, nb_losses = champ_stat_already_done[color][role][champion][:2]
                        # print("stat already computed")

                    ################
                    # player stats #
                    ################
                    player = row[column.replace('Champion', 'PlayerPuuid')]

                    # if either the player or the champion was played in input dataset but not in train dataset
                    # (checked on a previous match),
                    # then no need to loop
                    if player in players_not_in_train or champion in champions_not_in_train:
                        nb_wins_player, nb_losses_player = 0
                        player_stat_already_done[color][role][player][champion] = [nb_wins_player, nb_losses_player]
                    else:

                        if player not in player_stat_already_done[color][role]:

                            nb_wins_player, nb_losses_player = process_player_features(train_data, player, champion, role, color, game_row_id)

                            if game_row_id == None:
                                player_stat_already_done[color][role][player] = {}

                                if champion not in player_stat_already_done[color][role][player]:
                                    player_stat_already_done[color][role][player][champion] = [nb_wins_player, nb_losses_player]

                        else:
                            # if champ not already played
                            if champion not in player_stat_already_done[color][role][player]:
                                nb_wins_player, nb_losses_player = process_player_features(train_data, player, champion, role, color, game_row_id)

                                if game_row_id == None:

                                    player_stat_already_done[color][role][player][champion] = [nb_wins_player, nb_losses_player]
                            # the player already played, and already played the champion, no need to go loop
                            else :
                                nb_wins_player, nb_losses_player = player_stat_already_done[color][role][player][champion][:2]

                    ########################
                    # confrontations stats #
                    ########################
                    # in our case we decided that if we got blue vs red, then we can deduce red vs blue, so no need to compute for red vs blue
                    if color == 'blue':
                        opponent_color = 'red'
                        opponent = row[column.replace(color, opponent_color)]
                        # if the main champion stats were never computed
                        if champion not in confrontation_stat_already_done[color][role]:
            
                            nb_wins_against_opponent, nb_losses_against_opponent = process_champion_features_against_opponent(train_data, champion, opponent, role, game_row_id)

                            if game_row_id == None:

                                # the next key will be opponent champ name
                                confrontation_stat_already_done[color][role][champion] = {}

                                if opponent not in confrontation_stat_already_done[color][role][champion]:
                                    confrontation_stat_already_done[color][role][champion][opponent] = []
                                    confrontation_stat_already_done[color][role][champion][opponent].append(nb_wins_against_opponent)
                                    confrontation_stat_already_done[color][role][champion][opponent].append(nb_losses_against_opponent)

                        else:
                            # if opponent not already fought
                            if opponent not in confrontation_stat_already_done[color][role][champion]:
                                nb_wins_against_opponent, nb_losses_against_opponent = process_champion_features_against_opponent(train_data, champion, opponent, role, game_row_id)

                                if game_row_id == None:

                                    confrontation_stat_already_done[color][role][champion][opponent] = []
                                    confrontation_stat_already_done[color][role][champion][opponent].append(nb_wins_against_opponent)
                                    confrontation_stat_already_done[color][role][champion][opponent].append(nb_losses_against_opponent)
                            # the confrontation was already found
                            else :
                                nb_wins_against_opponent, nb_losses_against_opponent = confrontation_stat_already_done[color][role][champion][opponent][:2]

                    champ_features.append(nb_wins)
                    champ_features.append(nb_losses)

                    player_features.append(nb_wins_player)
                    player_features.append(nb_losses_player)

                    if color == 'blue':
                        confrontation_features.append(nb_wins_against_opponent)
                        confrontation_features.append(nb_losses_against_opponent)

            features_of_game.extend(draft_champs)
            # other features
            features_of_game.extend(champ_features)
            features_of_game.extend(player_features)
            features_of_game.extend(confrontation_features)

            winner = row['winner']
            features_of_game.append(winner)

            features_of_all_games.append(features_of_game)

            end_time_game = time.time()
            iteration_time_sec = end_time_game - start_time_game
            iteration_time_min = int(iteration_time_sec // 60)
            iteration_time_sec %= 60
            # print(f"game {index+1} took: {iteration_time_min} minutes and {iteration_time_sec:.2f} seconds to finish")

            # print(showCurrentTime(),f"For {dataset_type} split {fold_id} - game {index+1} based on {fold_type}")

        end_time = time.time()
        iteration_time_seconds = end_time - start_time
        iteration_time_minutes = int(iteration_time_seconds // 60)
        iteration_time_seconds %= 60
        print(f"split took: {iteration_time_minutes} minutes {iteration_time_seconds:.2f} seconds to finish")

        output_file_path = f"{input_folder}/{fold_type}/feature_vectors.csv"
        with open(output_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(features_list_format)
            for i in range(len(features_of_all_games)):
                csv_writer.writerow(features_of_all_games[i])

create_features_list("pro", "test")
create_features_list("pro", "train")
create_features_list("ranked", "test")
create_features_list("ranked", "train")