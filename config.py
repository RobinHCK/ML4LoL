#############################
# RANKED DATASET PARAMETERS #
#############################

# if you are interested in ranked games then True, if not False
ranked = True

# Please make sure to install Riot Watcher and pandas, (`pip install riotwatcher pandas`)
# Since we are limited with 100 requests to the API every 2 minutes, it will take a lot of time to generate everything depending on the amount of data you want. You may need to use several keys to gain some time.

# Please provide here your developer API key from https://developer.riotgames.com/
API_KEY = "Your API key from https://developer.riotgames.com/"

# Parameters to fulfill before running the `1_generate_datasets.py` file

# Set the regions you want to retrive games from (list of available regions here: https://developer.riotgames.com/docs/lol#routing-values_platform-routing-values)
regions = ["EUW1", "KR", "NA1"]
# Choose one queue type (`RANKED_SOLO_5x5` or `RANKED_FLEX_SR`)
queueType = "RANKED_SOLO_5x5"
# Choose the rank leagues of players you want to get games from (`challenger`, `grandmaster` or `master`). Note that it will get the players of the selected leagues at the time you launch the program.
rankLeagues = ["challenger", "grandmaster"]
# Choose the maximum number of games that you want to retrieve from every player
numberOfGamesPerPlayer = 200
# Choose the starting time from where you want to retrieve games (format [Year-Month-Day Hour:minute:second])
startTime = "2023-10-11 03:00:00"
# Choose the ending time until where you want to retrieve games (format [Year-Month-Day Hour:minute:second])
endTime = "2023-10-25 03:00:00"
# If you want to select a specific patch like us, put it to check when retrieving the games that they are from the patch specified. Note that if you want games of a specific patch, you have to set the specific `startTime` and `endTime` of this patch (please check the schedule on LoL website).
patch = "13.20"

##########################
# PRO DATASET PARAMETERS #
##########################

# if you are interested in pro games then True, 
# if not False
# (but ensure to download the pro dataset at oracle's elixir and to fill pro_dataset_name variable below)
pro = True

# You can specify the pro dataset you will be working with (from Oracle's Elixir), this file is obtainable through this link: 
# https://drive.google.com/drive/u/1/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH
# you can put None instead if you wish to ignore this, whatever, an exception handles the case the file existence in the dataset folder
pro_dataset_name = "2023_LoL_esports_match_data_from_OraclesElixir.csv"
# If you want to select a specific patch to retrieve games from the pro dataset, please specify it as a string. If None, then the patch with the highest amount of games will be selected
pro_patch = "13.01"

# You are good to go! Please now execute the `1_download_data_from_RIOT_API.py` file! The games will be generated at the root in a `data` folder.