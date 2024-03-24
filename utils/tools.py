import time
import json
import os
import csv
# Please install pandas, (`pip install pandas`)
import pandas as pd
from datetime import datetime
from datetime import timedelta

import sys

from utils.constants import championsList, draft_championsList

# Add the root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import patch from config.py at root
from config import patch

def showCurrentTime():
    """
    Gives the current time.

    :return: a string containing the current time like follow: [HH:MM:SS]
    """
    now = datetime.now()
    currentTime = now.strftime("%H:%M:%S")
    return "[" + currentTime + "] "

def showExecutionTime(startTime: float):
    """
    Gives the total execution time of the program.

    :param float startTime: The starting time of the program
    :return: a string containing the execution time like follow: HH:MM:SS.MS
    """
    executionTime = str(timedelta(seconds = (time.time() - startTime)))
    return executionTime

def convertDatetimeToTimestamp(date: str):
    """
    Converts a date string to a timestamp.

    :param date: A string in the format "YYYY-MM-DD HH:MM:SS"
    :return: A Unix timestamp (integer) representing the input date, or None if there's an error.
    """
    date_format = "%Y-%m-%d %H:%M:%S"  # Format of the date string
    # Check that the date format is respected
    try:
        # Parse the string and convert it to a datetime object
        date = datetime.strptime(date, date_format)
        # Get the current date and time
        current_datetime = datetime.now()
        # Check if the parsed date is in the future
        if date > current_datetime:
            print(showCurrentTime() + "Error. The date entered is in the future.")
            return None
        else:
            # Convert the datetime object to a timestamp
            date = int(date.timestamp())
            return date
    except ValueError:
        print(showCurrentTime() + "Error. The date string does not match the expected format. It must be: [Year-Month-Day Hour:minute:second]")
        return None

def jsonObjectToCSV(data: list, region: str, rankLeague: str = None, dataType: str = None, gameId: str = None):
    """
    Creates a CSV file with data sent in region folder of the data folder.

    :param list data: The array containing data to store in the CSV
    :param str region: The region that data comes from
    :param rankLeague: The rank league data come from (challenger, grandmaster, master)
    :type rankLeague: str or None
    :param dataType: The data type that the CSV holds
    :type dataType: str or None
    :param gameId: The id of the game
    :type gameId: str or None
    """
    # Path where the files will be stored (i.e the folder corresponding to the game's region in the data folder )
    if (gameId == None):
        fileName = "./data/" + region + "/" + region + "_" + rankLeague + "/" + dataType + "/" + region + "_" + dataType
    else:
        fileName = "./data/" + region + "/" + region  + "_" + rankLeague + "/" + dataType + "/"+ gameId + "_" + dataType

    # Creates the directory if it doesn't exist
    os.makedirs(os.path.dirname(fileName + ".json"), exist_ok = True)

    # Converts objects in a json string
    jsonString = json.dumps(data, indent = 4)
    # Store the json string in a file
    with open(fileName + ".json", "w+") as outfile:
        outfile.write(jsonString)

    # Read the json file previously created
    df = pd.read_json(fileName + ".json")
    # Convert the json into a CSV
    df.to_csv(fileName + ".csv", index = None)
    # Delete the json file
    os.remove(fileName + ".json")

def transformRankedDataset(csv_file_path):
    """
    Function to change champion names of the ranked CSV file according to its corresponding array and the array of correct champion names.

    :param str csv_file_path: The path of the ranked CSV file we want to update
    """
    # Create a dictionary to map values from array1 to array2
    value_mapping = dict(zip(draft_championsList, championsList))

    # Read the entire content of the CSV file into memory
    with open(csv_file_path, 'r') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file)

        # Read the header row
        header = next(csv_reader)

        # List to store the modified rows
        modified_rows = [header]

        # Loop through each row
        for row in csv_reader:
            # Loop through each column in the row
            for col_index, col_value in enumerate(row):
                # Check if the column value is in draft_championsList
                if col_value in draft_championsList:
                    # Replace the value with its corresponding value from value_mapping
                    row[col_index] = value_mapping[col_value]

            # Add the modified row to the list
            modified_rows.append(row)

    # Write the modified content back to the original file
    with open(csv_file_path, 'w', newline='') as modified_file:
        # Create a CSV writer object
        csv_writer = csv.writer(modified_file)

        # Write all modified rows to the file
        csv_writer.writerows(modified_rows)

def mergeCSVDraftFiles(regions: list, ranks: list):
    """
    Function to merge CSV drafts files of regions and ranks into one.

    :param list regions: The list of regions we want to merge
    :param list ranks: The list of ranks we want to merge
    """
    try:
        directories = []
        # Set the directory paths where the CSV files are located
        for region in regions:
            for rank in ranks:
                path = "data/"+region+"/"+region+"_"+rank+"/drafts"

                directories.append(path)

        # Initialize an empty list to store the dataframes
        dataframes = []

        # Iterate over all directories
        for directory in directories:
            # Iterate over all files in the directory
            for filename in os.listdir(directory):
                if filename.endswith(".csv"):
                    file_path = os.path.join(directory, filename)
                    # Read the CSV file and append the dataframe to the list
                    data = pd.read_csv(file_path)
                    dataframes.append(data)

        # Concatenate the dataframes in the list
        merged_data = pd.concat(dataframes, ignore_index=True)

        output_path = "dataset/ranked_dataset_" + patch + ".csv"

        # Write the merged data to a new CSV file
        merged_data.to_csv(output_path, index=False)
        transformRankedDataset(output_path)
    except:
        print(f"{showCurrentTime()}An exception occured while trying to merge data of {regions} and {ranks}.")


def transformProDataset(csv_path, pro_patch):

    # Read the CSV file into a pandas DataFrame
    original_dataframe = pd.read_csv(csv_path)

    # Show the 15 first rows of the CSV file
    original_dataframe.head(15)

    # Count occurrences of values in the patch column
    count = original_dataframe['patch'].value_counts()

    # Perform integer division by 12 (because one game is 12 rows) and convert the result to integers
    count_integer = (count // 12).astype(int)

    # Get the three main leagues associated with each patch
    top_leagues = original_dataframe.groupby('patch')['league'].apply(lambda x: x.value_counts().index[:3]).reset_index()

    # Merge the count and top leagues DataFrames
    result = pd.merge(count_integer.rename('Counts'), top_leagues.rename(columns={'league': 'Top Leagues'}), left_index=True, right_on='patch')

    # Sort the merged DataFrame by the "Counts" column
    result = result.sort_values(by='Counts', ascending=False)

    # Get the patch with the highest amount of games
    highest_patch = result.iloc[0]['patch']

    print(f"{showCurrentTime()}The patch with the highest amount of games is: {highest_patch:.2f}")

    # Display the result with the desired order
    result = result[['patch', 'Counts', 'Top Leagues']]

    print(showCurrentTime())
    print(result)

    # If the user provided a specific patch, get games of this patch, otherwise get games of the one with the highest number of games
    if pro_patch != None:
        selected_patch = float(pro_patch)
        print(f"{showCurrentTime()}The patch you decided to use (the one specified in the config file) is: {selected_patch:.2f}")
    else:
        selected_patch = highest_patch
        print(f"{showCurrentTime()}The patch we will use is: {selected_patch:.2f}")

    # Update the dataframe to only use games from the patch highest patch variable (of any region and division)
    updated_dataframe = original_dataframe.loc[original_dataframe['patch'] == selected_patch]

    # Format of the new dataset
    dataset_columns_format = [
        'gameid', 'blueTopChampion', 'blueJungleChampion', 'blueMidChampion', 'blueCarryChampion', 'blueSupportChampion',
        'redTopChampion', 'redJungleChampion', 'redMidChampion', 'redCarryChampion', 'redSupportChampion',
        'blueTopPlayerPuuid', 'blueJunglePlayerPuuid', 'blueMidPlayerPuuid', 'blueCarryPlayerPuuid', 'blueSupportPlayerPuuid',
        'redTopPlayerPuuid', 'redJunglePlayerPuuid', 'redMidPlayerPuuid', 'redCarryPlayerPuuid', 'redSupportPlayerPuuid',
        'winner'
    ]

    # Create an empty DataFrame to store the new dataset
    new_dataframe = pd.DataFrame(columns=dataset_columns_format)

    # Group the original DataFrame by 'gameid' and iterate through each group
    for gameid, group in updated_dataframe.groupby('gameid'):
        # Exclude the last two rows that contain global stats
        game_data = group.iloc[:-2]

        blue_team = game_data[game_data['side'] == 'Blue']
        red_team = game_data[game_data['side'] == 'Red']
        # If 1, Blue Team wins, otherwise it is Red
        if (blue_team.iloc[0]['result'] == 1):
            winner = "Blue Team"
        else:
            winner = "Red Team"

        # Check if any 'playerid' or 'champion' value is empty, and if so, skip this game
        if (blue_team['playerid'].isna().any() or red_team['playerid'].isna().any() or blue_team['champion'].isna().any() or red_team['champion'].isna().any()):
            print(gameid, "has missing values, we skip this game.")
            continue

        # Create a dictionary to store the values for each column
        game_info = {
            'gameid': gameid,
            'blueTopChampion': blue_team[blue_team['position'] == 'top']['champion'].values[0],
            'blueJungleChampion': blue_team[blue_team['position'] == 'jng']['champion'].values[0],
            'blueMidChampion': blue_team[blue_team['position'] == 'mid']['champion'].values[0],
            'blueCarryChampion': blue_team[blue_team['position'] == 'bot']['champion'].values[0],
            'blueSupportChampion': blue_team[blue_team['position'] == 'sup']['champion'].values[0],

            'redTopChampion': red_team[red_team['position'] == 'top']['champion'].values[0],
            'redJungleChampion': red_team[red_team['position'] == 'jng']['champion'].values[0],
            'redMidChampion': red_team[red_team['position'] == 'mid']['champion'].values[0],
            'redCarryChampion': red_team[red_team['position'] == 'bot']['champion'].values[0],
            'redSupportChampion': red_team[red_team['position'] == 'sup']['champion'].values[0],

            'blueTopPlayerPuuid': (blue_team[blue_team['position'] == 'top']['playerid'].values[0]).replace('oe:player:', ''),
            'blueJunglePlayerPuuid': (blue_team[blue_team['position'] == 'jng']['playerid'].values[0]).replace('oe:player:', ''),
            'blueMidPlayerPuuid': (blue_team[blue_team['position'] == 'mid']['playerid'].values[0]).replace('oe:player:', ''),
            'blueCarryPlayerPuuid': (blue_team[blue_team['position'] == 'bot']['playerid'].values[0]).replace('oe:player:', ''),
            'blueSupportPlayerPuuid': (blue_team[blue_team['position'] == 'sup']['playerid'].values[0]).replace('oe:player:', ''),

            'redTopPlayerPuuid': (red_team[red_team['position'] == 'top']['playerid'].values[0]).replace('oe:player:', ''),
            'redJunglePlayerPuuid': (red_team[red_team['position'] == 'jng']['playerid'].values[0]).replace('oe:player:', ''),
            'redMidPlayerPuuid': (red_team[red_team['position'] == 'mid']['playerid'].values[0]).replace('oe:player:', ''),
            'redCarryPlayerPuuid': (red_team[red_team['position'] == 'bot']['playerid'].values[0]).replace('oe:player:', ''),
            'redSupportPlayerPuuid': (red_team[red_team['position'] == 'sup']['playerid'].values[0]).replace('oe:player:', ''),

            'winner': winner,
        }

        # Add the complete game data to the dataframe as a new row
        new_dataframe = pd.concat([new_dataframe, pd.DataFrame([game_info])], ignore_index=True)

    # Drop duplicate rows based on the 'gameid' column
    new_dataframe = new_dataframe.drop_duplicates(subset='gameid', keep='first')

    # Reset the index of the new DataFrame
    new_dataframe = new_dataframe.reset_index(drop=True)

    # Drop the 'gameid' column
    new_dataframe = new_dataframe.drop(columns=['gameid'])

    # The total number of games we have now after removing duplicates, the ones with missing values, and the 'gameid' column
    print(showCurrentTime() + "Number of games in the new dataframe: " + str(len(new_dataframe)))

    if (len(new_dataframe) == 0):
        print(f"{showCurrentTime()}No games found for the patch {selected_patch:.2f}. File not generated, please select a patch with games inside the pro dataset.")
    else:
        # Save the file
        new_dataframe.to_csv(f'dataset/pro_dataset_{selected_patch:.2f}.csv', index=False)
        print(showCurrentTime() + "The file has been generated in the dataset folder!")
