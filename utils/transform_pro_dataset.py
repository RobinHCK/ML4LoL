import pandas as pd

import sys
import os

# Add the root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import pro_dataset_name from config.py at root
from config import pro_dataset_name, pro_patch

# Specify the CSV file path
file_name = pro_dataset_name
csv_file = 'dataset/'+file_name

# Read the CSV file into a pandas DataFrame
original_dataframe = pd.read_csv(csv_file)

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

print(f"The patch with the highest amount of games is: {highest_patch:.2f}")

# Display the result with the desired order
result = result[['patch', 'Counts', 'Top Leagues']]

print(result)

# If the user provided a specific patch, get games of this patch, otherwise get games of the one with the highest number of games
if pro_patch != None:
  selected_patch = float(pro_patch)
  print(f"The patch you decided to use (the one specified in the config file) is: {selected_patch:.2f}")
else:
  selected_patch = highest_patch
  print(f"The patch we will use is: {selected_patch:.2f}")

# Update the dataframe to only use games from the patch highest patch variable (of any region and division)
updated_dataframe = original_dataframe.loc[original_dataframe['patch'] == selected_patch]

# Check that we indeed have only games of patch highest patch variable in the updated dataframe
print("Unique values of the patch column:", updated_dataframe['patch'].unique())
# The number of games should be the same as before
print("Number of games of the updated dataframe:", len(updated_dataframe) // 12)

# Format of the new dataset
dataset_columns_format = [
    'blueTopChampion', 'blueJungleChampion', 'blueMidChampion', 'blueCarryChampion', 'blueSupportChampion',
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

# Reset the index of the new DataFrame
new_dataframe = new_dataframe.reset_index(drop=True)

# Print some games to see
print(new_dataframe.head(5).to_string())

# The total number of games we have now after removing the ones with missing values
print("Number of games of the new dataframe:", len(new_dataframe))

if (len(new_dataframe) == 0):
  print(f"No games found for the patch {selected_patch:.2f}. File not generated, please select a patch with games inside the pro dataset.")
else:
  # Save the file
  new_dataframe.to_csv(f'dataset/pro_dataset_{selected_patch:.2f}.csv', index=False)
