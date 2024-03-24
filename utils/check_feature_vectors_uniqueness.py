import pandas as pd

# Specify the file path
file_path = 'data/pro/fold1/train/feature_vectors.csv'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Specify the columns to print
columns_to_print = [
    'blueJungleChampionWinsAgainstRedJungle',
    'blueJungleChampionLossesAgainstRedJungle',
]

# Filter rows where 'Zac' and 'Vi' columns are different from 0
filtered_rows = df[(df['Zac'] != 0) & (df['Vi'] != -0)]

# Print the specified columns for the filtered rows
selected_data = filtered_rows.loc[:, columns_to_print]

print(selected_data)

######################################################################################################################################

# Specify the file path for the new CSV file
raw_data_file_path = 'data/pro/fold1/train/raw_data.csv'

# Load the new CSV file into a pandas DataFrame
raw_data_df = pd.read_csv(raw_data_file_path)

# Filter rows where 'Zac' is the value of column 'blueJungleChampion' and 'Vi' is the value of column 'redJungleChampion'
filtered_rows_raw_data = raw_data_df[(raw_data_df['blueJungleChampion'] == 'Zac') & (raw_data_df['redJungleChampion'] == 'Vi')]

# Print the filtered rows
print(filtered_rows_raw_data)