import pandas as pd

data_types = ["train", "test"]
game_type = "ranked"

for data_type in data_types:
    for i in range(5):

        input_path = "data/"+game_type+"/fold"+str(i+1)+"/"+data_type+"/feature_vectors.csv"

        print(f"\n--------\nFor {game_type} {data_type} split {i+1}")
        print(input_path)

        df = pd.read_csv(input_path)

        print(df["winner"][0])

        # Specify the column you want to check for balance
        target_column = 'winner'

        # Check the distribution of values in the specified column
        value_counts = df[target_column].value_counts()

        # Display the counts
        print(value_counts)

        # Calculate the balance ratio
        balance_ratio = value_counts.min() / value_counts.max()

        # Print the balance ratio
        print(f"Balance Ratio: {balance_ratio}")

print("\n------\nBALANCE RATIO ON MAIN DATASET")
# Read the CSV file into a DataFrame
file_path = 'dataset/ranked_dataset_13.20.csv'
df = pd.read_csv(file_path)

# Specify the column you want to check for balance
target_column = 'winner'

# Check the distribution of values in the specified column
value_counts = df[target_column].value_counts()

# Display the counts
print(value_counts)

# Calculate the balance ratio
balance_ratio = value_counts.min() / value_counts.max()

# Print the balance ratio
print(f"Balance Ratio: {balance_ratio}")