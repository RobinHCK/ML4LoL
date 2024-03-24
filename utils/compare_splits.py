import pandas as pd

data_types = ["train", "test"]
game_type = "ranked"

def compare_splits(split1, split2, data_type):
    input_path1 = f"data/{game_type}/fold{split1}/{data_type}/feature_vectors.csv"
    input_path2 = f"data/{game_type}/fold{split2}/{data_type}/feature_vectors.csv"

    print(f"\n--------\nComparing {game_type} {data_type} split {split1} and split {split2}")

    df1 = pd.read_csv(input_path1)
    df2 = pd.read_csv(input_path2)

    # Check if the DataFrames are equal
    if df1.equals(df2):
        print("The CSV files are the same.")
    else:
        print("The CSV files are different.")

def check_rows_in_other_split(split1, split2, data_type):
    input_path1 = f"data/{game_type}/fold{split1}/{data_type}/feature_vectors.csv"
    input_path2 = f"data/{game_type}/fold{split2}/{data_type}/feature_vectors.csv"

    print(f"Checking for rows in split {split1} present in split {split2} for {game_type} {data_type}")

    df1 = pd.read_csv(input_path1)
    df2 = pd.read_csv(input_path2)

    # Identify common rows between the two splits based on all columns
    common_rows = pd.merge(df1, df2, how='inner')

    if not common_rows.empty:
        print("Common rows:")
        print(common_rows)
    else:
        print("No common rows found.")


# Compare CSV files between different splits
for data_type in data_types:
    for i in range(1, 5):  # Compare all splits with each other
        compare_splits(i, i+1, data_type)
        check_rows_in_other_split(i, i+1, data_type)