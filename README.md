# Machine Learning for League of Legends Match Outcome Prediction: A Review

## Requirements

- Python 3.x
- riotwatcher
- pandas
- sklearn
- numpy
- keras
- tensorflow
- A professional dataset placed in the `dataset` folder
- An API key from the Riot Dev API for the ranked dataset’s generation

## Libraries Installation

You need to install the libraries listed in Requirements using your Python or Anaconda environment with the following commands:

```bash
pip install riotwatcher pandas sklearn numpy keras tensorflow
```

## File Structure

The project is structured as follows:

```
├── 1_generate_datasets.py
├── 2_split_dataset.py
├── 3_extract_features_from_games.py
├── 4_apply_method.py
├── config.py
├── dataset
└── utils
    ├── check_balance_dataset.py
    ├── check_feature_vectors_uniqueness.py
    ├── compare_splits.py
    ├── constants.py
    ├── print_columns
    ├── retrieveGames.py
    └── tools.py
```

- `config.py` script: contains the parameters to be used such as the patch, the number of games, the regions, or the game type you wish to use.
- `1_generate_datasets.py` script: generates and reshapes the datasets declared in the `config` file. The process for the ranked dataset may take time, almost 8 hours for 3 regions, a maximum of 200 games, challengers and masters, and for an entire patch duration.
- `dataset/` folder: contains the original professional dataset and the generated ranked and professional datasets from script 1.
- `2_split_dataset.py` script: generates 5 folds of the datasets and puts them in a `data` folder. The resulting structure should be as follows:

```
├── pro
│   ├── fold1
│   │   ├── test
│   │   │   └── feature_vectors.csv
│   │   └── train
│   │       └── feature_vectors.csv
│   ├── fold2
...
│   ├── fold3
...
│   ├── fold4
...
│   └── fold5
...
├── pro_fold1.csv
├── pro_fold2.csv
├── pro_fold3.csv
├── pro_fold4.csv
├── pro_fold5.csv
├── ranked
│   ├── fold1
...
│   ├── fold2
...
│   ├── fold3
...
│   ├── fold4
...
│   └── fold5
...
├── ranked_fold1.csv
├── ranked_fold2.csv
├── ranked_fold3.csv
├── ranked_fold4.csv
└── ranked_fold5.csv
```

- `3_extract_features_from_games.py` script: creates in the folders containing `raw_data.csv` a file called `feature_vectors.csv` containing the feature vector that will be used by the machine learning model. The process may take time, almost 15 min per fold.
- `4_apply_method.py` script: trains each model and tests their efficiency by showing their accuracy and their standard deviation value.
- `utils/` folder: contains the methods used in the project. You don’t have to execute any of the scripts inside.

## How to Run

1. Clone the repository to your local machine.
2. Open a terminal and navigate to the `DL-for-LoL-Match-Outcome-Prediction/` directory.
3. Make sure that Python is installed on your computer with the required libraries, if not, install them.
4. Ensure that every file of the data structure is present, and configure the `config.py` file to the parameters you wish.
5. Run the `1_generate_datasets.py` file by typing the following command:

```bash
python 1_generate_datasets.py
```

This should start the program, generate the CSV files of the ranked games with your selected parameters (if you decided to use ranked games) and transform the professional dataset gathered from Oracle’s Elixir to our format (if you decided to use professional games). CSV files will be located in the `dataset` folder. This program can take a long time, especially if you are retrieving ranked games with a simple key of Riot Developer (which only allows 100 requests every 2 minutes).

6. Run the `2_split_dataset.py` file by typing the following command:

```bash
python 2_split_dataset.py
```

This should start the program, divide the datasets into 5 folds, which will be placed into the `data` folder. Professional and ranked games have their folders, each containing a `fold`. In each `fold` folder, a `test` and `train` folder will contain a `raw_data.csv` file with information about the games.

7. Run the `3_extract_features_from_games.py` file by typing the following command:

```bash
python 3_extract_features_from_games.py
```

This should start the program, extract the features from every game of every fold and then place a `feature_vectors.csv` file in each `test` and `train` folder.

8. Run the `4_apply_method.py` file by typing the following command:

```bash
python 4_apply_method.py
```

This should start the program, apply every method of our program and print their results into the console. It also saves the confusion matrices into a `confusion_matrices` folder if you want to see them. This program may take time to execute depending on your machine.

## Requirements

Contact: robin.heckenauer@gmail.com

Main authors : A. Winterstein & K. Maxel (@github Antonin-Winterstein & Nexusprime22)
