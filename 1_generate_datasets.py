import time
from config import regions, queueType, rankLeagues, numberOfGamesPerPlayer, startTime, endTime, patch, pro_dataset_name, pro_patch, pro, ranked
from utils.tools import showCurrentTime
from utils.tools import showExecutionTime
from utils.tools import mergeCSVDraftFiles
from utils.tools import transformProDataset
from utils.retrieveGames import retrieveGames

# Store the total starting time to see after the execution time
globalExecutionStartTime = time.time()
print(showCurrentTime() + "Program started!")

# check that the user specified in config file that he wishes to generate the ranked dataset
if ranked:
  # ---Retrieve games---

  # Loop through all regions specified to retrieve games
  for region in regions:
    # Store the starting time to see after the execution time
    executionStartTime = time.time()
    print(showCurrentTime() + "Starting retrieving games for " + region + " with your parameters. Please wait...")
    retrieveGames(region, queueType, rankLeagues, numberOfGamesPerPlayer, startTime, endTime, patch)
    # Printing the execution time
    print(showCurrentTime() + "End of " + region + " games recuperation, execution time is: " + showExecutionTime(executionStartTime))

  # ---Merge games---

  print(showCurrentTime() + "Merging the collected draft files into one...")
  # Merge the draft files of regions and ranks retrieved previously into one
  mergeCSVDraftFiles(regions, rankLeagues)
else:
  print(showCurrentTime() + "You do not wish the ranked dataset to be generated")

# check that the user specified in config file that he wishes to generate the pro dataset
if pro:
  # check if the pro dataset name downloaded was correctly filled in config file
  if pro_dataset_name :

    try:
        transformProDataset("dataset/"+pro_dataset_name, pro_patch)
        print(showCurrentTime() + "Pro dataset has been identified and updated according to expected formats")
        pass
    except FileNotFoundError:
      print(showCurrentTime() + "File not found. Please check the file path or ensure that the file exists !")

  else :
    print(showCurrentTime() + "You haven't mentioned any pro dataset to manage")
else:
  print(showCurrentTime() + "You do not wish the pro dataset to be generated")

# Printing the total execution time
print(showCurrentTime() + "End of program, execution time is: " + showExecutionTime(globalExecutionStartTime))