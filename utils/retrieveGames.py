from utils.tools import convertDatetimeToTimestamp
from utils.tools import showCurrentTime
from utils.tools import jsonObjectToCSV
from config import API_KEY
# Please install Riot Watcher, (`pip install riotwatcher`)
from riotwatcher import LolWatcher, ApiError

# Start lolWatcher using the API key
lolWatcher = LolWatcher(API_KEY)
# The array that will store every games
globalGameStringArray = []

def getSummonersFromLeague(region: str, queueType: str, rankLeague: str):
  """
  Gets all players from challenger, grandmaster or master of a specific region and a specific queue (Solo queue or flex)

  :param str region: The region of the summoners
  :param str queueType: The queue type (RANKED_SOLO_5x5 or RANKED_FLEX_SR)
  :param str rankLeague: The rank league (challenger, grandmaster or master)
  :return: a list containing summoners ids and names of summoners retrieved from the corresponding region, queue and league
  """

  leagueData = None

  if (queueType != "RANKED_SOLO_5x5" and queueType != "RANKED_FLEX_SR"):
    print(showCurrentTime() + "The queue type specified is wrong. Please provide a valid one (RANKED_SOLO_5x5 or RANKED_FLEX_SR).")
    return

  # Call to API and store data
  try:
    if(rankLeague == "challenger"):
      leagueData = lolWatcher.league.challenger_by_queue(region, queueType)

    if(rankLeague == "grandmaster"):
      leagueData = lolWatcher.league.grandmaster_by_queue(region, queueType)

    if(rankLeague == "master"):
      leagueData = lolWatcher.league.masters_by_queue(region, queueType)

  except:
    print(showCurrentTime() + "An exception occured while trying to get data of the " + queueType + " league: " + rankLeague + ". (" + region + ")")


  if (leagueData != None):
    entriesArrayLength = len(leagueData['entries'])

    entryStringArray = []
    # Loop through all entries of the league (i.e each player of the league)
    for i in range(entriesArrayLength):
      entry = leagueData['entries'][i]

      # Create a new object with the entry (summoner) informations
      entryObject = {
        "summonerId": entry['summonerId'],
        "summonerName": entry['summonerName']
      }

      # Adds the entry (summoner) to the array
      entryStringArray.append(entryObject)

    # Call the function to create the CSV file
    jsonObjectToCSV(entryStringArray, region, rankLeague = rankLeague, dataType = "summoners")

    return entryStringArray
  else:
    print(showCurrentTime() + "Error. Can't find data of the " + queueType + " league: " + rankLeague + ". (" + region + ")")


def getGameHistoryFromSummoner(region: str, summonerId: str, numberOfGames: int = 20, queueType : str = "", startTime: int = None, endTime: int = None):
  """
  Gets the games of the history of a summoner.

  :param str region: The region of the summoner
  :param summonerId: The summoner id
  :param numberOfGames: The number of games that we want to retrieve from the summoner
  :param queueType: The queue type (RANKED_SOLO_5x5 or RANKED_FLEX_SR)
  :param startTime: the date from where we want to start from represented using the epoch timestamp, if not provided it takes every game
  :param endTime: the date from where we want to end represented using the epoch timestamp, if not provided it takes every game
  :return: a list containing ids of all games retrieved
  """
  summonerData = None
  summonerpuuid = None
  # Call to API and store data
  try:
    summonerData = lolWatcher.summoner.by_id(region, summonerId)
    summonerpuuid = summonerData['puuid']
  except:
    print(showCurrentTime() + "An exception occured while trying to get data of the summoner: " + summonerId + ". (" + region + ")")

  if (summonerData != None and summonerpuuid != None):
    summonerGameHistoryData = None
    global globalGameStringArray
    gameStringArray = []
    j = 0
    startIndex = 0
    numberOfRequest = numberOfGames / 100
    count = 100

    # Queue id 420 is the one for ranked solo queue in Summoner's Rift
    # Link to all queues: https://static.developer.riotgames.com/docs/lol/queues.json
    if (queueType == "RANKED_SOLO_5x5"):
      queueId = 420
    elif (queueType == "RANKED_FLEX_SR"):
      queueId = 440
    else:
      print(showCurrentTime() + "The queue type specified is wrong. Please provide a valid one (RANKED_SOLO_5x5 or RANKED_FLEX_SR).")
      return

    # Loop numberOfRequest times to get all games id
    while j < numberOfRequest:
      # Check for the rest of games
      if (startIndex + numberOfGames < startIndex + 100):
        count = numberOfGames % 100
      else:
        count = 100

      # Call to API and store data
      try:
        summonerGameHistoryData = lolWatcher.match.matchlist_by_puuid(region, summonerpuuid, startIndex, count, queueId, "ranked", startTime, endTime)

      except:
          print(showCurrentTime() + "An exception occured while trying to get the game history data of the summoner: " + summonerId + ". (" + region + ")")

      if (summonerGameHistoryData != None):
        gameHistoryArrayLength = len(summonerGameHistoryData)

        # Loop through all match id
        for i in range(gameHistoryArrayLength):
          gameId = summonerGameHistoryData[i]

          # Check if the gameId doesn't already exist in the globalGameStringArray
          if not any(game["gameId"] == gameId for game in globalGameStringArray):
            # Create a new object with the game id
            gameObject = {
              "gameId": gameId
            }

            # Add the game id to the local and global array
            globalGameStringArray.append(gameObject)
            gameStringArray.append(gameObject)
      else:
        print(showCurrentTime() + "Error. Can't find the game history data of " + summonerId + ". (" + region + ")")

      # Add 100 to j, to the start index and remove 100 games of the numberOfGames
      j += 1
      startIndex += 100
      numberOfGames -= 100

    return gameStringArray
  else:
    print(showCurrentTime() + "Error. Can't find the summoner data of " + summonerId + ". (" + region + ")")

def getGamesOfPlayersFromLeague(region: str, numberOfGames: int = 20, queueType: str = "", rankLeague: str = "challenger", startTime: int = None, endTime: int = None):
  """
  Gets games of all players from a specific queue (Solo queue or flex) for a specific rank league (challenger, grandmaster or master)

  :param str region: The region of the summoners
  :param numberOfGames: The number of games that we want to retrieve from the summoner
  :param str queueType: The queue type (RANKED_SOLO_5x5 or RANKED_FLEX_SR)
  :param str rankLeague: The rank league (challenger, grandmaster or master)
  :param startTime: the date from where we want to start from represented using the epoch timestamp, if not provided it takes every game
  :param endTime: the date from where we want to end represented using the epoch timestamp, if not provided it takes every game
  :return: a list containing games ids of games retrieved from the corresponding region, queue and league
  """
  # Store the summoners names and ids of a particular region, queue and league inside an array 
  playersFromLeague = getSummonersFromLeague(region, queueType, rankLeague)

  # The array that will hold games
  gameStringArray = []

  # Loop through all the players retrieved
  for summoner in playersFromLeague:
    summonerId = summoner["summonerId"]
    # Add the games of the user to the gameStringArray
    gameStringArray = gameStringArray + getGameHistoryFromSummoner(region, summonerId, numberOfGames, queueType, startTime, endTime)

  jsonObjectToCSV(gameStringArray, region, rankLeague = rankLeague, dataType = "games")
  return gameStringArray

def getDraftGameData(gameId, region, rankLeague, patch):
  """
  Gets draft data about a game (champions and players played in each team) with the winning team and store it inside a CSV.

  :param str gameId: The game id
  :param str region: The region of the game
  :param rankLeague: The rank league (challenger, grandmaster or master)
  :param patch: The patch version to match the game version against (optional)
  """
  gameDataId = None
  # Call to API and store data
  try:
    gameDataId = lolWatcher.match.by_id(region, gameId)
  except:
      print(showCurrentTime() + "An exception occured while trying to get data of the game: " + gameId + ". (" + region + ")")

  # For some reason, some games have no data (remake possible)
  if gameDataId != None:
      gameVersion = gameDataId['info']['gameVersion']

      # Check if patch is provided and if it matches the game version
      if patch is not None and gameVersion.find(patch):
          print(showCurrentTime() + f"Error. The game version '{gameVersion}' does not match the specified patch '{patch}'")
          return

      if (gameDataId['info']['gameId'] != 0):
          draftObject = {
              "blueTopChampion": "",
              "blueJungleChampion": "",
              "blueMidChampion": "",
              "blueCarryChampion": "",
              "blueSupportChampion": "",

              "redTopChampion": "",
              "redJungleChampion": "",
              "redMidChampion": "",
              "redCarryChampion": "",
              "redSupportChampion": "",

              "blueTopPlayerPuuid": "",
              "blueJunglePlayerPuuid": "",
              "blueMidPlayerPuuid": "",
              "blueCarryPlayerPuuid": "",
              "blueSupportPlayerPuuid": "",

              "redTopPlayerPuuid": "",
              "redJunglePlayerPuuid": "",
              "redMidPlayerPuuid": "",
              "redCarryPlayerPuuid": "",
              "redSupportPlayerPuuid": "",

              "winner": ""
          }

          # Loop through the 10 participants
          for participant in gameDataId['info']['participants']:
              champion = participant['championName']
              role = participant['teamPosition']
              teamSide = participant['teamId']
              playerPuuid = participant['puuid']
              # If teamId = 100, it's blue side
              # If teamId = 200, it's red side,
              if (teamSide == 100):
                  teamSide = "blue"
              else:
                  teamSide = "red"

              match(role):
                  case "TOP":
                      role = "Top"
                  case "JUNGLE":
                      role = "Jungle"
                  case "MIDDLE":
                      role = "Mid"
                  case "BOTTOM":
                      role = "Carry"
                  case "UTILITY":
                      role = "Support"
                  # If a role is empty, just don't create the CSV (example: EUW1_6408571839)
                  case "":
                      print(showCurrentTime() + "Error. There are missing values for " + gameId + ". (" + region + ")")
                      return

              # Update the according column to put the champion name
              championColumn = teamSide + role + "Champion"
              draftObject[championColumn] = champion

              # Update the according column to put the player puuid
              playerPuuidColumn = teamSide + role + "PlayerPuuid"
              draftObject[playerPuuidColumn] = playerPuuid

          for team in gameDataId['info']['teams']:
              teamSide = team['teamId']
              winBool = team['win']
              # Check which team is the winner
              if (teamSide == 100 and winBool == True):
                  draftObject["winner"] = "Blue Team"
              if (teamSide == 200 and winBool == True):
                  draftObject["winner"] = "Red Team"

          draftObjectArray = [draftObject]
          # Create the draft CSV for the game
          jsonObjectToCSV(draftObjectArray, region, rankLeague = rankLeague, dataType = "drafts", gameId = gameId)

      else:
          print(showCurrentTime() + "Error. Can't find the game data of " + gameId + ". (" + region + ")")

def retrieveGames(region: str, queueType: str, rankLeague: list, numberOfGames: int = 20, startTime: str = None, endTime: str = None, patch: str = None):
  """
  Gets games of all players from a specific queue (Solo queue or flex) for specific rank leagues (challenger, grandmaster or master) in a specific region (EUW1, KR, NA1) and store them in CSVs

  :param str region: The region of the summoners
  :param str queueType: empty if we want both flex and solo duo, otherwise RANKED_SOLO_5x5 or RANKED_FLEX_SR
  :param list rankLeague: A list containing the rank leagues (challenger, grandmaster or master)
  :param int numberOfGames: The number of games that we want to retrieve from the summoner
  :param str startTime: the date from where we want to start from represented using the format [Year-Month-Day Hour:minute:second], if not provided it takes every game
  :param str endTime: the date from where we want to end represented using the format [Year-Month-Day Hour:minute:second], if not provided it takes every game
  :param str patch: The patch version to match the game version against (optional)
  :return: a list containing ids of all games retrieved
  """

  # Reset the array that holds game ids
  global globalGameStringArray
  globalGameStringArray.clear()

  # Check if we provided a startTime
  if startTime != None:
    # Convert the datetime to timestamp
    startTime = convertDatetimeToTimestamp(startTime)
    # If an error occured when converting the datetime to timestamp, stop the function
    if startTime == None:
      return

  # Check if we provided an endTime
  if endTime != None:
    # Convert the datetime to timestamp
    endTime = convertDatetimeToTimestamp(endTime)
    # If an error occured when converting the datetime to timestamp, stop the function
    if endTime == None:
      return

  gamesOfPlayersFromLeagueArray = []

  try:
    # Loop through all leagues to get games of each player of the league
    for league in rankLeague:
      # Get games of all players in the current league
      gamesOfPlayersFromLeague = getGamesOfPlayersFromLeague(region, numberOfGames, queueType, league, startTime, endTime)

      gamesOfPlayersFromLeagueArray.append(list(gamesOfPlayersFromLeague))

    total_length = 0
    # Just to print the total length of games
    for gamesOfPlayersFromLeague in gamesOfPlayersFromLeagueArray:
      total_length += len(gamesOfPlayersFromLeague)

    print(showCurrentTime() + "Total number of unique games for " + region + " with your parameters is: " + str(total_length))

  except:
    print(f"{showCurrentTime()}An exception occured while trying to get the list of games of players from {queueType} league: {rankLeague} ({region})")

  try:
    # Loop through all league games arrays
    for i, gamesOfPlayersFromLeague in enumerate(gamesOfPlayersFromLeagueArray):
      league = rankLeague[i]
      # Loop through all games retrieved
      for game in gamesOfPlayersFromLeague:
        gameId = game['gameId']
        getDraftGameData(gameId, region, league, patch)
  except:
    print(f"{showCurrentTime()}An exception occured while trying to get the drafts of games of players from {queueType} league: {rankLeague} ({region})")