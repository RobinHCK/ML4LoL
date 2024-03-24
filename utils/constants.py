import pandas as pd

# when a new champion is released, you must add the champion name following alphabetical order

# Create a dictionary to map champions to indices (from 0 to 164)
# these are the correct names of the champions
championsList = ["Aatrox", "Ahri", "Akali", "Akshan", "Alistar", "Amumu",
                 "Anivia", "Annie", "Aphelios", "Ashe", "Aurelion Sol",
                 "Azir", "Bard", "Bel'Veth", "Blitzcrank", "Brand", "Braum", "Briar",
                 "Caitlyn", "Camille", "Cassiopeia", "Cho'Gath", "Corki",
                 "Darius", "Diana", "Draven", "Dr. Mundo", "Ekko", "Elise",
                 "Evelynn", "Ezreal", "Fiddlesticks", "Fiora", "Fizz", "Galio",
                 "Gangplank", "Garen", "Gnar", "Gragas", "Graves", "Gwen",
                 "Hecarim", "Heimerdinger", "Illaoi", "Irelia", "Ivern",
                 "Janna", "Jarvan IV", "Jax", "Jayce", "Jhin", "Jinx", "Kai'Sa",
                 "Kalista", "Karma", "Karthus", "Kassadin", "Katarina", "Kayle",
                 "Kayn", "Kennen", "Kha'Zix", "Kindred", "Kled", "Kog'Maw",
                 "K'Sante", "LeBlanc", "Lee Sin", "Leona", "Lillia", "Lissandra",
                 "Lucian", "Lulu", "Lux", "Malphite", "Malzahar", "Maokai",
                 "Master Yi", "Milio", "Miss Fortune", "Mordekaiser",
                 "Morgana", "Naafiri", "Nami", "Nasus", "Nautilus", "Neeko", "Nidalee",
                 "Nilah", "Nocturne", "Nunu & Willump", "Olaf", "Orianna", "Ornn",
                 "Pantheon", "Poppy", "Pyke", "Qiyana", "Quinn", "Rakan",
                 "Rammus", "Rek'Sai", "Rell", "Renata Glasc", "Renekton", "Rengar",
                 "Riven", "Rumble", "Ryze", "Samira", "Sejuani", "Senna",
                 "Seraphine", "Sett", "Shaco", "Shen", "Shyvana", "Singed",
                 "Sion", "Sivir", "Skarner", "Sona", "Soraka", "Swain",
                 "Sylas", "Syndra", "Tahm Kench", "Taliyah", "Talon", "Taric",
                 "Teemo", "Thresh", "Tristana", "Trundle", "Tryndamere",
                 "Twisted Fate", "Twitch", "Udyr", "Urgot", "Varus", "Vayne",
                 "Veigar", "Vel'Koz", "Vex", "Vi", "Viego", "Viktor", "Vladimir",
                 "Volibear", "Warwick", "Wukong", "Xayah", "Xerath", "Xin Zhao", "Yasuo",
                 "Yone", "Yorick", "Yuumi", "Zac", "Zed", "Zeri", "Ziggs",
                 "Zilean", "Zoe", "Zyra"]

# these are the names from the API (wukong is monkeyking for instance)
draft_championsList = ["Aatrox", "Ahri", "Akali", "Akshan", "Alistar", "Amumu",
                       "Anivia", "Annie", "Aphelios", "Ashe", "AurelionSol",
                       "Azir", "Bard", "Belveth", "Blitzcrank", "Brand", "Braum", "Briar",
                       "Caitlyn", "Camille", "Cassiopeia", "Chogath", "Corki",
                       "Darius", "Diana", "Draven", "DrMundo", "Ekko", "Elise",
                       "Evelynn", "Ezreal", "FiddleSticks", "Fiora", "Fizz", "Galio",
                       "Gangplank", "Garen", "Gnar", "Gragas", "Graves", "Gwen",
                       "Hecarim", "Heimerdinger", "Illaoi", "Irelia", "Ivern",
                       "Janna", "JarvanIV", "Jax", "Jayce", "Jhin", "Jinx", "Kaisa",
                       "Kalista", "Karma", "Karthus", "Kassadin", "Katarina", "Kayle",
                       "Kayn", "Kennen", "Khazix", "Kindred", "Kled", "KogMaw",
                       "KSante", "Leblanc", "LeeSin", "Leona", "Lillia", "Lissandra",
                       "Lucian", "Lulu", "Lux", "Malphite", "Malzahar", "Maokai",
                       "MasterYi", "Milio", "MissFortune", "Mordekaiser",
                       "Morgana", "Naafiri", "Nami", "Nasus", "Nautilus", "Neeko", "Nidalee",
                       "Nilah", "Nocturne", "Nunu", "Olaf", "Orianna", "Ornn",
                       "Pantheon", "Poppy", "Pyke", "Qiyana", "Quinn", "Rakan",
                       "Rammus", "RekSai", "Rell", "Renata", "Renekton", "Rengar",
                       "Riven", "Rumble", "Ryze", "Samira", "Sejuani", "Senna",
                       "Seraphine", "Sett", "Shaco", "Shen", "Shyvana", "Singed",
                       "Sion", "Sivir", "Skarner", "Sona", "Soraka", "Swain",
                       "Sylas", "Syndra", "TahmKench", "Taliyah", "Talon", "Taric",
                       "Teemo", "Thresh", "Tristana", "Trundle", "Tryndamere",
                       "TwistedFate", "Twitch", "Udyr", "Urgot", "Varus", "Vayne",
                       "Veigar", "Velkoz", "Vex", "Vi", "Viego", "Viktor", "Vladimir",
                       "Volibear", "Warwick", "MonkeyKing", "Xayah", "Xerath", "XinZhao", "Yasuo",
                       "Yone", "Yorick", "Yuumi", "Zac", "Zed", "Zeri", "Ziggs",
                       "Zilean", "Zoe", "Zyra"]

# championsList not sorted to avoid index confusion
champion_id_map = {champion: index for index, champion in enumerate(championsList)}