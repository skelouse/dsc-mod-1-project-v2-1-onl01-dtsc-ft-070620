 ['titleId', 'ordering', 'title', 'region', 'language', 'types',
    'attributes', 'isOriginalTitle']  # columns
name_basics = data['imdb.name.basics']  # saving df to a variable for pysql to access

['titleId', 'ordering', 'title', 'region', 'language', 'types',
    'attributes', 'isOriginalTitle']
title_akas = data['imdb.title.akas']

['tconst', 'titleType', 'primaryTitle', 'originalTitle', 'isAdult',
    'startYear', 'endYear', 'runtimeMinutes', 'genres']
title_basics = data['imdb.title.basics']

['tconst', 'directors', 'writers']
title_crew = data['imdb.title.crew']

['tconst', 'parentTconst', 'seasonNumber', 'episodeNumber']
title_episode = data['imdb.title.episode']

['tconst', 'ordering', 'nconst', 'category', 'job', 'characters']
title_principals = data['imdb.title.principals']

['tconst', 'averageRating', 'numVotes']
title_ratings = data['imdb.title.ratings']

['tconst', 'budget', 'ww_gross', 'rating']
scraped_money = data['scraped_money']