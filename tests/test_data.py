def main(data):
    import pandasql as ps
    ['titleId', 'ordering', 'title', 'region', 'language', 'types',
       'attributes', 'isOriginalTitle']
    name_basics = data['imdb.name.basics']

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


    q1 = """
SELECT tconst FROM title_ratings
JOIN title_basics
USING(tconst)
WHERE startYear > 2000
"""
    qlist = ''
    x = 0
    df = ps.sqldf(q1, locals())
    for i in df['tconst']:
        x+=1
        qlist += str(i+',')
    
    qlist += '\b'
    print(x)
    with open('qlist.txt', 'w') as f:
        f.write(qlist)


def make_info(data):
    for df in data.items():
        print("*"*20)
        print("*"*20)
        print("\n%s\n" % df[0])
        print("*"*20)
        print('describe()')
        print(df[1].describe())
        print("*"*20)
        print('info()')
        print(df[1].info())
        print("*"*20)
        print('.iloc[0]')
        print(df[1].iloc[0])
        print("*"*20)
        print("*"*20)
