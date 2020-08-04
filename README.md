## Final Project Submission

Please fill out:
* Student name: Sam Stoltenberg
* Student pace: full time
* Scheduled project review date/time: 8/7/2020 6:00PM 
* Instructor name: James Irving
* Blog post URL:  https://skelouse.github.io/project_placeholder



```python
# Relevant imports
import os
import pandas as pd
import pandasql as ps
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('mode.chained_assignment', None)
```


```python
store_folder = 'Data'
data = {}  # dictionary of dataframes
total_size = 0  # size of all of the files
for filename in os.listdir(store_folder):
    path = (store_folder+'/'+filename)
    size = os.stat(path).st_size/(1024**2)  # gets the size of the file in MB
    total_size += size
    print("reading", filename, size, 'MB')
    try:
        if path.endswith('tsv.gz'):  # for tsv files
            data[filename[:-7]] = pd.read_csv(path,
                                                delimiter='\t',
                                                low_memory=False)
        else:  # for csv files
            data[filename[:-7]] = pd.read_csv(path,
                                                low_memory=False)
    except UnicodeDecodeError:
        # Changing encoding to reflect utf-8 unknown characters
        if path.endswith('tsv.gz'):
            data[filename[:-7]] = pd.read_csv(path,
                                            delimiter='\t',
                                            low_memory=False,
                                            encoding='windows-1252')
        else:
            data[filename[:-7]] = pd.read_csv(path,
                                            low_memory=False,
                                            encoding='windows-1252')
print('Done loading %s MB' % total_size)
```

    reading imdb.name.basics.tsv.gz 190.70305347442627 MB
    reading imdb.title.akas.tsv.gz 189.05681037902832 MB
    reading imdb.title.basics.tsv.gz 119.10677146911621 MB
    reading imdb.title.crew.tsv.gz 47.09960460662842 MB
    reading imdb.title.episode.tsv.gz 26.076172828674316 MB
    reading imdb.title.principals.tsv.gz 314.29254722595215 MB
    reading imdb.title.ratings.tsv.gz 4.99261474609375 MB
    Done loading 891.3275747299194 MB
    


```python
# Open df from csv that was scraped as a group
data['scraped_money'] = pd.read_csv('scraped_data/budget_ratings.csv', index_col=0)
```


```python
# Loading each of the csv/tsv files into variables for pysql
# Showing the columns for each for reference

['nconst', 'primaryName', 'birthYear', 'deathYear', 'primaryProfession', 'knownForTitles']
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

['tconst', 'budget', 'ww_gross', 'rating']
scraped_money = data['scraped_money']
```


```python
# Setting the index of scraped_money to match the rest
scraped_money = scraped_money.set_index('tconst')
```


```python
# Taking out the top ratings that have needed values(budget, gross, etc)
# Cleaning ratings below, because of genres in rating column
df = scraped_money
ratings = ['G', 'PG', 'PG-13', 'R']
df['rating'].value_counts().head(10)
```




    NotRated       5834
    R              2272
    PG-13          1265
    TV-MA           949
    Drama           871
    Documentary     811
    TV-14           760
    PG              723
    Unrated         663
    Comedy          573
    Name: rating, dtype: int64




```python
# Cleaning up scraped money to remove zeros and null values
# Setting each relevent column to integers
df['rating'] = df['rating'].map(lambda x: np.NaN if x not in ratings else x)
df['budget'] = df['budget'].map(lambda x: np.NaN if x in[None, 0, '0'] else x)
df['gross'] = df['gross'].map(lambda x: np.NaN if x in[None, 0, '0'] else x)
df['ww_gross'] = df['ww_gross'].map(lambda x: np.NaN if x in[None, 0] else x)
df = df.dropna()
df['budget'] = df['budget'].astype('int64')
df['ww_gross'] = df['ww_gross'].astype('int64')
df['gross'] = df['gross'].astype('int64')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>gross</th>
      <th>ww_gross</th>
      <th>rating</th>
    </tr>
    <tr>
      <th>tconst</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>tt3567288</th>
      <td>5000000</td>
      <td>65206105</td>
      <td>98450062</td>
      <td>PG-13</td>
    </tr>
    <tr>
      <th>tt3569230</th>
      <td>30000000</td>
      <td>1872994</td>
      <td>42972994</td>
      <td>R</td>
    </tr>
    <tr>
      <th>tt3576728</th>
      <td>30000000</td>
      <td>490973</td>
      <td>3087832</td>
      <td>PG-13</td>
    </tr>
    <tr>
      <th>tt3602422</th>
      <td>1310000</td>
      <td>38901</td>
      <td>38901</td>
      <td>R</td>
    </tr>
    <tr>
      <th>tt3605418</th>
      <td>2500000</td>
      <td>36336</td>
      <td>5567103</td>
      <td>R</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>tt1605803</th>
      <td>9000000</td>
      <td>27000</td>
      <td>27000</td>
      <td>PG-13</td>
    </tr>
    <tr>
      <th>tt1606378</th>
      <td>92000000</td>
      <td>67349198</td>
      <td>304654182</td>
      <td>R</td>
    </tr>
    <tr>
      <th>tt1606389</th>
      <td>30000000</td>
      <td>125014030</td>
      <td>196114570</td>
      <td>PG-13</td>
    </tr>
    <tr>
      <th>tt1608290</th>
      <td>50000000</td>
      <td>28848693</td>
      <td>56722693</td>
      <td>PG-13</td>
    </tr>
    <tr>
      <th>tt1609488</th>
      <td>350000</td>
      <td>40645</td>
      <td>40645</td>
      <td>R</td>
    </tr>
  </tbody>
</table>
<p>1343 rows × 4 columns</p>
</div>




```python
# Which is more profitable, domestic or worldwide markets?
new_df = df.copy(deep=True)[0:100]

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12.5, 8.5))
sns.regplot("budget", "ww_gross", data=df, ax=ax)
sns.regplot("budget", "gross", data=df, ax=ax)

ax.legend(["Worldwide Gross", "Domestic Gross"])
ax.set_title("Domestic & Worldwide Gross to Budget")
ax.title.set_fontsize(20)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
ax.xaxis.label.set_text("Budget (Billions)")
ax.yaxis.label.set_text("Gross (Billions)")

# As you can see, movies that are worldwide make more money than domestic movies
```


![svg](img/output_8_0.svg)



```python
# Take df(scraped_money) and add a column with the % difference of budget-> ww_gross
# i.e ROI % (Return on investment %)
def compare(budget, gross):
    net = gross - budget
    return net/budget
df['perc'] = compare(df['budget'], df['ww_gross'])
df.sort_values(by='perc', ascending=False)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>gross</th>
      <th>ww_gross</th>
      <th>rating</th>
      <th>perc</th>
    </tr>
    <tr>
      <th>tconst</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>tt1560985</th>
      <td>1000000</td>
      <td>53261944</td>
      <td>101758490</td>
      <td>R</td>
      <td>100.758490</td>
    </tr>
    <tr>
      <th>tt7668870</th>
      <td>880000</td>
      <td>26020957</td>
      <td>75462037</td>
      <td>PG-13</td>
      <td>84.752315</td>
    </tr>
    <tr>
      <th>tt1591095</th>
      <td>1500000</td>
      <td>54009150</td>
      <td>99557032</td>
      <td>PG-13</td>
      <td>65.371355</td>
    </tr>
    <tr>
      <th>tt3713166</th>
      <td>1000000</td>
      <td>32482090</td>
      <td>62882090</td>
      <td>R</td>
      <td>61.882090</td>
    </tr>
    <tr>
      <th>tt5052448</th>
      <td>4500000</td>
      <td>176040665</td>
      <td>272495873</td>
      <td>R</td>
      <td>59.554638</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>tt5303442</th>
      <td>20000000</td>
      <td>11255</td>
      <td>11255</td>
      <td>R</td>
      <td>-0.999437</td>
    </tr>
    <tr>
      <th>tt4414438</th>
      <td>18000000</td>
      <td>7162</td>
      <td>7162</td>
      <td>R</td>
      <td>-0.999602</td>
    </tr>
    <tr>
      <th>tt0762138</th>
      <td>2500000</td>
      <td>663</td>
      <td>663</td>
      <td>R</td>
      <td>-0.999735</td>
    </tr>
    <tr>
      <th>tt5143890</th>
      <td>18000000</td>
      <td>3259</td>
      <td>3259</td>
      <td>R</td>
      <td>-0.999819</td>
    </tr>
    <tr>
      <th>tt3789946</th>
      <td>2124000</td>
      <td>120</td>
      <td>120</td>
      <td>PG</td>
      <td>-0.999944</td>
    </tr>
  </tbody>
</table>
<p>1343 rows × 5 columns</p>
</div>




```python
q1 = """
SELECT DISTINCT tconst, primaryTitle, genres, budget, ww_gross, rating, perc, runtimeMinutes, directors, writers
FROM df
JOIN title_basics tb
USING(tconst)
JOIN title_crew tc
USING(tconst)
JOIN title_akas ta
ON tconst = titleId
"""
```


```python
# Join directors, writers, runtime, title, and genres to my scraped dataframe on 'tconst'.
called_df = ps.sqldf(q1, locals())
called_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>genres</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>rating</th>
      <th>perc</th>
      <th>runtimeMinutes</th>
      <th>directors</th>
      <th>writers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0326965</td>
      <td>In My Sleep</td>
      <td>Drama,Mystery,Thriller</td>
      <td>1000000</td>
      <td>30158</td>
      <td>PG-13</td>
      <td>-0.969842</td>
      <td>104</td>
      <td>nm1075006</td>
      <td>nm1075006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0337692</td>
      <td>On the Road</td>
      <td>Adventure,Drama,Romance</td>
      <td>25000000</td>
      <td>9617377</td>
      <td>R</td>
      <td>-0.615305</td>
      <td>124</td>
      <td>nm0758574</td>
      <td>nm0449616,nm1433580</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0359950</td>
      <td>The Secret Life of Walter Mitty</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>90000000</td>
      <td>188133322</td>
      <td>PG</td>
      <td>1.090370</td>
      <td>114</td>
      <td>nm0001774</td>
      <td>nm0175726,nm0862122</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0365907</td>
      <td>A Walk Among the Tombstones</td>
      <td>Action,Crime,Drama</td>
      <td>28000000</td>
      <td>58834384</td>
      <td>R</td>
      <td>1.101228</td>
      <td>114</td>
      <td>nm0291082</td>
      <td>nm0088747,nm0291082</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0369610</td>
      <td>Jurassic World</td>
      <td>Action,Adventure,Sci-Fi</td>
      <td>150000000</td>
      <td>1670400637</td>
      <td>PG-13</td>
      <td>10.136004</td>
      <td>124</td>
      <td>nm1119880</td>
      <td>nm0415425,nm0798646,nm1119880,nm2081046,nm0000341</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1332</th>
      <td>tt9208444</td>
      <td>Impractical Jokers: The Movie</td>
      <td>Comedy</td>
      <td>3000000</td>
      <td>10691091</td>
      <td>PG-13</td>
      <td>2.563697</td>
      <td>92</td>
      <td>nm0376260</td>
      <td>nm0376260,nm2665746,nm2098978,nm1978079,nm1742600</td>
    </tr>
    <tr>
      <th>1333</th>
      <td>tt9214832</td>
      <td>Emma.</td>
      <td>Comedy,Drama</td>
      <td>10000000</td>
      <td>25587304</td>
      <td>PG</td>
      <td>1.558730</td>
      <td>124</td>
      <td>nm2127315</td>
      <td>nm7414254,nm0000807</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>tt9354944</td>
      <td>Jexi</td>
      <td>Comedy,Romance</td>
      <td>5000000</td>
      <td>9342073</td>
      <td>R</td>
      <td>0.868415</td>
      <td>84</td>
      <td>nm0524190,nm0601859</td>
      <td>nm0524190,nm0601859</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>tt9426210</td>
      <td>Weathering with You</td>
      <td>Animation,Drama,Family</td>
      <td>11100000</td>
      <td>193168568</td>
      <td>PG-13</td>
      <td>16.402574</td>
      <td>112</td>
      <td>nm1396121</td>
      <td>nm1396121</td>
    </tr>
    <tr>
      <th>1336</th>
      <td>tt9779516</td>
      <td>I Still Believe</td>
      <td>Biography,Drama,Music</td>
      <td>12000000</td>
      <td>11502842</td>
      <td>PG</td>
      <td>-0.041430</td>
      <td>116</td>
      <td>nm3401779,nm2296528</td>
      <td>nm1705229,nm2296528,nm0348197</td>
    </tr>
  </tbody>
</table>
<p>1337 rows × 10 columns</p>
</div>




```python
# Top genre list from Max. Generated by comparing each genre median/mean to the median/mean of all.
# https://github.com/zero731
genre_list = ['Thriller',  'Animation',  'Sci-Fi',  'Mystery',  'Music',  'Adventure',  'Fantasy',  'Comedy']
def fix_genre(genres):
    """Function to split the genres into lists, and remove genres that arn't in the list"""
    new_genres = []
    genres = genres.split(',')
    for g in genres:
        if g in genre_list:
            new_genres.append(g)
    return new_genres
genre_df = called_df.copy(deep=True).dropna()
genre_df['genres'] = genre_df['genres'].apply(lambda x: fix_genre(x))
# Remove movies that no longer have a genre
genre_df['genres'] = genre_df['genres'].map(lambda x: np.NaN if not x else x)
genre_df = genre_df.dropna()
```


```python
# Explode genres
# i.e  ['Comedy', 'Animation'] to one row of 'Animation' and one of 'Comedy'
genre_df = genre_df.explode(column='genres')
```


```python
# Sort by ROI %
genre_df = genre_df.sort_values(by='perc', ascending=False)
genre_df['runtimeMinutes'] = genre_df['runtimeMinutes'].astype('int64')
```


```python
# What would be the optimal runtime of a movie
fig, ax = plt.subplots(figsize=(12.5, 8.5))
sns.distplot(genre_df['runtimeMinutes'][0:500], bins=10, ax=ax)
ax.set_title("Run Time Compared vs Return On Investment %")
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
ax.xaxis.label.set_text("Runtime (Minutes)")
ax.yaxis.label.set_text("Return on Investment %")
ax.tick_params(axis='both', which='major', labelsize=15)

# A runtime of 90-110 minutes would be ideal
```


![svg](img/output_15_0.svg)



```python
# Which genre would make the most
fig, ax = plt.subplots(figsize=(10, 6.5))
dicta = genre_df['genres'][0:200].value_counts().to_dict()
ax.bar(dicta.keys(), dicta.values())

dicta = genre_df['genres'][0:100].value_counts().to_dict()
ax.bar(dicta.keys(), dicta.values())

dicta = genre_df['genres'][0:50].value_counts().to_dict()
ax.bar(dicta.keys(), dicta.values())

ax.legend(["Top 200", "Top 100", "Top 50"])
ax.set_title("Genre vs Return On Investment %")
ax.title.set_fontsize(20)
ax.yaxis.label.set_fontsize(17)
ax.yaxis.label.set_text("Quanity")
ax.tick_params(axis='both', which='major', labelsize=11)

# Comedy or thriller seem to take the cake
```


![svg](img/output_16_0.svg)



```python
# Creating a dataframe of all the names to pull out director's and writer's names
q2 = """
SELECT *
FROM name_basics
"""
name_df = ps.sqldf(q2, locals())
name_df  # Name Dataframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nconst</th>
      <th>primaryName</th>
      <th>birthYear</th>
      <th>deathYear</th>
      <th>primaryProfession</th>
      <th>knownForTitles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm0000001</td>
      <td>Fred Astaire</td>
      <td>1899</td>
      <td>1987</td>
      <td>soundtrack,actor,miscellaneous</td>
      <td>tt0053137,tt0031983,tt0050419,tt0072308</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm0000002</td>
      <td>Lauren Bacall</td>
      <td>1924</td>
      <td>2014</td>
      <td>actress,soundtrack</td>
      <td>tt0071877,tt0038355,tt0117057,tt0037382</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm0000003</td>
      <td>Brigitte Bardot</td>
      <td>1934</td>
      <td>\N</td>
      <td>actress,soundtrack,music_department</td>
      <td>tt0054452,tt0057345,tt0059956,tt0049189</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm0000004</td>
      <td>John Belushi</td>
      <td>1949</td>
      <td>1982</td>
      <td>actor,soundtrack,writer</td>
      <td>tt0078723,tt0080455,tt0077975,tt0072562</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm0000005</td>
      <td>Ingmar Bergman</td>
      <td>1918</td>
      <td>2007</td>
      <td>writer,director,actor</td>
      <td>tt0060827,tt0050976,tt0083922,tt0050986</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10248840</th>
      <td>nm9993714</td>
      <td>Romeo del Rosario</td>
      <td>\N</td>
      <td>\N</td>
      <td>animation_department,art_department</td>
      <td>tt2455546</td>
    </tr>
    <tr>
      <th>10248841</th>
      <td>nm9993716</td>
      <td>Essias Loberg</td>
      <td>\N</td>
      <td>\N</td>
      <td>None</td>
      <td>\N</td>
    </tr>
    <tr>
      <th>10248842</th>
      <td>nm9993717</td>
      <td>Harikrishnan Rajan</td>
      <td>\N</td>
      <td>\N</td>
      <td>cinematographer</td>
      <td>tt8736744</td>
    </tr>
    <tr>
      <th>10248843</th>
      <td>nm9993718</td>
      <td>Aayush Nair</td>
      <td>\N</td>
      <td>\N</td>
      <td>cinematographer</td>
      <td>\N</td>
    </tr>
    <tr>
      <th>10248844</th>
      <td>nm9993719</td>
      <td>Andre Hill</td>
      <td>\N</td>
      <td>\N</td>
      <td>None</td>
      <td>\N</td>
    </tr>
  </tbody>
</table>
<p>10248845 rows × 6 columns</p>
</div>




```python
name_df = name_df.set_index('nconst')
name_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>primaryName</th>
      <th>birthYear</th>
      <th>deathYear</th>
      <th>primaryProfession</th>
      <th>knownForTitles</th>
    </tr>
    <tr>
      <th>nconst</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nm0000001</th>
      <td>Fred Astaire</td>
      <td>1899</td>
      <td>1987</td>
      <td>soundtrack,actor,miscellaneous</td>
      <td>tt0053137,tt0031983,tt0050419,tt0072308</td>
    </tr>
    <tr>
      <th>nm0000002</th>
      <td>Lauren Bacall</td>
      <td>1924</td>
      <td>2014</td>
      <td>actress,soundtrack</td>
      <td>tt0071877,tt0038355,tt0117057,tt0037382</td>
    </tr>
    <tr>
      <th>nm0000003</th>
      <td>Brigitte Bardot</td>
      <td>1934</td>
      <td>\N</td>
      <td>actress,soundtrack,music_department</td>
      <td>tt0054452,tt0057345,tt0059956,tt0049189</td>
    </tr>
    <tr>
      <th>nm0000004</th>
      <td>John Belushi</td>
      <td>1949</td>
      <td>1982</td>
      <td>actor,soundtrack,writer</td>
      <td>tt0078723,tt0080455,tt0077975,tt0072562</td>
    </tr>
    <tr>
      <th>nm0000005</th>
      <td>Ingmar Bergman</td>
      <td>1918</td>
      <td>2007</td>
      <td>writer,director,actor</td>
      <td>tt0060827,tt0050976,tt0083922,tt0050986</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>nm9993714</th>
      <td>Romeo del Rosario</td>
      <td>\N</td>
      <td>\N</td>
      <td>animation_department,art_department</td>
      <td>tt2455546</td>
    </tr>
    <tr>
      <th>nm9993716</th>
      <td>Essias Loberg</td>
      <td>\N</td>
      <td>\N</td>
      <td>None</td>
      <td>\N</td>
    </tr>
    <tr>
      <th>nm9993717</th>
      <td>Harikrishnan Rajan</td>
      <td>\N</td>
      <td>\N</td>
      <td>cinematographer</td>
      <td>tt8736744</td>
    </tr>
    <tr>
      <th>nm9993718</th>
      <td>Aayush Nair</td>
      <td>\N</td>
      <td>\N</td>
      <td>cinematographer</td>
      <td>\N</td>
    </tr>
    <tr>
      <th>nm9993719</th>
      <td>Andre Hill</td>
      <td>\N</td>
      <td>\N</td>
      <td>None</td>
      <td>\N</td>
    </tr>
  </tbody>
</table>
<p>10248845 rows × 5 columns</p>
</div>




```python
director_split_df = genre_df.copy(deep=True)
director_split_df['directors'] = director_split_df['directors'].map(lambda x: x.split(','))
director_split_df['director_count'] = director_split_df['directors'].map(lambda x: len(x))
```


```python
# How many directors would be ideal
fig, ax = plt.subplots(figsize=(12.5, 6.5))
sns.regplot("director_count", "perc", data=director_split_df, ax=ax)
ax.set_title("ROI % vs Quantity of Directors")
ax.title.set_fontsize(20)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
ax.xaxis.label.set_text("Director Count")
ax.yaxis.label.set_text("Return on Budget %")

# The graph is showing a downward trend the more directors you have
# So 1 director would be the best case.
```


![svg](img/output_20_0.svg)



```python
# Remove movies that have more than one director
director_split_df['director_count'] = director_split_df['director_count'].apply(lambda x: 1 if x == 1 else np.NaN)
director_split_df = director_split_df.dropna()
director_split_df = director_split_df.drop(columns=['director_count'])
director_split_df['directors'] =  director_split_df['directors'].apply(lambda x: x[0])
```


```python
# Get the top 200 directors that have made more than one movie
x_directors = director_split_df['directors'][0:200].value_counts().apply(lambda x: np.NaN if x < 2 else x).dropna()
x_directors
```




    nm1490123    6.0
    nm0796117    5.0
    nm0484907    4.0
    nm1443502    4.0
    nm0323239    3.0
                ... 
    nm0190859    2.0
    nm0619110    2.0
    nm1720541    2.0
    nm1980431    2.0
    nm0680846    2.0
    Name: directors, Length: 63, dtype: float64




```python
# Remove all but said top 200 from df
director_split_df['directors'] = director_split_df['directors'].apply(lambda x: x if x in x_directors else np.NaN)
director_split_df = director_split_df.dropna()
director_split_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>genres</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>rating</th>
      <th>perc</th>
      <th>runtimeMinutes</th>
      <th>directors</th>
      <th>writers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1288</th>
      <td>tt7668870</td>
      <td>Searching</td>
      <td>Thriller</td>
      <td>880000</td>
      <td>75462037</td>
      <td>PG-13</td>
      <td>84.752315</td>
      <td>102</td>
      <td>nm3792134</td>
      <td>nm3792134,nm3539578</td>
    </tr>
    <tr>
      <th>1288</th>
      <td>tt7668870</td>
      <td>Searching</td>
      <td>Mystery</td>
      <td>880000</td>
      <td>75462037</td>
      <td>PG-13</td>
      <td>84.752315</td>
      <td>102</td>
      <td>nm3792134</td>
      <td>nm3792134,nm3539578</td>
    </tr>
    <tr>
      <th>687</th>
      <td>tt1591095</td>
      <td>Insidious</td>
      <td>Thriller</td>
      <td>1500000</td>
      <td>99557032</td>
      <td>PG-13</td>
      <td>65.371355</td>
      <td>103</td>
      <td>nm1490123</td>
      <td>nm1191481</td>
    </tr>
    <tr>
      <th>687</th>
      <td>tt1591095</td>
      <td>Insidious</td>
      <td>Mystery</td>
      <td>1500000</td>
      <td>99557032</td>
      <td>PG-13</td>
      <td>65.371355</td>
      <td>103</td>
      <td>nm1490123</td>
      <td>nm1191481</td>
    </tr>
    <tr>
      <th>838</th>
      <td>tt3713166</td>
      <td>Unfriended</td>
      <td>Thriller</td>
      <td>1000000</td>
      <td>62882090</td>
      <td>R</td>
      <td>61.882090</td>
      <td>83</td>
      <td>nm0300174</td>
      <td>nm4532532</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>tt0409847</td>
      <td>Cowboys &amp; Aliens</td>
      <td>Sci-Fi</td>
      <td>163000000</td>
      <td>174822325</td>
      <td>PG-13</td>
      <td>0.072530</td>
      <td>119</td>
      <td>nm0269463</td>
      <td>nm0649460,nm0476064,nm0511541,nm1318843,nm1319...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>tt0409847</td>
      <td>Cowboys &amp; Aliens</td>
      <td>Thriller</td>
      <td>163000000</td>
      <td>174822325</td>
      <td>PG-13</td>
      <td>0.072530</td>
      <td>119</td>
      <td>nm0269463</td>
      <td>nm0649460,nm0476064,nm0511541,nm1318843,nm1319...</td>
    </tr>
    <tr>
      <th>967</th>
      <td>tt4463894</td>
      <td>Shaft</td>
      <td>Comedy</td>
      <td>35000000</td>
      <td>21360215</td>
      <td>R</td>
      <td>-0.389708</td>
      <td>111</td>
      <td>nm1103162</td>
      <td>nm0862781,nm1244069,nm1113415</td>
    </tr>
    <tr>
      <th>992</th>
      <td>tt4572792</td>
      <td>The Book of Henry</td>
      <td>Thriller</td>
      <td>10000000</td>
      <td>4596705</td>
      <td>PG-13</td>
      <td>-0.540330</td>
      <td>105</td>
      <td>nm1119880</td>
      <td>nm3884127</td>
    </tr>
    <tr>
      <th>865</th>
      <td>tt3813310</td>
      <td>Cop Car</td>
      <td>Thriller</td>
      <td>800000</td>
      <td>143658</td>
      <td>R</td>
      <td>-0.820427</td>
      <td>88</td>
      <td>nm1218281</td>
      <td>nm1218281,nm1755986</td>
    </tr>
  </tbody>
</table>
<p>202 rows × 10 columns</p>
</div>




```python
director_dict = {}
for director in director_split_df['directors'].unique():
    director_dict[director] = director_split_df.loc[director_split_df['directors'] == director]['perc'].sum()
director_df = pd.DataFrame(director_dict, index=[0])
director_df = director_df.transpose().reset_index()
director_df.columns=['nconst', 'value']
director_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nconst</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm3792134</td>
      <td>169.504630</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm1490123</td>
      <td>173.048808</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm0300174</td>
      <td>123.764180</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm1443502</td>
      <td>142.627735</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm0821844</td>
      <td>75.147252</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>58</th>
      <td>nm1139726</td>
      <td>9.988849</td>
    </tr>
    <tr>
      <th>59</th>
      <td>nm0000965</td>
      <td>14.664250</td>
    </tr>
    <tr>
      <th>60</th>
      <td>nm0000631</td>
      <td>17.501734</td>
    </tr>
    <tr>
      <th>61</th>
      <td>nm0000881</td>
      <td>9.526093</td>
    </tr>
    <tr>
      <th>62</th>
      <td>nm0005363</td>
      <td>21.266235</td>
    </tr>
  </tbody>
</table>
<p>63 rows × 2 columns</p>
</div>




```python
# Add in the real name of each director from name_basics based on nconst
director_df['name'] = director_df['nconst'].apply(lambda x: name_df.loc[x]['primaryName'])
director_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nconst</th>
      <th>value</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm3792134</td>
      <td>169.504630</td>
      <td>Aneesh Chaganty</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm1490123</td>
      <td>173.048808</td>
      <td>James Wan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm0300174</td>
      <td>123.764180</td>
      <td>Levan Gabriadze</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm1443502</td>
      <td>142.627735</td>
      <td>Jordan Peele</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm0821844</td>
      <td>75.147252</td>
      <td>Daniel Stamm</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>58</th>
      <td>nm1139726</td>
      <td>9.988849</td>
      <td>Neil Burger</td>
    </tr>
    <tr>
      <th>59</th>
      <td>nm0000965</td>
      <td>14.664250</td>
      <td>Danny Boyle</td>
    </tr>
    <tr>
      <th>60</th>
      <td>nm0000631</td>
      <td>17.501734</td>
      <td>Ridley Scott</td>
    </tr>
    <tr>
      <th>61</th>
      <td>nm0000881</td>
      <td>9.526093</td>
      <td>Michael Bay</td>
    </tr>
    <tr>
      <th>62</th>
      <td>nm0005363</td>
      <td>21.266235</td>
      <td>Guy Ritchie</td>
    </tr>
  </tbody>
</table>
<p>63 rows × 3 columns</p>
</div>




```python
# Which directors would be ideal to hire
fig, ax = plt.subplots(figsize=(12, 12))
sns.barplot('value', 'name', data=director_df.sort_values(by='value', ascending=False), ax=ax)
ax.set_title("Top Directors by Return on Budget for All Their Movies")
ax.title.set_fontsize(20)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
ax.xaxis.label.set_text("Return on Investment %")
ax.yaxis.label.set_text("Director Name")

# Here are the top 60 directors
```


![svg](img/output_26_0.svg)



```python
# How many writers would be ideal
writer_split_df = genre_df.copy(deep=True)
writer_split_df['writers'] = writer_split_df['writers'].map(lambda x: x.split(','))
writer_split_df['writer_count'] = writer_split_df['writers'].map(lambda x: len(x))

fig, ax = plt.subplots(figsize=(12.5, 6.5))
sns.regplot("writer_count", "perc", data=writer_split_df, ax=ax)
ax.set_title("ROI % vs Quantity of Writers")
ax.title.set_fontsize(20)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
ax.xaxis.label.set_text("Writer Count")
ax.yaxis.label.set_text("Return on Budget %")
# The graph is showing a downward trend the more writers you have
# So 1 writer would be the best case.
```


![svg](img/output_27_0.svg)



```python
# Parse and remove movies with more than one writer
writer_split_df['writer_count'] = writer_split_df['writer_count'].apply(lambda x: 1 if x == 1 else np.NaN)
writer_split_df = writer_split_df.dropna()
writer_split_df = writer_split_df.drop(columns=['writer_count'])
writer_split_df['writers'] =  writer_split_df['writers'].apply(lambda x: x[0])
# Create a list of writers that have written more than one movie
x_writers = writer_split_df['writers'][0:200].value_counts().apply(lambda x: np.NaN if x < 2 else x).dropna()
x_writers[0:10]
```




    nm0796117    6.0
    nm1191481    5.0
    nm1347153    5.0
    nm0000095    5.0
    nm1443502    4.0
    nm0191717    3.0
    nm0009190    3.0
    nm2752098    3.0
    nm2704527    3.0
    nm0831557    3.0
    Name: writers, dtype: float64




```python
# Get top movies with one writer who has written more than one movie
writer_split_df['writers'] = writer_split_df['writers'].apply(lambda x: x if x in x_writers else np.NaN)
writer_split_df = writer_split_df.dropna()
writer_split_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tconst</th>
      <th>primaryTitle</th>
      <th>genres</th>
      <th>budget</th>
      <th>ww_gross</th>
      <th>rating</th>
      <th>perc</th>
      <th>runtimeMinutes</th>
      <th>directors</th>
      <th>writers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>687</th>
      <td>tt1591095</td>
      <td>Insidious</td>
      <td>Thriller</td>
      <td>1500000</td>
      <td>99557032</td>
      <td>PG-13</td>
      <td>65.371355</td>
      <td>103</td>
      <td>nm1490123</td>
      <td>nm1191481</td>
    </tr>
    <tr>
      <th>687</th>
      <td>tt1591095</td>
      <td>Insidious</td>
      <td>Mystery</td>
      <td>1500000</td>
      <td>99557032</td>
      <td>PG-13</td>
      <td>65.371355</td>
      <td>103</td>
      <td>nm1490123</td>
      <td>nm1191481</td>
    </tr>
    <tr>
      <th>838</th>
      <td>tt3713166</td>
      <td>Unfriended</td>
      <td>Thriller</td>
      <td>1000000</td>
      <td>62882090</td>
      <td>R</td>
      <td>61.882090</td>
      <td>83</td>
      <td>nm0300174</td>
      <td>nm4532532</td>
    </tr>
    <tr>
      <th>838</th>
      <td>tt3713166</td>
      <td>Unfriended</td>
      <td>Mystery</td>
      <td>1000000</td>
      <td>62882090</td>
      <td>R</td>
      <td>61.882090</td>
      <td>83</td>
      <td>nm0300174</td>
      <td>nm4532532</td>
    </tr>
    <tr>
      <th>1074</th>
      <td>tt5052448</td>
      <td>Get Out</td>
      <td>Thriller</td>
      <td>4500000</td>
      <td>272495873</td>
      <td>R</td>
      <td>59.554638</td>
      <td>104</td>
      <td>nm1443502</td>
      <td>nm1443502</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>tt5610554</td>
      <td>Tully</td>
      <td>Comedy</td>
      <td>13000000</td>
      <td>15636462</td>
      <td>R</td>
      <td>0.202805</td>
      <td>95</td>
      <td>nm0718646</td>
      <td>nm1959505</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>tt5610554</td>
      <td>Tully</td>
      <td>Mystery</td>
      <td>13000000</td>
      <td>15636462</td>
      <td>R</td>
      <td>0.202805</td>
      <td>95</td>
      <td>nm0718646</td>
      <td>nm1959505</td>
    </tr>
    <tr>
      <th>1207</th>
      <td>tt6348138</td>
      <td>Missing Link</td>
      <td>Animation</td>
      <td>100000000</td>
      <td>26249469</td>
      <td>PG</td>
      <td>-0.737505</td>
      <td>93</td>
      <td>nm2752098</td>
      <td>nm2752098</td>
    </tr>
    <tr>
      <th>1207</th>
      <td>tt6348138</td>
      <td>Missing Link</td>
      <td>Adventure</td>
      <td>100000000</td>
      <td>26249469</td>
      <td>PG</td>
      <td>-0.737505</td>
      <td>93</td>
      <td>nm2752098</td>
      <td>nm2752098</td>
    </tr>
    <tr>
      <th>1207</th>
      <td>tt6348138</td>
      <td>Missing Link</td>
      <td>Comedy</td>
      <td>100000000</td>
      <td>26249469</td>
      <td>PG</td>
      <td>-0.737505</td>
      <td>93</td>
      <td>nm2752098</td>
      <td>nm2752098</td>
    </tr>
  </tbody>
</table>
<p>121 rows × 10 columns</p>
</div>




```python
writer_dict = {}
# Get the values for each writer, based on overall ROI %
for writer in writer_split_df['writers'].unique():
    writer_dict[writer] = writer_split_df.loc[writer_split_df['writers'] == writer]['perc'].sum()
writer_df = pd.DataFrame(writer_dict, index=[0])
writer_df = writer_df.transpose().reset_index()
writer_df.columns=['nconst', 'value']
writer_df[0:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nconst</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm1191481</td>
      <td>164.648328</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm4532532</td>
      <td>123.764180</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm1443502</td>
      <td>142.627735</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm0796117</td>
      <td>91.150728</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm1245146</td>
      <td>50.283028</td>
    </tr>
    <tr>
      <th>5</th>
      <td>nm2477891</td>
      <td>38.868785</td>
    </tr>
    <tr>
      <th>6</th>
      <td>nm0839812</td>
      <td>30.049974</td>
    </tr>
    <tr>
      <th>7</th>
      <td>nm3227090</td>
      <td>27.765748</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nm0868066</td>
      <td>25.190732</td>
    </tr>
    <tr>
      <th>9</th>
      <td>nm3398282</td>
      <td>22.442269</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add in the names of said writers from name_basics
writer_df['name'] = writer_df['nconst'].apply(lambda x: name_df.loc[x]['primaryName'])
writer_df[0:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nconst</th>
      <th>value</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm1191481</td>
      <td>164.648328</td>
      <td>Leigh Whannell</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm4532532</td>
      <td>123.764180</td>
      <td>Nelson Greaves</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm1443502</td>
      <td>142.627735</td>
      <td>Jordan Peele</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm0796117</td>
      <td>91.150728</td>
      <td>M. Night Shyamalan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm1245146</td>
      <td>50.283028</td>
      <td>Scott Lobdell</td>
    </tr>
    <tr>
      <th>5</th>
      <td>nm2477891</td>
      <td>38.868785</td>
      <td>Gary Dauberman</td>
    </tr>
    <tr>
      <th>6</th>
      <td>nm0839812</td>
      <td>30.049974</td>
      <td>Stephen Susco</td>
    </tr>
    <tr>
      <th>7</th>
      <td>nm3227090</td>
      <td>27.765748</td>
      <td>Damien Chazelle</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nm0868066</td>
      <td>25.190732</td>
      <td>Akira Toriyama</td>
    </tr>
    <tr>
      <th>9</th>
      <td>nm3398282</td>
      <td>22.442269</td>
      <td>Scotty Landes</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Which writers would be best
fig, ax = plt.subplots(figsize=(12, 12))
sns.barplot('value', 'name', data=writer_df.sort_values(by='value', ascending=False), ax=ax)
ax.set_title("Top Writers by Return on Budget for All Their Movies")
ax.title.set_fontsize(20)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
ax.xaxis.label.set_text("Return on Investment %")
ax.yaxis.label.set_text("Writer Name")

# Here are the top 60 writers
```


![svg](img/output_32_0.svg)



```python
# Top 10 writers
writer_df[0:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nconst</th>
      <th>value</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm1191481</td>
      <td>164.648328</td>
      <td>Leigh Whannell</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm4532532</td>
      <td>123.764180</td>
      <td>Nelson Greaves</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm1443502</td>
      <td>142.627735</td>
      <td>Jordan Peele</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm0796117</td>
      <td>91.150728</td>
      <td>M. Night Shyamalan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm1245146</td>
      <td>50.283028</td>
      <td>Scott Lobdell</td>
    </tr>
    <tr>
      <th>5</th>
      <td>nm2477891</td>
      <td>38.868785</td>
      <td>Gary Dauberman</td>
    </tr>
    <tr>
      <th>6</th>
      <td>nm0839812</td>
      <td>30.049974</td>
      <td>Stephen Susco</td>
    </tr>
    <tr>
      <th>7</th>
      <td>nm3227090</td>
      <td>27.765748</td>
      <td>Damien Chazelle</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nm0868066</td>
      <td>25.190732</td>
      <td>Akira Toriyama</td>
    </tr>
    <tr>
      <th>9</th>
      <td>nm3398282</td>
      <td>22.442269</td>
      <td>Scotty Landes</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Top 10 directors
director_df[0:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nconst</th>
      <th>value</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm3792134</td>
      <td>169.504630</td>
      <td>Aneesh Chaganty</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm1490123</td>
      <td>173.048808</td>
      <td>James Wan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm0300174</td>
      <td>123.764180</td>
      <td>Levan Gabriadze</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm1443502</td>
      <td>142.627735</td>
      <td>Jordan Peele</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm0821844</td>
      <td>75.147252</td>
      <td>Daniel Stamm</td>
    </tr>
    <tr>
      <th>5</th>
      <td>nm0796117</td>
      <td>91.150728</td>
      <td>M. Night Shyamalan</td>
    </tr>
    <tr>
      <th>6</th>
      <td>nm2497546</td>
      <td>73.569613</td>
      <td>David F. Sandberg</td>
    </tr>
    <tr>
      <th>7</th>
      <td>nm0905592</td>
      <td>38.927414</td>
      <td>Jeff Wadlow</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nm0484907</td>
      <td>62.638617</td>
      <td>Christopher Landon</td>
    </tr>
    <tr>
      <th>9</th>
      <td>nm0094435</td>
      <td>44.365252</td>
      <td>Bong Joon Ho</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Best movie would be worldwide
# directed and written by one of the top 10
# rated PG-13 or R
# either a comedy or a thriller
# being 90-110 minutes long
```
