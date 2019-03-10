import requests
import pandas as pd
import re
import datetime as dt
from bs4 import BeautifulSoup as bs

# comprehend list for years
years = [str(2000 + i) for i in range(5,19)]
this_year = '2019'
# print(years)

# multiple functions for cleaning data
# regex for finding round names
pattern = re.compile("^(Round|Week|Semifinal|Final|Qualifiers|Semis)(\ \d{1,2})?.*$")

def parse_date(date):
    date = dt.datetime.strptime(date, '%d %b %Y')
    return date

def outcome(f):
    '''game outcome for home team V: victory L: loss D: draw'''
    if f > 0:
        return 'V'
    elif f < 0:
        return 'L'
    elif f == 0:
        return 'D'
    else:
        return 'D'

def fix_round(f):
    '''extract round number or final type'''
    if f[:4] == 'Week':
        return f[5:7]
    elif f[:5] == 'Round':
        return f[6:8]
    elif f[:10] == 'Qualifiers' or f[:13] == 'Quarterfinals':
        return 'QF' # quarter final
    elif f[:6] == 'Finals' or f == 'Semifinals' or f == 'Semis' or f == 'Semifinal':
        return 'SF' # semi final
    elif f[:6] == 'Final ' or f == 'Final':
        return 'GF' # grand final
    else:
        return f.strip()
    
def team_loc(team):
    au = ['Brumbies','Rebels','Reds','Sunwolves','Waratahs','Western Force']
    nz = ['Blues','Chiefs','Crusaders','Highlanders','Hurricanes']
    sa = ['Bulls','Jaguares','Lions','Sharks','Stormers','Cheetahs','Kings']
    x = ''
    if any(team in s for s in au):
        x = 'au'
    if any(team in s for s in nz):
        x = 'nz'
    if any(team in s for s in sa):
        x = 'sa'
    return x

def name_fixer(name):
    if 'High' in name:
        name = 'Highlanders'
    if 'Wara' in name:
        name = 'Waratahs'
    if 'Storm' in name:
        name = 'Stormers'
    if 'Sunw' in name:
        name = 'Sunwolves'
    if 'Force' in name:
        name = 'Western Force'    
    if 'Hurr' in name:
        name = 'Hurricanes'     
    if 'Warra' in name:
        name = 'Waratahs'  
    if 'Cheet' in name:
        name = 'Cheetahs'    
    if 'Cats' in name:
        name = 'Lions' # Team name change in 2005    
    return name

def data_nice(year):
    table_nice = []
    table_round = []
    with open('data/data_' + year + '.txt') as f:
        data = bs(f.read(), features='lxml')
    rows = data.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols_nice = [ele.text.strip() for ele in cols]
        cols_round = [x.text.strip() for x in cols if pattern.match(x.text.strip())]
        table_nice.append([ele for ele in cols_nice if ele]) # Get rid of empty values
        table_round.append([ele for ele in cols_round if ele]) # Get rid of empty values
    df1 = pd.DataFrame(table_nice)
    df2 = pd.DataFrame(table_round).fillna(method='ffill')
    df = pd.concat([df1, df2], axis=1).dropna()
    df['year'] = year
    df.columns = ['date','teams','location','time','score','round','year']
    df['date'] = df['date'] + ' ' + df['year']
    df['home'] = df['teams'].str.split(' v ').str[0]
    df['away'] = df['teams'].str.split(' v ').str[1]
    df['home'] = df['home'].str.strip()
    df['away'] = df['away'].str.strip()
    df['home'] = [name_fixer(str(x)) for x in df['home']]
    df['away'] = [name_fixer(str(x)) for x in df['away']]
    df['home_loc'] = [team_loc(str(x)) for x in df['home']]
    df['away_loc'] = [team_loc(str(x)) for x in df['away']]
    df['fthp'] = df['score'].str.split('-').str[0].astype('int') # full time home points
    df['ftap'] = df['score'].str.split('-').str[1].astype('int') # full time away points
    df['ftr'] = [outcome(x) for x in df['fthp'] - df['ftap']] # home outcome ftr (full time result)
    df['round'] = [fix_round(x) for x in df['round']]
    remove_columns = ['teams','score','year','location','time']
    df = df.drop(columns=remove_columns)
    return df  

    # creating dataframes, cleaning up data:

df_2005 = data_nice('2005')
df_2006 = data_nice('2006')
df_2007 = data_nice('2007')
df_2008 = data_nice('2008')
df_2009 = data_nice('2009')
df_2010 = data_nice('2010')
df_2011 = data_nice('2011')
df_2012 = data_nice('2012')
df_2013 = data_nice('2013')
df_2014 = data_nice('2014')
df_2015 = data_nice('2015')
df_2016 = data_nice('2016')
df_2017 = data_nice('2017')
df_2018 = data_nice('2018')
df_2019 = data_nice('2019')

print(df_2010.columns)

# more fixing data inconsistancies
df_2005.loc[(df_2005['date'] == '28 May 2005'), 'round'] = "GF" # 2005 no final fixed
df_2006.drop(5, inplace=True) # remove bogus final data from 2006
df_2018.drop(10, inplace=True) # remove bogus final data from 2018
# List of series missing from each year

'''
au = ['Brumbies','Rebels','Reds','Sunwolves','Waratahs','Western Force']
nz = ['Blues','Chiefs','Crusaders','Highlanders','Hurricanes']
sa = ['Bulls','Jaguares','Lions','Sharks','Stormers','Cheetahs','Kings']
'''

missing_games_2007 = [pd.Series(['12 May 2007', 'SF', 'Sharks', 'Blues', 'sa', 'nz', 34, 18, 'V'], index=df_2007.columns ) ,
                      pd.Series(['12 May 2007', 'SF', 'Bulls', 'Crusaders', 'sa', 'nz', 27, 12, 'V'], index=df_2007.columns )]

missing_games_2008 = [pd.Series(['31 May 2008', 'GF', 'Crusaders', 'Waratahs', 'nz','au', 20, 12, 'V'], index=df_2008.columns ) ,
                      pd.Series(['24 May 2008', 'SF', 'Waratahs', 'Sharks', 'au' ,'sa', 28, 13, 'V'], index=df_2008.columns ),
                      pd.Series(['24 May 2008', 'SF', 'Crusaders', 'Hurricanes', 'nz','nz',33, 22, 'V'], index=df_2008.columns )]

missing_games_2017 = [pd.Series(['21 Jul 2017', 'QF', 'Brumbies', 'Hurricanes', 'au','nz',16, 35, 'L'], index=df_2017.columns ) ,
                      pd.Series(['22 Jul 2017', 'QF', 'Crusaders', 'Highlanders', 'nz','nz',17, 0, 'V'], index=df_2017.columns ),
                      pd.Series(['23 Jul 2017', 'QF', 'Lions', 'Sharks', 'sa','sa', 23, 21, 'V'], index=df_2017.columns ),
                      pd.Series(['23 Jul 2017', 'QF', 'Stormers', 'Chiefs', 'sa','nz',11, 17, 'L'], index=df_2017.columns )]

# Pass a list of series to the append() to add multiple rows to 2007
df_2007 = df_2007.append(missing_games_2007 , ignore_index=True)
df_2008 = df_2008.append(missing_games_2008 , ignore_index=True)
df_2017 = df_2017.append(missing_games_2017 , ignore_index=True)


df_2009.at[6, 'home'] = 'Chiefs'
df_2009.at[7, 'home'] = 'Bulls'
df_2009.at[8, 'home'] = 'Bulls'

df_2010.at[4, 'home'] = 'Bulls'

df_2013.at[2, 'home'] = 'Crusaders'
df_2013.at[3, 'home'] = 'Brumbies'
df_2013.at[4, 'home'] = 'Chiefs'
df_2013.at[5, 'home'] = 'Bulls'
df_2013.at[6, 'home'] = 'Chiefs'

df_2014.at[6, 'round'] = 'GF'
df_2014.at[2, 'round'] = 'QF'
df_2014.at[3, 'round'] = 'QF'

df_2015.at[2, 'round'] = 'QF'
df_2015.at[3, 'round'] = 'QF'
df_2015.at[6, 'round'] = 'GF'

df_2016.at[2, 'round'] = 'QF'
df_2016.at[3, 'round'] = 'QF'
df_2016.at[4, 'round'] = 'QF'
df_2016.at[5, 'round'] = 'QF'
df_2016.at[8, 'round'] = 'GF'

df_2016.at[6, 'home'] = 'Crusaders'
df_2016.at[7, 'home'] = 'Lions'

df_2018.at[152, 'round'] = 'QF'
df_2018.at[153, 'round'] = 'QF'
df_2018.at[154, 'round'] = 'QF'
df_2018.at[155, 'round'] = 'QF'

# parse dates and sort, reset indexes
df_2005.date = df_2005.date.apply(parse_date)
df_2006.date = df_2006.date.apply(parse_date)
df_2007.date = df_2007.date.apply(parse_date)
df_2008.date = df_2008.date.apply(parse_date)
df_2009.date = df_2009.date.apply(parse_date)
df_2010.date = df_2010.date.apply(parse_date)
df_2011.date = df_2011.date.apply(parse_date)
df_2012.date = df_2012.date.apply(parse_date)
df_2013.date = df_2013.date.apply(parse_date)
df_2014.date = df_2014.date.apply(parse_date)
df_2015.date = df_2015.date.apply(parse_date)
df_2016.date = df_2016.date.apply(parse_date)
df_2017.date = df_2017.date.apply(parse_date)
df_2018.date = df_2018.date.apply(parse_date)

# reset indexes
df_2005 = df_2005.sort_values(by=['date']).reset_index(drop=True)
df_2006 = df_2006.sort_values(by=['date']).reset_index(drop=True)
df_2007 = df_2007.sort_values(by=['date']).reset_index(drop=True)
df_2008 = df_2008.sort_values(by=['date']).reset_index(drop=True)
df_2009 = df_2009.sort_values(by=['date']).reset_index(drop=True)
df_2010 = df_2010.sort_values(by=['date']).reset_index(drop=True)
df_2011 = df_2011.sort_values(by=['date']).reset_index(drop=True)
df_2012 = df_2012.sort_values(by=['date']).reset_index(drop=True)
df_2013 = df_2013.sort_values(by=['date']).reset_index(drop=True)
df_2014 = df_2014.sort_values(by=['date']).reset_index(drop=True)
df_2015 = df_2015.sort_values(by=['date']).reset_index(drop=True)
df_2016 = df_2016.sort_values(by=['date']).reset_index(drop=True)
df_2017 = df_2017.sort_values(by=['date']).reset_index(drop=True)
df_2018 = df_2018.sort_values(by=['date']).reset_index(drop=True)

# get running sum of points and points conceded by round for home and away teams
# need to be up to that point/game (hence minus x:)
def get_cum_points(df):    
    # home team points scored htps
    df['htps'] = df.groupby(['home'])['fthp'].apply(lambda x: x.cumsum() - x) 
    # home team points conceded htpc
    df['htpc'] = df.groupby(['home'])['ftap'].apply(lambda x: x.cumsum() - x)
    # away team points scored atps
    df['atps'] = df.groupby(['away'])['ftap'].apply(lambda x: x.cumsum() - x)
    # away team points conceded atpc
    df['atpc'] = df.groupby(['away'])['fthp'].apply(lambda x: x.cumsum() - x)
    return df

# Apply to each dataset
df_2005 = get_cum_points(df_2005)
df_2006 = get_cum_points(df_2006)
df_2007 = get_cum_points(df_2007)
df_2008 = get_cum_points(df_2008)
df_2009 = get_cum_points(df_2009)
df_2010 = get_cum_points(df_2010)
df_2011 = get_cum_points(df_2011)
df_2012 = get_cum_points(df_2012)
df_2013 = get_cum_points(df_2013)
df_2014 = get_cum_points(df_2014)
df_2015 = get_cum_points(df_2015)
df_2016 = get_cum_points(df_2016)
df_2017 = get_cum_points(df_2017)
df_2018 = get_cum_points(df_2018)

def get_home_points(ftr):
    '''The most common bonus point system is:
    4 points for winning a match.
    2 points for drawing a match.
    0 points for losing a match.
    1 losing bonus point for losing by 7 points (or fewer)
    1 try bonus point for scoring (at least) 3 tries more than the opponent.'''
    points = 0
    if ftr == 'V':
        points += 4
    elif ftr == 'D':
        points += 2
    else:
        points += 0
    return points

def get_away_points(ftr):
    if ftr == 'V':
        return 0
    elif ftr == 'D':
        return 2
    else:
        return 4

def get_cumcomp_points(df):   
    df['homepoint'] = [get_home_points(x) for x in df['ftr']]
    df['awaypoint'] = [get_away_points(x) for x in df['ftr']]
    df['htp'] = df.groupby(['home'])['homepoint'].apply(lambda x: x.cumsum() - x) 
    df['atp'] = df.groupby(['away'])['awaypoint'].apply(lambda x: x.cumsum() - x)
    remove_columns = ['homepoint','awaypoint']
    df = df.drop(columns=remove_columns)
    return df

df_2005 = get_cumcomp_points(df_2005)
df_2006 = get_cumcomp_points(df_2006)
df_2007 = get_cumcomp_points(df_2007)
df_2008 = get_cumcomp_points(df_2008)
df_2009 = get_cumcomp_points(df_2009)
df_2010 = get_cumcomp_points(df_2010)
df_2011 = get_cumcomp_points(df_2011)
df_2012 = get_cumcomp_points(df_2012)
df_2013 = get_cumcomp_points(df_2013)
df_2014 = get_cumcomp_points(df_2014)
df_2015 = get_cumcomp_points(df_2015)
df_2016 = get_cumcomp_points(df_2016)
df_2017 = get_cumcomp_points(df_2017)
df_2018 = get_cumcomp_points(df_2018)

def opp_res(x):
    if x == 'V':
        return 'L'
    elif x == 'L':
        return 'V'
    else:
        return 'D'

def get_form(df):
    ''' gets last game result for last 5 games'''
    # home form
    df['hm1'] = df.groupby(['home'])['ftr'].shift(1).fillna('M')
    df['hm2'] = df.groupby(['home'])['ftr'].shift(2).fillna('M')
    df['hm3'] = df.groupby(['home'])['ftr'].shift(3).fillna('M')
    df['hm4'] = df.groupby(['home'])['ftr'].shift(4).fillna('M')
    df['hm5'] = df.groupby(['home'])['ftr'].shift(5).fillna('M')
    # away form need to reverse result to get opposit...
    df['am1'] = df.groupby(['away'])['ftr'].shift(1).fillna('M')
    df['am2'] = df.groupby(['away'])['ftr'].shift(2).fillna('M')
    df['am3'] = df.groupby(['away'])['ftr'].shift(3).fillna('M')
    df['am4'] = df.groupby(['away'])['ftr'].shift(4).fillna('M')
    df['am5'] = df.groupby(['away'])['ftr'].shift(5).fillna('M')
    return df

# apply each dataset the form
df_2005 = get_form(df_2005)
df_2006 = get_form(df_2006)
df_2007 = get_form(df_2007)
df_2008 = get_form(df_2008)
df_2009 = get_form(df_2009)
df_2010 = get_form(df_2010)
df_2011 = get_form(df_2011)
df_2012 = get_form(df_2012)
df_2013 = get_form(df_2013)
df_2014 = get_form(df_2014)
df_2015 = get_form(df_2015)
df_2016 = get_form(df_2016)
df_2017 = get_form(df_2017)
df_2018 = get_form(df_2018)


# get round numbers for all rounds including quarters, semis, and finals
df_2005['rn'] = df_2005.groupby([True]*len(df_2005))['round'].transform(lambda x: pd.factorize(x)[0]+1)
df_2006['rn'] = df_2006.groupby([True]*len(df_2006))['round'].transform(lambda x: pd.factorize(x)[0]+1)
df_2007['rn'] = df_2007.groupby([True]*len(df_2007))['round'].transform(lambda x: pd.factorize(x)[0]+1)
df_2008['rn'] = df_2008.groupby([True]*len(df_2008))['round'].transform(lambda x: pd.factorize(x)[0]+1)
df_2009['rn'] = df_2009.groupby([True]*len(df_2009))['round'].transform(lambda x: pd.factorize(x)[0]+1)
df_2010['rn'] = df_2010.groupby([True]*len(df_2010))['round'].transform(lambda x: pd.factorize(x)[0]+1)
df_2011['rn'] = df_2011.groupby([True]*len(df_2011))['round'].transform(lambda x: pd.factorize(x)[0]+1)
df_2012['rn'] = df_2012.groupby([True]*len(df_2012))['round'].transform(lambda x: pd.factorize(x)[0]+1)
df_2013['rn'] = df_2013.groupby([True]*len(df_2013))['round'].transform(lambda x: pd.factorize(x)[0]+1)
df_2014['rn'] = df_2014.groupby([True]*len(df_2014))['round'].transform(lambda x: pd.factorize(x)[0]+1)
df_2015['rn'] = df_2015.groupby([True]*len(df_2015))['round'].transform(lambda x: pd.factorize(x)[0]+1)
df_2016['rn'] = df_2016.groupby([True]*len(df_2016))['round'].transform(lambda x: pd.factorize(x)[0]+1)
df_2017['rn'] = df_2017.groupby([True]*len(df_2017))['round'].transform(lambda x: pd.factorize(x)[0]+1)
df_2018['rn'] = df_2018.groupby([True]*len(df_2018))['round'].transform(lambda x: pd.factorize(x)[0]+1)



# display(df_2005[['ftr','hm1','hm2','am1','am2','home','away']].loc[df_2005['home'] == 'Blues'])

playing_stat = pd.concat([df_2005,df_2006,df_2007,df_2008,df_2009,df_2010,df_2011,df_2012,df_2013
                          ,df_2014,df_2015,df_2016,df_2017,df_2018],                        
                         ignore_index=True)

# Gets the form points.
def get_form_points(string):
    sum = 0
    for letter in string:
        sum += get_home_points(letter)
    return sum

playing_stat['htformptsstr'] = playing_stat['hm1'] + playing_stat['hm2'] + playing_stat['hm3'] + playing_stat['hm4'] + playing_stat['hm5']
playing_stat['atformptsstr'] = playing_stat['am1'] + playing_stat['am2'] + playing_stat['am3'] + playing_stat['am4'] + playing_stat['am5']

playing_stat['htformpts'] = playing_stat['htformptsstr'].apply(get_form_points)
playing_stat['atformpts'] = playing_stat['atformptsstr'].apply(get_form_points)

# Identify Win/Loss Streaks if any.
def get_3game_ws(string):
    if string[-3:] == 'VVV':
        return 1
    else:
        return 0
    
def get_5game_ws(string):
    if string == 'VVVVV':
        return 1
    else:
        return 0
    
def get_3game_ls(string):
    if string[-3:] == 'LLL':
        return 1
    else:
        return 0
    
def get_5game_ls(string):
    if string == 'LLLLL':
        return 1
    else:
        return 0
    
playing_stat['HTWinStreak3'] = playing_stat['htformptsstr'].apply(get_3game_ws)
playing_stat['HTWinStreak5'] = playing_stat['htformptsstr'].apply(get_5game_ws)
playing_stat['HTLossStreak3'] = playing_stat['htformptsstr'].apply(get_3game_ls)
playing_stat['HTLossStreak5'] = playing_stat['htformptsstr'].apply(get_5game_ls)

playing_stat['ATWinStreak3'] = playing_stat['atformptsstr'].apply(get_3game_ws)
playing_stat['ATWinStreak5'] = playing_stat['atformptsstr'].apply(get_5game_ws)
playing_stat['ATLossStreak3'] = playing_stat['atformptsstr'].apply(get_3game_ls)
playing_stat['ATLossStreak5'] = playing_stat['atformptsstr'].apply(get_5game_ls)

# Get Goal Difference
playing_stat['htgd'] = playing_stat['htps'] - playing_stat['htpc']
playing_stat['atgd'] = playing_stat['atps'] - playing_stat['atpc']

# Diff in points
playing_stat['diffpts'] = playing_stat['htp'] - playing_stat['atp']
playing_stat['diffformpts'] = playing_stat['htformpts'] - playing_stat['atformpts']

# Scale DiffPts , DiffFormPts, HTGD, ATGD by Matchweek.
cols = ['htgd','atgd','diffpts','diffformpts','htp','atp']
playing_stat.rn = playing_stat.rn.astype(float)

for col in cols:
    playing_stat[col] = playing_stat[col] / playing_stat.rn

def only_hw(string):
    if string == 'V':
        return 'H'
    else:
        return 'NH'
    
playing_stat['ftr'] = playing_stat['ftr'].apply(only_hw)  


playing_stat['homeaway'] = playing_stat['home'] + playing_stat['away']
playing_stat['won_last_home'] = [1 if x == 'H' else 0 for x in playing_stat.groupby(['homeaway'])['ftr'].shift(1)]

playing_stat['awayhome'] = playing_stat['away'] + playing_stat['home']
playing_stat['won_last_away'] = [1 if x == 'NH' else 0 for x in playing_stat.groupby(['awayhome'])['ftr'].shift(1)]


print(playing_stat.loc[playing_stat['homeaway'] == 'CrusadersLions'])
print(playing_stat.loc[playing_stat['awayhome'] == 'CrusadersLions'])


'''
todo

features:
one hot encode team names
did home team rank higher than away team last season
did home team win against away team last time they played
offense rating (based on wins and points)
offense rating (based on wins and points)

GridSearch to find parameters
'''


playing_stat.to_csv("data/final_dataset.csv")