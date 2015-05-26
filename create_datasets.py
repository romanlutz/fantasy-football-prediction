# Copyright (c) Roman Lutz. All rights reserved.
# The use and distribution terms for this software are covered by the
# Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
# which can be found in the file LICENSE.md at the root of this distribution.
# By using this software in any fashion, you are agreeing to be bound by
# the terms of this license.
# You must not remove this notice, or any other, from this software.

from get_data import fetch_defense_stats, fetch_qb_stats, test_players, determine_team
import numpy as np

""" Returns stats of the player or team corresponding to the id
for the last game before the given week in the given year
"""
def last_game(statistics, id, year, week):
    # if the week was the first, go back by one week
    week -= 1
    if week == 0:
        week = 17
        year -= 1
    # check if there are previous years
    if year < 2009 or year > 2014:
        return None, None, None

    # check if the team/player played in the given week
    # if not played, recursively call previous week to check
    if not(statistics[id][str(year)][str(week)]['played']):
        return last_game(statistics, id, year, week)
    # team/player played in the given week, return stats
    else:
        return statistics[id][str(year)][str(week)], year, week

""" Returns the statistics of the last k games for the given
player or team corresponding to the id. If there are less than
k games, only the existing ones are returned.
"""
def last_k_games(k, statistics, id, year, week):
    stats, year, week = last_game(statistics, id, year, week)
    last_k = [stats]
    k -= 1
    # case 1: no prior games
    if stats == None:
        return []
    # case 2: only one game requested
    if k == 0:
        return last_k

    # case 3: multiple games requested
    # repeatedly check if there are further games
    while last_k[-1] != None and k > 0:
        stats, year, week = last_game(statistics, id, year, week)
        last_k.append(stats)
        k -= 1
    # if None appears in the list, remove it
    return last_k[:-1] if last_k[-1] == None else last_k

""" Calculates the average stats of a defense over the games
handed over to the function.
"""
def average_defense_stats(games):
    points = 0
    passing_yards = 0
    rushing_yards = 0
    turnovers = 0

    n_games = len(games)

    if n_games == 0:
        return None

    for game in games:
        points += game['points_allowed']
        passing_yards += game['passing_yards_allowed']
        rushing_yards += game['rushing_yards_allowed']
        turnovers += game['turnovers']

    return {
        'points_allowed': float(points)/float(n_games),
        'passing_yards_allowed': float(passing_yards)/float(n_games),
        'rushing_yards_allowed': float(rushing_yards)/float(n_games),
        'turnovers': float(turnovers)/float(n_games),
    }

""" Calculates the average QB stats over the games
handed over to the function.
"""
def average_qb_stats(games):
    passing_attempts = 0
    passing_yards = 0
    passing_touchdowns = 0
    passing_interceptions = 0
    passing_two_point_attempts = 0
    passing_two_point_made = 0
    rushing_attempts = 0
    rushing_yards = 0
    rushing_touchdowns = 0
    rushing_two_point_attempts = 0
    rushing_two_point_made = 0
    fumbles = 0

    n_games = len(games)

    if n_games == 0:
        return None

    for game in games:
        passing_attempts += game['passing_attempts']
        passing_yards += game['passing_yards']
        passing_touchdowns += game['passing_touchdowns']
        passing_interceptions += game ['passing_interceptions']
        passing_two_point_attempts += game['passing_two_point_attempts']
        passing_two_point_made += game['passing_two_point_made']
        rushing_attempts += game['rushing_attempts']
        rushing_yards += game['rushing_yards']
        rushing_touchdowns += game['rushing_touchdowns']
        rushing_two_point_attempts += game['rushing_two_point_attempts']
        rushing_two_point_made += game['rushing_two_point_made']
        fumbles += game['fumbles']

    return {
        'passing_attempts': float(passing_attempts)/float(n_games),
        'passing_yards': float(passing_yards)/float(n_games),
        'passing_touchdowns': float(passing_touchdowns)/float(n_games),
        'passing_interceptions': float(passing_interceptions)/float(n_games),
        'passing_two_point_attempts': float(passing_two_point_attempts)/float(n_games),
        'passing_two_point_made': float(passing_two_point_made)/float(n_games),
        'rushing_attempts': float(rushing_attempts)/float(n_games),
        'rushing_yards': float(rushing_yards)/float(n_games),
        'rushing_touchdowns': float(rushing_touchdowns)/float(n_games),
        'rushing_two_point_attempts': float(rushing_two_point_attempts)/float(n_games),
        'rushing_two_point_made': float(rushing_two_point_made)/float(n_games),
        'fumbles': float(fumbles)/float(n_games)
    }

""" Calculates the age of a player for a given game.
"""
def calculate_age(birthdate, game_week, game_year):
    if birthdate[1] == '/':
        birthdate = '0' + birthdate
    birth_month = int(birthdate[0:2])
    if birthdate[4] == '/':
        birthdate = birthdate[:3] + '0' + birthdate[3:]
    birth_day = int(birthdate[3:5])
    birth_year = int(birthdate[6:10])
    total_days = 1 + (game_week - 1) * 7
    game_month = 9 + int(total_days / 31)
    game_day = total_days % 31
    age = game_year - birth_year
    age += float(game_month - birth_month) / 12
    age += float(game_day - birth_day) / 365
    return age

""" Creates a row for the dataset. It is assumed 
that the player with the given ID actually played 
in the given week and year.
"""
def create_row(qb_statistics, defense_statistics, rookie_statistics, id, year, week):
    age = calculate_age(qb_statistics[id]['birthdate'], week, year)
    years_pro = qb_statistics[id]['years_pro'] - (2015 - year)
    last_game_qb_stats = average_qb_stats(last_k_games(1, qb_statistics, id, year, week))
    if last_game_qb_stats == None:
        # replace last_game_stats with rookie stats
        last_game_qb_stats = rookie_statistics

    last_10_games_qb_stats = average_qb_stats(last_k_games(10, qb_statistics, id, year, week))
    if last_10_games_qb_stats == None:
        # replace last_10_games with rookie stats
        last_10_games_qb_stats = rookie_statistics

    # find out the opposing team by determining which team the QB plays for
    # the API does only allow to query the current team (as of 2015)
    # therefore this has to be done differently
    qb_team = determine_team(qb_statistics[id][str(year)])
    # if QB had multiple teams in the given year, don't include QB stats
    if qb_team == None:
        return None

    home_team = qb_statistics[id][str(year)][str(week)]['home']
    away_team = qb_statistics[id][str(year)][str(week)]['away']
    # take other team as opponent
    opponent = home_team if away_team == qb_team else away_team

    # the defense stats should only be used based on some data
    # it cannot be substituted by 'rookie' stats
    last_game_defense_stats = average_defense_stats(last_k_games(1, defense_statistics, opponent, year, week))
    last_10_games_defense_stats = average_defense_stats(last_k_games(10, defense_statistics, opponent, year, week))

    # row consists of
    # 0: QB id
    # 1: QB name
    # 2: QB age
    # 3: QB years pro
    # 4-15: last game QB stats
    # 16-27: last 10 games QB stats
    # 28-31: last game defense stats
    # 32-35: last 10 games defense stats
    # 36: actual fantasy score = target
    return [id,
            qb_statistics[id]['name'],
            age,
            years_pro,
            last_game_qb_stats['passing_attempts'],
            last_game_qb_stats['passing_yards'],
            last_game_qb_stats['passing_touchdowns'],
            last_game_qb_stats['passing_interceptions'],
            last_game_qb_stats['passing_two_point_attempts'],
            last_game_qb_stats['passing_two_point_made'],
            last_game_qb_stats['rushing_attempts'],
            last_game_qb_stats['rushing_yards'],
            last_game_qb_stats['rushing_touchdowns'],
            last_game_qb_stats['rushing_two_point_attempts'],
            last_game_qb_stats['rushing_two_point_made'],
            last_game_qb_stats['fumbles'],
            last_10_games_qb_stats['passing_attempts'],
            last_10_games_qb_stats['passing_yards'],
            last_10_games_qb_stats['passing_touchdowns'],
            last_10_games_qb_stats['passing_interceptions'],
            last_10_games_qb_stats['passing_two_point_attempts'],
            last_10_games_qb_stats['passing_two_point_made'],
            last_10_games_qb_stats['rushing_attempts'],
            last_10_games_qb_stats['rushing_yards'],
            last_10_games_qb_stats['rushing_touchdowns'],
            last_10_games_qb_stats['rushing_two_point_attempts'],
            last_10_games_qb_stats['rushing_two_point_made'],
            last_10_games_qb_stats['fumbles'],
            last_game_defense_stats['points_allowed'],
            last_game_defense_stats['passing_yards_allowed'],
            last_game_defense_stats['rushing_yards_allowed'],
            last_game_defense_stats['turnovers'],
            last_10_games_defense_stats['points_allowed'],
            last_10_games_defense_stats['passing_yards_allowed'],
            last_10_games_defense_stats['rushing_yards_allowed'],
            last_10_games_defense_stats['turnovers'],
            fantasy_score(qb_statistics[id][str(year)][str(week)]['passing_yards'],
            qb_statistics[id][str(year)][str(week)]['passing_touchdowns'],
            qb_statistics[id][str(year)][str(week)]['passing_interceptions'],
            qb_statistics[id][str(year)][str(week)]['rushing_yards'],
            qb_statistics[id][str(year)][str(week)]['rushing_touchdowns'],
            qb_statistics[id][str(year)][str(week)]['fumbles'],
            qb_statistics[id][str(year)][str(week)]['rushing_two_point_made'] + qb_statistics[id][str(year)][str(week)]['passing_two_point_made'])
            ]

""" Calculate the fantasy score based on NFL standard rules.
"""
def fantasy_score(passing_yards, passing_touchdowns, interceptions, rushing_yards, rushing_touchdowns, fumbles, two_point):
    return float(passing_yards) / 25 + passing_touchdowns * 4.0 - interceptions * 2.0 + float(rushing_yards) / 10 + rushing_touchdowns * 6.0 - fumbles * 2.0 + two_point * 2

""" Calculate the average stats of all Rookie QBs in the
observed years.
"""
def rookie_qb_average(qb_statistics):
    games = []
    for qb in qb_statistics.keys():
        rookie_year = 2015 - qb_statistics[qb]['years_pro'] + 1
        if rookie_year >= 2009:
            # some wrong NFL data has 2014 rookies labelled as 2015 rookies
            if rookie_year > 2014:
                rookie_year = 2014
            # data on rookie_year is available
            for week in range(1, 18):
                if qb_statistics[qb][str(rookie_year)][str(week)]['played']:
                    games.append(qb_statistics[qb][str(rookie_year)][str(week)])
    return average_qb_stats(games)

""" Create the dataset by determining all rows
"""
def create_all_rows(qb_statistics, defense_statistics, start_year, end_year):
    rows = []
    rookie_qb_stats = rookie_qb_average(qb_statistics)
    # year 2009 should be left such that some previous defense data is available
    for year in range(start_year, end_year):
        for week in range(1, 18):
            for qb in qb_statistics.keys():
                if qb_statistics[qb][str(year)][str(week)]['played']:
                    row = create_row(qb_statistics, defense_statistics, rookie_qb_stats, qb, year, week)
                    # if stats are inconclusive, don't use them
                    # for more information see create_row documentation
                    if row != None:
                        rows.append(row)
    return rows

# save data sets to files
np.save('train.npy', np.array(create_all_rows(fetch_qb_stats(), fetch_defense_stats(), 2010, 2014)))
np.save('test.npy', np.array(create_all_rows(fetch_qb_stats(), fetch_defense_stats(), 2014, 2015)))



