# Copyright (c) Roman Lutz. All rights reserved.
# The use and distribution terms for this software are covered by the
# Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
# which can be found in the file LICENSE.md at the root of this distribution.
# By using this software in any fashion, you are agreeing to be bound by
# the terms of this license.
# You must not remove this notice, or any other, from this software.

import nflgame

""" create dictionary consisting of all games in the used
year. All these games have a single attribute 'played'
set to False.
"""
def create_empty_entry():
    dict = {}
    for year in range(2009, 2015):
        dict[str(year)] = {}
        for week in range(1, 18):
            dict[str(year)][str(week)] = {'played': False}
    return dict

""" Returns a dictionary with the name, birthdate
and the number of years the player has spent as a
professional player.
"""
def get_static_data(id):
    player = nflgame.players[id]
    return {'name': player.full_name,
            'birthdate': player.birthdate,
            'years_pro': player.years_pro}


""" Checks if player had a single team in one season and
return team, if multiple teams: return None
"""
def determine_team(year_data):
    teams = {}
    games = 0
    for week in year_data.keys():
        if year_data[week]['played']:
            games += 1
            if year_data[week]['home'] in teams.keys():
                teams[year_data[week]['home']] += 1
            else:
                teams[year_data[week]['home']] = 1
            if year_data[week]['away'] in teams.keys():
                teams[year_data[week]['away']] += 1
            else:
                teams[year_data[week]['away']] = 1
    for team in teams.keys():
        # if one team occurs in every game, return game
        if teams[team] == games:
            return team
    # no team occurs in every game
    return None

""" Gets all QB statistics in a single dictionary.
The keys are the player names, the value for each player
is a dictionary with all his game statistics.
"""
def fetch_qb_stats():
    # statistics is a dictionary of all player stats
    # the keys are player names, the values are lists
    # each list contains dictionaries that contain single game stats
    statistics = {}
    teams = map(lambda x: x[0], nflgame.teams)
    for year in range(2009, 2015):
        for week in range(1, 18):
            games = nflgame.games(year=year, week=week)
            for index, game in enumerate(games):
                players = nflgame.combine([games[index]])
                # every player with at least 5 passing attempts
                # less than five is not taken into account
                for player in filter(lambda player: player.passing_att >= 5, players.passing()):
                    # if player has not been saved before create entry
                    if not(player.playerid in statistics.keys()):
                        statistics[player.playerid] = create_empty_entry()
                        statistics[player.playerid].update(get_static_data(id = player.playerid))
                    # save data in dictionary
                    statistics[player.playerid][str(year)][str(week)]= {
                        'home': game.home,
                        'away': game.away,
                        'passing_attempts': player.passing_att,
                        'passing_yards': player.passing_yds,
                        'passing_touchdowns': player.passing_tds,
                        'passing_interceptions': player.passing_ints,
                        'passing_two_point_attempts': player.passing_twopta,
                        'passing_two_point_made': player.passing_twoptm,
                        'rushing_attempts': player.rushing_att,
                        'rushing_yards': player.rushing_yds,
                        'rushing_touchdowns': player.rushing_tds,
                        'rushing_two_point_attempts': player.rushing_twopta,
                        'rushing_two_point_made': player.rushing_twoptm,
                        'fumbles': player.fumbles_tot,
                        'played': True
                        }
    return statistics

# the test players were selected based on their stats
test_players = {
    '00-0029263': 'Russell Wilson',
    '00-0023459': 'Aaron Rodgers',
    '00-0026143': 'Matt Ryan',
    '00-0020531': 'Drew Brees',
    '00-0026158': 'Joe Flacco',
    '00-0027973': 'Andy Dalton',
    '00-0024226': 'Jay Cutler',
    '00-0023436': 'Alex Smith',
    '00-0029701': 'Ryan Tannehill',
    '00-0019596': 'Tom Brady',
    '00-0031280': 'Derek Carr',
    '00-0022924': 'Ben Roethlisberger',
    '00-0026625': 'Brian Hoyer',
    '00-0021678': 'Tony Romo',
    '00-0027974': 'Colin Kaepernick',
    '00-0010346': 'Peyton Manning',
    '00-0029668': 'Andrew Luck',
    '00-0026498': 'Matthew Stafford',
    '00-0022803': 'Eli Manning',
    '00-0022942': 'Philip Rivers',
    '00-0027939': 'Cam Newton',
    '00-0031237': 'Teddy Bridgewater',
    '00-0031407': 'Blake Bortles',
    '00-0023541': 'Kyle Orton'
}

""" Get the game statistics of all 32 defenses of all
games in the observed time. The dictionary is indexed
by the team's abbreviation, e.g. ATL for Atlanta.
"""
def fetch_defense_stats():
    # team defense statistics
    statistics = {}
    for team in map(lambda x: x[0], nflgame.teams):
        statistics[team] = create_empty_entry()

    for year in range(2009, 2015):
        for week in range(1, 18):
            for game in nflgame.games(year=year, week=week):
                home = game.home
                away = game.away
                statistics[home][str(year)][str(week)] = {
                    'home': home,
                    'away': away,
                    'points_allowed': game.score_away,
                    'passing_yards_allowed': game.stats_away[2],
                    'rushing_yards_allowed': game.stats_away[3],
                    'turnovers': game.stats_away[6],
                    'played': True

                }
                statistics[away][str(year)][str(week)] = {
                    'home': home,
                    'away': away,
                    'points_allowed': game.score_home,
                    'passing_yards_allowed': game.stats_home[2],
                    'rushing_yards_allowed': game.stats_home[3],
                    'turnovers': game.stats_home[6],
                    'played': True
                }

    return statistics



