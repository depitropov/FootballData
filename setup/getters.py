import sqlite3 as sq
import pandas as pd
import numpy as np

conn = sq.connect('database.db')
matches = pd.read_sql_query('SELECT * FROM matches', conn)
conn.close()


def get_10last_home(date, name):
    """Get last 10 matches of home team."""
    last_home = matches.query('HomeTeam == "{0}" & Date < "{1}"'.format(name, date)). \
        sort_values(inplace=False, by='Date', ascending=False).head(10)
    if last_home.index.size == 10:
        last_home_ids = last_home.index.values.tolist().__str__().translate(None, ' []')
    else:
        last_home_ids = np.nan
    return last_home_ids


def get_10last_away(date, name):
    """Get last 10 matches of away team."""
    last_away = matches.query('AwayTeam == "{0}" & Date < "{1}"'.format(name, date)). \
        sort_values(inplace=False, by='Date', ascending=False).head(10)
    if last_away.index.size == 10:
        last_away_ids = last_away.index.values.tolist().__str__().translate(None, ' []')
    else:
        last_away_ids = np.nan
    return last_away_ids


def get_5last_home(date, name):
    """Get last 5 matches of home team."""
    last_home = matches.query('HomeTeam == "{0}" & Date < "{1}"'.format(name, date)). \
        sort_values(inplace=False, by='Date', ascending=False).head(5)
    if last_home.index.size == 5:
        last_home_ids = last_home.index.values.tolist().__str__().translate(None, ' []')
    else:
        last_home_ids = np.nan
    return last_home_ids


def get_5last_away(date, name):
    """Get last 5 matches of away team."""
    last_away = matches.query('AwayTeam == "{0}" & Date < "{1}"'.format(name, date)). \
        sort_values(inplace=False, by='Date', ascending=False).head(5)
    if last_away.index.size == 5:
        last_away_ids = last_away.index.values.tolist().__str__().translate(None, ' []')
    else:
        last_away_ids = np.nan
    return last_away_ids


def get_all_last_home(date, name):
    """Get list of 2 pd DataFrames: 1: all home matches of home_team 2: all away matches of away team"""
    last_matches = matches.query('HomeTeam == "{0}" & Date < "{1}"'.format(name, date))
    return last_matches


def get_all_last_away(date, name):
    """Get list of 2 pd DataFrames: 1: all home matches of home_team 2: all away matches of away team"""
    last_matches = matches.query('AwayTeam == "{0}" & Date < "{1}"'.format(name, date))
    return last_matches


def read_last(last_matches):
    """Get pd DataFrames of all matches in last_home_ids"""
    last_list = map(int, last_matches.split(","))
    return matches.loc[last_list, :]


def read_last_away(last_matches):
    """Get pd DataFrames of all matches in away_team_ids"""
    last_away_list = map(int, last_matches.split(","))
    return matches.loc[last_away_list, :]


def get_last_direct(date, teams):
    """Get last 5 direct matches between teams"""
    print date, teams
    last_direct = matches.query('HomeTeam == "{0}" & AwayTeam == "{1}" & Date < "{2}"'.format(teams[0], teams[1], date)). \
        sort_values(inplace=False, by='Date', ascending=False).head(5)
    if last_direct.index.size == 5:
        last_direct_ids = last_direct.index.values.tolist().__str__().translate(None, ' []')
    else:
        last_direct_ids = np.nan
    return last_direct_ids
