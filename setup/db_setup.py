#! /usr/bin/env python
import pandas as pd
import sqlite3 as sq
from os import listdir
from multiprocessing import Pool

import getters
from converters import *

p = Pool(7)


def init_db():
    """Initiates the database. Transform and populate data from all CVS located in input folder"""
    csv_files = listdir('input')
    conn = sq.connect('database.db')
    cur = conn.cursor()
    cur.execute('DROP TABLE IF EXISTS matches')
    cur.close()
    conn.commit()
    conn.close()

    for csv_file in csv_files:
        print csv_file
        temp_frame = pd.read_csv('input/{0}'.format(csv_file))
        temp_frame.dropna(how='all', inplace=True)  # Remove empty rows
        temp_frame.dropna(axis=1, how='all', inplace=True)  # Remove empty columns
        temp_frame.dropna(subset=['HTR', 'FTR'], inplace=True)  # Remove matches without half time or full time results
        temp_frame['league'] = temp_frame['Div']  # Create new column for league name
        temp_frame['country'] = temp_frame['Div']  # Create new column for country name
        temp_frame.Date = p.map(convert_date, temp_frame.Date)
        temp_frame.country = p.map(country, temp_frame.country)
        temp_frame.league = p.map(league, temp_frame.league)
        temp_frame.HTR = p.map(h_d_a, temp_frame.HTR)
        temp_frame.FTR = p.map(h_d_a, temp_frame.FTR)
        temp_frame['HTFTR'] = p.map(int, temp_frame['HTR'].map(str) + temp_frame['FTR'].map(str))
        temp_frame.drop(['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'Div', 'BWH', 'BWD',
                         'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH',
                         'VCD', 'VCA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5',
                         'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbAH', 'BbAHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA',
                         'PSCH', 'PSCD', 'PSCA', 'BSH', 'BSD', 'BSA', 'Referee', 'GBH', 'GBA', 'GBD', 'SBH', 'SBD', 'SBA',
                         'SJH', 'SJD', 'SJA'], axis=1, inplace=True, errors='ignore')
        temp_frame.replace("", np.nan)
        conn = sq.connect('database.db')
        temp_frame.to_sql('matches', conn, if_exists='append', index=False)
        conn.commit()
        conn.close()


def set_additional_data():
    """Populate additional data to existing DB. Adds last matches and ids for country, league and teams"""
    conn = sq.connect('database.db')
    matches = pd.read_sql_query('SELECT * FROM matches', conn)
    matches['country_id'] = pd.Categorical((pd.factorize(matches.country)[0] + 1))
    matches['league_id'] = pd.Categorical((pd.factorize(matches.league)[0] + 1))
    matches['home_id'] = pd.Categorical((pd.factorize(matches.HomeTeam)[0] + 1))
    unique_teams = matches[['HomeTeam', 'home_id']].drop_duplicates()
    unique_teams.columns = ['AwayTeam', 'away']
    unique_teams.to_sql('teams', conn, index=False, if_exists='replace')
    matches = matches.merge(unique_teams, left_on='AwayTeam', right_on='AwayTeam')
    matches = matches.rename(columns={'away': 'away_id'})
    matches['last_home_ids_10'] = p.map(getters.get_10last_home, matches['Date'], matches['HomeTeam'])
    matches['last_away_ids_10'] = p.map(getters.get_10last_away, matches['Date'], matches['AwayTeam'])
    matches['last_home_ids_5'] = p.map(getters.get_5last_home, matches['Date'], matches['HomeTeam'])
    matches['last_away_ids_5'] = p.map(getters.get_5last_away, matches['Date'], matches['AwayTeam'])
    matches['last_direct'] = p.map(getters.get_last_direct, matches['Date'],
                                 zip(matches['HomeTeam'], matches['AwayTeam']))
    matches.to_sql('matches', conn, if_exists='replace')
    conn.close()


if __name__ == '__main__':
    pass