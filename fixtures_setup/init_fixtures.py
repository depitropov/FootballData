#! /usr/bin/env python
import pandas as pd
import sqlite3 as sq


#import setup.getters
from setup.converters import *
import setup.getters
from setup.algorithms import *
import itertools


def read_fixtures():
    fixtures_frame = pd.read_csv('../fixtures/fixtures.csv')
    fixtures_frame.dropna(how='all', inplace=True)  # Remove empty rows
    fixtures_frame.dropna(axis=1, how='all', inplace=True)  # Remove empty columns
    fixtures_frame['league'] = fixtures_frame['Div']  # Create new column for league name
    fixtures_frame['country'] = fixtures_frame['Div']  # Create new column for country name
    fixtures_frame.Date = map(convert_date, fixtures_frame.Date)
    fixtures_frame.country = map(country, fixtures_frame.country)
    fixtures_frame.league = map(league, fixtures_frame.league)
    fixtures_frame.drop(['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'Div', 'BWH', 'BWD',
                     'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH',
                     'VCD', 'VCA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbMx>2.5',
                     'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'BbAH', 'BbAHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA',
                     'PSCH', 'PSCD', 'PSCA', 'BSH', 'BSD', 'BSA', 'Referee', 'GBH', 'GBA', 'GBD', 'SBH', 'SBD', 'SBA',
                     'SJH', 'SJD', 'SJA'], axis=1, inplace=True, errors='ignore')
    fixtures_frame.replace("", np.nan)
    return fixtures_frame


def set_additional_data():
    fixtures = read_fixtures()
    fixtures['last_home_ids_10'] = map(setup.getters.get_10last_home, fixtures['Date'], fixtures['HomeTeam'])
    fixtures['last_away_ids_10'] = map(setup.getters.get_10last_away, fixtures['Date'], fixtures['AwayTeam'])
    fixtures['last_home_ids_5'] = map(setup.getters.get_5last_home, fixtures['Date'], fixtures['HomeTeam'])
    fixtures['last_away_ids_5'] = map(setup.getters.get_5last_away, fixtures['Date'], fixtures['AwayTeam'])
    return fixtures


def fixtures_np():
    """Get Pandas dataframe with raw matches. 
    Get statistics and write only needed columns to numpy.db for ML algorithms"""
    fixtures = set_additional_data()
    fixtures_np = fixtures[
        ['HomeTeam', 'AwayTeam', 'Date', 'B365H', 'B365A', 'B365D',
         'last_home_ids_10', 'last_away_ids_10',
         'last_home_ids_5', 'last_away_ids_5']].dropna()
    for num_fixtures in [5]:
        for side in ['home', 'away']:
            for algorithm in algorithms:
                fixtures_np['{0}_{1}'.format(side, algorithm.func_name)] \
                    = map(algorithm, map(setup.getters.read_last, fixtures_np['last_{0}_ids_{1}'.format(side, num_fixtures)]),
                          itertools.repeat(side, len(fixtures_np)))

        fin_fixtures = fixtures_np.drop(['Date', 'HomeTeam', 'AwayTeam', 'last_home_ids_10', 'last_away_ids_10',
                                       'last_home_ids_5', 'last_away_ids_5'], axis=1)
    return fin_fixtures

if __name__ == '__main__':
    fixtures_np()
