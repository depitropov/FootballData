#! /usr/bin/env python
import sqlite3 as sq
import pandas as pd
import numpy as np
#from main import matches
from getters import read_last, get_all_last_home, get_all_last_away
from algorithms import *
from multiprocessing import Pool
import itertools
p = Pool(7)

def init_numpy_db():
    """Get Pandas dataframe with raw matches. 
    Get statistics and write only needed columns to numpy.db for ML algorithms"""
    conn = sq.connect('database.db')
    cur = conn.cursor()
    matches = pd.read_sql_query('SELECT * FROM matches', conn, index_col='index')
    matches_np = matches[['HTFTR', 'HomeTeam', 'AwayTeam', 'league_id', 'country_id', 'Date', 'B365H', 'B365A', 'B365D', 'last_home_ids_10', 'last_away_ids_10',
                          'last_home_ids_5', 'last_away_ids_5']].dropna()
    for num_matches in [5, 10]:
        for side in ['home', 'away']:
            for algorithm in algorithms:
                matches_np['{0}_{1}'.format(side, algorithm.func_name)] \
                    = p.map(algorithm, p.map(read_last, matches_np['last_{0}_ids_{1}'.format(side, num_matches)]),
                          itertools.repeat(side, len(matches_np)))
                for side in [get_all_last_home, get_all_last_away]:
                    if side == get_all_last_home:
                        side_name = 'home'
                        side_col = 'HomeTeam'
                    elif side == get_all_last_away:
                        side_name = 'away'
                        side_col = 'AwayTeam'
                    matches_np['{0}_{1}_all'.format(side_name, algorithm.func_name)] = \
                        p.map(algorithm,
                            p.map(side, matches_np['Date'], matches_np[side_col]), itertools.repeat(side_name, len(matches_np)))
        fin_matches = matches_np.drop(['Date', 'HomeTeam', 'AwayTeam', 'last_home_ids_10', 'last_away_ids_10',
                                       'last_home_ids_5', 'last_away_ids_5'], axis=1)
        cur.execute('DROP TABLE IF EXISTS numpy_{0}'.format(num_matches))
        conn.commit()
        fin_matches.to_sql('numpy_{0}'.format(num_matches), conn, if_exists='replace')
    cur.close()
    conn.close()


if __name__ == '__main__':
    init_numpy_db()