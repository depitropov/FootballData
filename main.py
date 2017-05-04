import sqlite3 as sq
import pandas as pd
import setup

if __name__ == '__main__':
    #setup.init_db()
    setup.set_additional_data()
    setup.init_numpy_db()
