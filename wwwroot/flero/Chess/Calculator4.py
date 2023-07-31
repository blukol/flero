import datetime
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import configglobal as db
db.init()
import configchess as dbc
import oo2chess as oo2
db.tables()
dbc.tables()
#import cProfile
#import re


##import sqlite3
##from sqlalchemy import Column, Integer, Unicode, UnicodeText, String
##from sqlalchemy import create_engine
##from sqlalchemy.orm import sessionmaker
##engine = create_engine("sqlite+pysqlite:///:memory:", echo=True, future=True)
##Session = sessionmaker(bind=engine)
##s = Session()
CW = oo2.ChessWorld()
db.Base.metadata.create_all(db.engine)
db.session.commit()
CW.Chessrun(3**7,datetime.date(2022,4,27),datetime.date(2022,4,27),5.0)
db.session.commit()
db.session.close()
#CW.Chessrun(3**9,datetime.date(2022,2,12),datetime.date(2022,4,21),6.0)
#cProfile.run('CW.Chessrun(3**10,datetime.date(2022,3,12),datetime.date(2022,4,12),10.0)',sort='cumtime')
