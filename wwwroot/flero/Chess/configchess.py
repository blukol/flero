import datetime
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
from sqlalchemy import create_engine
engine = create_engine("sqlite:///"+currentdir+"\\flerotron.db", echo=False)
from sqlalchemy import Column, Index, UniqueConstraint, Integer, String, ForeignKey, Table, Date, MetaData, LargeBinary,Float,Boolean, asc, desc, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,relationship
import numpy
import configglobal as db
    
def tables():
    global PGN
    PGN = Table('PGN',db.Base.metadata,
                   Column('format',String),
                   Column('tournament',String, index=True),
                   Column('filename',String),
                   Column('Event',String, nullable=False),
                   Column('Site',String, nullable=False),
                   Column('Date',String, nullable=False),
                   Column('Round',String, nullable=False),
                   Column('White',String, nullable=False),
                   Column('Black',String, nullable=False),
                   Column('Result',String, nullable=False),
                   Column('Moves',String),
                   Column('Board',String),
                   Column('WhiteFideId',String),
                   Column('BlackFideId',String),
                   Column('WhiteTitle',String),
                   Column('BlackTitle',String),
                   Column('WhiteElo',String),
                   Column('BlackElo',String),
                   Column('WhiteCountry',String),
                   Column('BlackCountry',String),
                   Column('WhiteEloChange',String),
                   Column('BlackEloChange',String),
                   Column('Status',String, index=True),
                   Column('ChessGameID',Integer,ForeignKey("chessgame.id"), index=True),
                   Index('filename','Event','Site','Date','Round','White','Black'),
                   UniqueConstraint('filename','Event','Site','Date','Round','White','Black')) #to allow file to be moved and moves/result to be updated
    db.session.commit()
    db.Base.metadata.create_all(db.engine)
