import datetime
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
from sqlalchemy import create_engine
engine = create_engine("sqlite:///"+currentdir+"\\flerotron.db", echo=False)
from sqlalchemy import Column, UniqueConstraint, Integer, String, ForeignKey, Table, Date, MetaData, LargeBinary,Float,Boolean, asc, desc, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,relationship
import numpy

boolDT = numpy.int64
intDT = numpy.int64
floatDT = numpy.float32

#

def dt(a):
    if type(a)==float: return None
    elif type(a)==datetime.date: return a
    return datetime.date(int(a[0:4]),int(a[5:7]),int(a[8:10]))

def init():
    Session = sessionmaker(bind = engine)#,autoflush=False)
    global session,conn,Base
    Base = declarative_base()
    session = Session()
    conn = engine.connect()
    global boolDT,intDT,floatDT
    boolDT = numpy.int32
    intDT = numpy.int32
    floatDT = numpy.float32

def init2(Rty):
    global Reality
    #Rty = session.query(SimPoint).filter(SimPoint.N == 1).one()[0]
    Reality = Rty#conn.execute('SELECT id FROM Simpoint where N == 1').one()[0]

def tables():
    global PGNtoTournament,SetStructure,FirstSeeds,RankSeeds,SimEncounter,Elo,Tasks,EncounterRanks
##    PGNtoTournament = Table('PGNtoTournament',Base.metadata,
##                   Column('PGN',Integer,ForeignKey('set.id'), index=True),
##                   Column('Comp',Integer,ForeignKey('set.id')))
    SetStructure = Table('SetStructure',Base.metadata,
                   Column('parent',Integer,ForeignKey('set.id'), index=True),
                   Column('child',Integer,ForeignKey('encounter.id'), index=True),
                   Column('spot',Integer),
                   UniqueConstraint('parent', 'child'),
                   UniqueConstraint('parent', 'spot'))
    Tasks = Table('Tasks',Base.metadata,
                   Column('Point',Integer,ForeignKey('point.id')),
                   Column('Encounter',Integer,ForeignKey('encounter.id')),
                   Column('Task',String),
                   Column('Priority',Integer,index=True),
                   UniqueConstraint('Point', 'Encounter', 'Task'))
    FirstSeeds = Table('FirstSeeds',Base.metadata,
                   Column('encounter',Integer,ForeignKey('encounter.id'), index=True),
                   Column('spot',Integer),
                   Column('player',Integer,ForeignKey('player.id'), index=True),
                   UniqueConstraint('encounter', 'spot'))
    RankSeeds = Table('RankSeeds',Base.metadata,
                   Column('encounter',Integer,ForeignKey('encounter.id'), index=True),
                   Column('spot',Integer),
                   Column('rootencounter',Integer,ForeignKey('encounter.id'), index=True),
                   Column('rank',Integer),
                   UniqueConstraint('encounter', 'spot'))
    ##        self.OutcomeRanks = Table('OutcomeRanks',Base.metadata,                          CONTINUE HERE.
    ##                       Column('encounter',Integer,ForeignKey('encounter.id'), index=True),
    ##                       Column('simpoint',Integer,ForeignKey('simpoint.id')),
    ##                       Column('rank',Integer),
    ##                       Column('player',LargeBinary)
    SimEncounter = Table('SimEncounter',Base.metadata,
                  Column('SimPoint',Integer,ForeignKey('simpoint.id'), index=True),
                  Column('Encounter',Integer,ForeignKey('encounter.id'), index=True),
                  Column('Score',LargeBinary),
                  UniqueConstraint('SimPoint', 'Encounter'))
    SimEncounter.rel_root = relationship("SimPoint",back_populates = "rel_leaf_Enc",cascade = "all, delete, delete-orphan")
    EncounterRanks = Table('EncounterRanks',Base.metadata,
                  Column('SimPoint',Integer,ForeignKey('simpoint.id'), index=True),
                  Column('Encounter',Integer,ForeignKey('encounter.id'), index=True),
                  Column('Ranks',Integer),
                  Column('IDs',LargeBinary),
                  UniqueConstraint('SimPoint', 'Encounter','Ranks'))
    Elo = Table('Elo',Base.metadata,
                  Column('Point',Integer,ForeignKey('point.id'), index=True),
                  Column('PlayerID',Integer,ForeignKey('player.id'), index=True),
                  Column('MatrSeed',Integer),
                  Column('Categories',String),
                  Column('Bucket',Integer),
                  Column('Elo',Float),
                  Column('SimmedElo',LargeBinary),
                  UniqueConstraint('Point', 'MatrSeed'),
                  UniqueConstraint('Point', 'PlayerID', 'Categories'))
    Elo.rel_root = relationship("Point",back_populates = "rel_leaf_Elo",cascade = "all, delete, delete-orphan")
    session.commit()
    Base.metadata.create_all(engine)

def views():
    conn.execute('DROP VIEW IF EXISTS Gamedetail')
    #conn.execute('DROP VIEW IF EXISTS Gamedetai')
    #session.commit()
    #conn.execute('DELETE FROM Simencounter')
    #conn.execute('DELETE FROM EncounterRanks')
    conn.execute('CREATE VIEW IF NOT EXISTS Gamedetail AS \
                    SELECT Po.Date, E.id, En.Name, E.Startdate, E.Enddate, WE.Name, FW.spot, SE.Score \
                    FROM Encounter E \
                    JOIN SimEncounter SE ON SE.Encounter = E.ID \
                    JOIN Point Po ON SE.Simpoint = Po.id \
                    JOIN Entity En ON E.ID = En.ID \
                    JOIN FirstSeeds FW ON FW.encounter = E.id \
                    JOIN Player PW ON PW.id = FW.player \
                    JOIN Entity WE ON WE.ID = FW.player \
                    ORDER BY E.Startdate ASC, E.id ASC, SE.Simpoint ASC, FW.spot ASC')
    session.commit()
    Base.metadata.create_all(engine)

#init()
#views()