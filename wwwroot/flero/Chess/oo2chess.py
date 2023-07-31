import os, sys
import io
import datetime
import numpy
import pandas
#import PGNExtractorToSQL
import math
import configglobal as db
import Chess.configchess as dbc
import chess
import chess.pgn
from sqlalchemy import Column, Integer, String, ForeignKey, Table, UniqueConstraint, and_, insert, LargeBinary, Float, Boolean
from sqlalchemy.orm import aliased


boolDT = db.boolDT
intDT = db.intDT
floatDT = db.floatDT

def byAlpha(x):
    return str(x)

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

#import oo3Klappt as oo3
import oo3
#import viz
#import PGNExtractor

Restable = {'1-0':1.0,'1/2-1/2':0.5,'0-1':0.0,'*':None}

def ddt(a):
    return datetime.date(int(a[0:4]),int(a[5:7]),int(a[8:10]))

def dt(a):
    if type(a)==float: return None
    try:
        return datetime.date(int(a[0:4]),int(a[5:7]),int(a[8:10]))
    except:
        return None

def dt2(a):
    if type(a)==float: return None,None
    try:
        if len(a) == 10:
            if a[5:7] == '??':
                return datetime.date(int(a[0:4]),1,1),datetime.date(int(a[0:4]),12,31)
            elif a[8:10] == '??':
                return datetime.date(int(a[0:4]),int(a[5:7]),1),datetime.date(int(a[0:4]),int(a[5:7]),28)
            else:
                return datetime.date(int(a[0:4]),int(a[5:7]),int(a[8:10])),datetime.date(int(a[0:4]),int(a[5:7]),int(a[8:10]))
        else:
            print(a)
            return None,None
    except:
        print(a)
        return None,None
    
class ChessWorld(oo3.World):
    def __init__(self):
        oo3.World.__init__(self,'Chess')
        Table('chessplayermap',db.Base.metadata,
                       Column('name',String, index=True),
                       Column('country',String),
                       Column('event',Integer),
                       Column('playerid',Integer,ForeignKey('chessplayer.id'), index=True),
                       UniqueConstraint('name','event'))
        #oo3.World.Tables(self)
        self.DrawishR = 0.5
        self.currentdir = currentdir

##    def AddToPlayermap():
##
##    def CreateNewPlayer():
##
##    def CheckPlayer(Name,Event,Date,OneG):
##        #Same FideID in playermap? take most recent existing ID
##        CPM = Base.metadata.tables['chessplayermap']
##        if 'FideID' in OneG:
##            Gefunden = self.session.query(CPM).filter(CPM.FideID = OneG['FideID']).order_by(LastDate.desc()).one()
##            print(Gefunden)
##            PlID = Gefunden.PlayerID
##        else:
##            Gefunden = self.session.query(CPM).filter(CPM.Name = Name).order_by(LastDate.desc()).one()
##            
##        Same Name in playermap? yes: take existing ID, no: create new player
##        Same Name/Tournament not in playermap? add name/tournament
##        else

    def CheckEvent(self,ev,SD,ED,fo):
        #CPM = self.Base.metadata.tables['chessplayermap']
        Outp = db.session.query(ChessSet).filter(ChessSet.Name == ev).all()#.order_by(ChessSet.LastDate.desc()).one()
        if Outp == []:
            N = ChessSet('PGN',{},SD,ED,ev,self,fo,ev)
            db.session.add(N)
            db.session.commit()
            return(N)
        else:
            #print(Outp[0])
            #print(type(SD))
            #print(type(Outp[0].StartDate),Outp[0].StartDate)
            Outp[0].StartDate = min(SD,dt(Outp[0].StartDate))
            Outp[0].EndDate = max(ED,dt(Outp[0].EndDate))
            db.session.commit()
            return(Outp[0])

    def CheckPlayer(self,Name,FideID,Country,Event):
        CPM = db.Base.metadata.tables['chessplayermap']
        #Gefunden = self.session.query(CPM).filter(CPM.FideID = OneG['FideID']).order_by(LastDate.desc()).all()       
        #Outp = self.session.query(ChessPlayer).filter(ChessPlayer.Name == pl).all()#.order_by(ChessSet.LastDate.desc()).one()
        if FideID != None:
            Outp = db.session.query(ChessPlayer).filter(ChessPlayer.FideID == FideID).all()
            if Outp == []:
                Gefunden = db.session.query(CPM).filter(CPM.c.name == Name).all()
                if Gefunden == []:
                    N = ChessPlayer(['',Name,'',Country,'','',FideID])
                    db.session.add(N)
                    return(N)             
                else:
                    Outp2 = db.session.query(ChessPlayer).filter(ChessPlayer.id == Gefunden[0].playerid).all()
                    Outp2[0].FideID = FideID
                    return(Outp2[0])
            else:
                return(Outp[0])
        else:
            Gefunden = db.session.query(CPM).filter(CPM.c.name == Name).all()
            if Gefunden == []:
                N = ChessPlayer(['',Name,'',Country,'','',''])
                db.session.add(N)
                db.session.commit()
                PM = CPM.insert().values(name=Name, country=Country, event=Event.id ,playerid=N.id)
                #print('PM',PM)
                db.conn.execute(PM)
                return(N)             
            else:
                Gefunden2 = db.session.query(CPM).filter(CPM.c.name == Name).filter(CPM.c.event == Event.id).all()
                #print(Gefunden[0].event)
                #print(Event.id)
                if Gefunden2 == []:#[0].event != Event.id:
                    PM = CPM.insert().values(name=Name, country=Country, event=Event.id ,playerid=Gefunden[0].playerid)
                    db.conn.execute(PM)
                #print(Gefunden[0])
                Outp = db.session.query(ChessPlayer).filter(ChessPlayer.id == Gefunden[0].playerid).all()
                return(Outp[0])

    #def AddSingleGametoDB(self, game, filename2, tformat, tournament2, NT, SeedsOnly):
    #OneG = dict(game.headers)


    def MovesandTC(self, game):
        Moves = bytearray()
        board = game.board()
        try:
            for move in game.mainline_moves():
                legal_moves = list(board.legal_moves)
                legal_moves.sort(key=byAlpha, reverse=False)
                Moves.append(legal_moves.index(move))
                board.push(move)
        except:
            print('Moves error', OneG)
        TC = None
        try:
            m = 0
            firsttime = []
            for node in game.mainline():
                firsttime.append(float(node.clock()))
                m += 1
                if m == 2:
                    break
            # print(firsttime)
            if firsttime[0] < 300.1 and firsttime[0] - firsttime[1] > 45.0:
                TC = 'Armageddon 5&4'
            elif max(firsttime) < 306.5:
                TC = '5+3'
            elif max(firsttime) < 610.5:
                TC = '10+5'
            elif max(firsttime) < 905.5:
                TC = '15+5'
            elif max(firsttime) < 920.5:
                TC = '15+10'
            elif max(firsttime) < 1510.5:
                TC = '25+5'
            elif max(firsttime) < 1520.5:
                TC = '25+10'
            else:
                TC = 'Classical'
        except:
            TC = 'Unknown'
        return Moves,TC

    def CreatePGN(self,row):
        Text = ''
        OneG = row._mapping
        for r in OneG.keys():
            if r == 'Moves': continue
            Text+='['+r+' "'+str(OneG[r])+'"]\n'
        Text+='\n'+OneG['Moves']+'\n'
        return Text

    def CheckTournaments(self,Tournaments):
        for t in Tournaments:
            NTs = db.session.query(ChessSet).filter(ChessSet.Name == t[1],ChessSet.Type == t[0]).all()
            if len(NTs) == 1:
                NT = NTs[0]
            else: #Sea4rch for Seeds.
                Players = db.conn.execute('SELECT distinct F.player FROM PGN P \
                                                JOIN FirstSeeds F ON P.ChessGameID = F.encounter \
                                              WHERE P.format = "' + t[0] + '" AND P.tournament = "'+t[1]+'"').all()
                Seeding = {}
                for p in range(len(Players)):
                    Pl = db.session.query(ChessPlayer).filter(ChessPlayer.id==int(Players[p][0])).one()
                    Seeding[p] = [None,'First',Pl]
                NT = ChessSet(t[0], Seeding, db.dt(t[2]), db.dt(t[3]),t[1],self,t[0],t[1])
                db.session.add(NT)
            #return NT

#
    def CheckGames(self): #WHERE Status IN (\'New\',\'Updated\') \
        #Tournaments1 = db.session.query(dbc.PGN).filter(dbc.PGN.c.tournament=='Goldmoney Asian Rapid').all()
        #Tournaments = db.session.query(dbc.PGN).filter(dbc.PGN.c.Status == 'Updated').group_by(dbc.PGN.c.format,dbc.PGN.c.tournament).all()

        Tournaments = db.conn.execute('SELECT format,tournament,MIN(DATE),MAX(DATE) FROM PGN \
                                    WHERE Status IN ("Updated","New") AND format != "Other" \
                                    GROUP BY format,tournament').all()
        Games = db.session.query(dbc.PGN).filter(dbc.PGN.c.Status == 'New').all()
        for g in Games:
            OneG = g._mapping
            NTs = db.session.query(ChessSet).filter(ChessSet.Name==g[1],ChessSet.Type==g[0]).all()
            if len(NTs) == 0:
                NT = None
            else:
                NT = NTs[1]
            GID = self.CheckGame(OneG,g,NT)
            Qu = dbc.PGN.update().where(dbc.PGN.c.filename == OneG['filename'],
                                            dbc.PGN.c.Event == OneG['Event'],
                                            dbc.PGN.c.Site == OneG['Site'],
                                            dbc.PGN.c.Date == OneG['Date'],
                                            dbc.PGN.c.Round == OneG['Round'],
                                            dbc.PGN.c.White == OneG['White'],
                                            dbc.PGN.c.Black == OneG['Black']).values(Status='Done',
                                                                                     ChessGameID=str(GID))
            db.conn.execute(Qu)
        Games = db.session.query(dbc.PGN).filter(dbc.PGN.c.Status == 'Updated').all()
        for g in Games:
            OneG = g._mapping
            if 'ChessGameID' in OneG.keys():
                if OneG['ChessGameID'] == None:
                    continue
                if OneG['ChessGameID'] == 'None':
                    continue
                G = db.session.query(ChessGame).filter(ChessGame.id == OneG['ChessGameID']).one()
                G.Result = Restable[OneG['Result']]
                PGN1 = self.CreatePGN(g)
                #print(PGN1)
                game = chess.pgn.read_game(io.StringIO(PGN1))
                G.Moves,G.TimeControl = self.MovesandTC(game)
            db.session.commit()
            Qu = dbc.PGN.update().where(dbc.PGN.c.filename == OneG['filename'],
                                            dbc.PGN.c.Event == OneG['Event'],
                                            dbc.PGN.c.Site == OneG['Site'],
                                            dbc.PGN.c.Date == OneG['Date'],
                                            dbc.PGN.c.Round == OneG['Round'],
                                            dbc.PGN.c.White == OneG['White'],
                                            dbc.PGN.c.Black == OneG['Black']).values(Status='Done')
            db.conn.execute(Qu)
        return Tournaments

    def CheckGame(self,OneG,row,NT):
        if OneG['White'] in ['bye', '?', '', ' '] or OneG['Black'] in ['bye', '?', '', ' ']:
            return None
        SD, ED = dt2(OneG['Date'])
        if None in [SD, ED]:
            return None
        IDE = self.CheckEvent(OneG['Event'], SD, ED, OneG['format'])
        ####db.PGNtoTournament.insert().values(IDE,)
        WFID, BFID, WC, BC = None,None,None,None
        if 'WhiteFideId' in OneG.keys():
            WFID = OneG['WhiteFideId']
        if 'BlackFideId' in OneG.keys():
            BFID = OneG['BlackFideId']
        if 'WhiteCountry' in OneG.keys():
            WC = OneG['WhiteCountry']
        if 'BlackCountry' in OneG.keys():
            BC = OneG['BlackCountry']
        IDW = self.CheckPlayer(OneG['White'], WFID, WC, IDE)
        IDB = self.CheckPlayer(OneG['Black'], BFID, BC, IDE)
        #Moves = bytearray()
        #TC = None
        PGN1 = self.CreatePGN(row)
        game = chess.pgn.read_game(io.StringIO(PGN1))
        Moves,TC = self.MovesandTC(game)
        R = OneG['Round']
        GSeedsW = aliased(db.FirstSeeds)
        GSeedsB = aliased(db.FirstSeeds)
        #print('IDW',IDW)
        Outp = db.session.query(ChessGame).\
               join(GSeedsW,GSeedsW.c.encounter == ChessGame.id).\
               join(GSeedsB,GSeedsB.c.encounter == ChessGame.id).\
               filter(GSeedsW.c.spot == 0).\
               filter(GSeedsB.c.spot == 1).\
               filter(IDW.id == GSeedsW.c.player).\
               filter(IDB.id == GSeedsB.c.player).\
               filter(ChessGame.StartDate <= ED).\
               filter(ChessGame.EndDate >= SD).\
               filter(ChessGame.Round == R).\
               all()#.order_by(ChessSet.LastDate.desc()).one()
        if Outp == []:
            Result = Restable[OneG['Result']]
            if TC != None:
                Timecontrol = TC
            elif 'Blitz' in OneG['Event']:
                Timecontrol = 'Blitz'
            elif 'Rapid' in OneG['Event']:
                Timecontrol = 'Rapid'
            elif 'TB' in OneG['Event'] or 'Tiebreak' in OneG['Event'] or 'Playoffs' in OneG['Event']:
                Timecontrol = 'unknown'
            else:
                Timecontrol = 'Classical'
            if NT != None:
                NT.SwapHAGame(ED,R,IDW,IDB,Result,Timecontrol,Moves,False)         
                G = NT.DumpGame(ED,R,IDW,IDB,Result,Timecontrol,Moves,IDE)
            else: #def __init__(self,Seeds,TimeControl,Date,EndDate,World,Round,Name='',PGNEvent=None,Moves=bytearray(),Result=None):
                G = ChessGame({0:[None,'First',IDW],1:[None,'First',IDB]},Timecontrol,SD,ED,self,R,'',IDE.id,Moves,Result)
                db.session.add(G)
            db.session.commit()
            print(OneG)
            return G.id
        else:
            print(OneG,Outp)
            return None

    def AddSingleGametoDB(self,game,filename2,tformat,tournament2):
        OneG = dict(game.headers)
        Games = db.session.query(dbc.PGN).filter(dbc.PGN.c.filename == filename2,
                                                 dbc.PGN.c.Event == OneG['Event'],
                                                 dbc.PGN.c.Site == OneG['Site'],
                                                 dbc.PGN.c.Date == OneG['Date'],
                                                 dbc.PGN.c.Round == OneG['Round'],
                                                 dbc.PGN.c.White == OneG['White'],
                                                 dbc.PGN.c.Black == OneG['Black']).all()
        for col in dbc.PGN.c:
            if col.key not in OneG:
                OneG[col.key] = None
        if len(Games) == 0:
            Qu = dbc.PGN.insert().values(format = tformat,
                                        tournament = tournament2,
                                        filename = filename2,
                                                 Event = OneG['Event'],
                                                 Site = OneG['Site'],
                                                 Date = OneG['Date'],
                                                 Round = OneG['Round'],
                                                 White = OneG['White'],
                                                 Black = OneG['Black'],
                                                 Result = OneG['Result'],
                                                 Moves = str(game.mainline_moves()),
                                                 Board = OneG['Board'],
                                                 WhiteFideId = OneG['WhiteFideId'],
                                                 BlackFideId = OneG['BlackFideId'],
                                                 WhiteTitle = OneG['WhiteTitle'],
                                                 BlackTitle = OneG['BlackTitle'],
                                                 WhiteElo = OneG['WhiteElo'],
                                                 BlackElo = OneG['BlackElo'],
                                                 WhiteCountry = OneG['WhiteCountry'],
                                                 BlackCountry = OneG['BlackCountry'],
                                                 WhiteEloChange = OneG['WhiteEloChange'],
                                                 BlackEloChange = OneG['BlackEloChange'],
                                                 Status = 'New')
        else:
            Qu = dbc.PGN.update().where(dbc.PGN.c.filename == filename2,
                                                 dbc.PGN.c.Event == OneG['Event'],
                                                 dbc.PGN.c.Site == OneG['Site'],
                                                 dbc.PGN.c.Date == OneG['Date'],
                                                 dbc.PGN.c.Round == OneG['Round'],
                                                 dbc.PGN.c.White == OneG['White'],
                                                 dbc.PGN.c.Black == OneG['Black'],
                                                 dbc.PGN.c.Status != 'New').values(format = tformat,
                                                    tournament = tournament2,
                                                    filename = filename2,
                                                    Result = OneG['Result'],
                                                    Moves = str(game.mainline_moves()),
                                                    Board = OneG['Board'],
                                                    WhiteFideId = OneG['WhiteFideId'],
                                                    BlackFideId = OneG['BlackFideId'],
                                                    WhiteTitle = OneG['WhiteTitle'],
                                                    BlackTitle = OneG['BlackTitle'],
                                                    WhiteElo = OneG['WhiteElo'],
                                                    BlackElo = OneG['BlackElo'],
                                                    WhiteCountry = OneG['WhiteCountry'],
                                                    BlackCountry = OneG['BlackCountry'],
                                                    WhiteEloChange = OneG['WhiteEloChange'],
                                                    BlackEloChange = OneG['BlackEloChange'],
                                                    Status = 'Updated')
        db.conn.execute(Qu)

    def AddPGNtoDB(self,PGNfile,filename,tformat,tournament):
        while True:
            first_game = chess.pgn.read_game(PGNfile)
            if first_game == None:
                break
            self.AddSingleGametoDB(first_game,filename,tformat,tournament)
        db.session.commit()
            
    def CrawlPGNs(self):
        Fold2 = currentdir+'\\PGNOnl\\'
        Formats = os.listdir(Fold2)
        for fo in Formats:   
            Fold3 = Fold2+fo+'\\'
            CompGroup = os.listdir(Fold3)
            for cg in CompGroup:
                Fold4 = Fold3+cg+'\\'
                Files = os.listdir(Fold4)
                for f in Files:
                    DB = open(Fold4+f)
                    self.AddPGNtoDB(DB,f,fo,cg)

        
    def Chessrun(self,N,Heute,Speichern,SigPerDay):
##        GamesBetwSPs = db.session.query(ChessGame)
##        for g in GamesBetwSPs:
##            if g.TimeControl in ['25+5','15+10','15+5','10+5','10+3']:
##                g.Included = True
##            else:
##                g.Included = False
##        
##        SP5s = self.session.query(oo3.SavePoint).all() #.filter(oo3.SavePoint.id==29411)
##        for SP5 in SP5s:
##            self.conn.execute('DELETE FROM Elo')
##            self.session.delete(SP5)
##            self.session.commit()
##        #return
                
##        Magn = db.session.query(ChessPlayer).filter(ChessPlayer.Name == 'Carlsen, Magnus').all()[0]
##        SP = oo3.SavePoint('Chess',self,datetime.date(2020,1,1),Magn)
##        #SP = self.session.query(oo3.SavePoint).filter(oo3.SavePoint.id==24742).all()[0]
##        SP.World = self
##        ND = SP.cascade()
##        while ND != None:
##            print(ND)
##            SP = oo3.SavePoint('Chess'+str(ND),self,ND)
##            ND = SP.cascade()     
##        #self.session.add(SP1)
##        return
    
##        SP6s = self.session.query(oo3.SavePoint).all() #.filter(oo3.SavePoint.id==18227)
##        for SP6 in SP6s:
##            self.session.delete(SP6)
##            self.session.commit()
##        SP5s = db.session.query(oo3.SavePoint).filter(oo3.SavePoint.id==18248).all()
##        for SP5 in SP5s:
##            print(SP5.Date)
##            SP5.World = self
##            SP5.cascade2()
##        return
##        SP7s = db.session.query(oo3.SimPoint).filter(oo3.SimPoint.id==18301).all()[0]
##       SP7s.World = self
##        SP7s.PrepSim()
##        return
        

        
##        A = ChessSet('Meltwater Tournament',{0:[None,'First',2],1:[None,'First',3],2:[None,'First',10],3:[None,'First',11],
##                                             4:[None,'First',22],5:[None,'First',24],6:[None,'First',26],7:[None,'First',27],
##                                             8:[None,'First',54],9:[None,'First',55],10:[None,'First',60],11:[None,'First',65],
##                                             12:[None,'First',66],13:[None,'First',71],14:[None,'First',124],15:[None,'First',126]},
##                     datetime.date(2022,1,1),datetime.date(2022,1,9),'Meltwater Tournament',self)
        
        #SP7s = db.session.query(oo3.SimPoint).filter(oo3.SimPoint.id==18301).all()[0]
        #Gm = db.session.query(ChessGame).filter(ChessGame.id==10371).all()[0]
        #Gm.World = self
        #Gm.Simulate2(SP7s,self)
        #Tourn = db.session.query(ChessSet).filter(ChessSet.id==357).all()[0]
        #Tourn.Ranking2()
        #db.session.commit()
        #return
        #return
        #try:
        Rty = db.session.query(oo3.SimPoint).filter(oo3.SimPoint.N == 1).all()[0]
        db.init2(Rty)
        if False:#except:
            SP = oo3.SimPoint('Reality',None,None,None,None)
            SP.RootSP = SP.id
            db.session.add(SP)
            db.session.commit()
            db.init2(SP)
        if False:
            self.CrawlPGNs()
            T = self.CheckGames()

            #T = db.conn.execute('SELECT format,tournament,MIN(DATE),MAX(DATE) FROM PGN \
            #                            WHERE format != "Other" GROUP BY format,tournament').all()
            self.CheckTournaments(T)
            #return
            #D = db.Tasks.delete()
            #db.conn.execute(D)
            #db.session.commit()
            Magn = db.session.query(ChessPlayer).filter(ChessPlayer.Name == 'Carlsen, Magnus').all()[0]
            #db.session.commit()
            SP = oo3.SavePoint('Chess',self,datetime.date(2020,1,1),Magn,None)
            db.session.add(SP)
            db.session.commit()
            I = db.Tasks.insert().values(Point=SP.id,Encounter=None,Task='SavePoint',Priority=0)
            #I = db.Tasks.insert().values(Point=18547, Encounter=None, Task='SavePoint', Priority=0)
            #I = db.Tasks.insert().values(Point=19870,Encounter=None,Task='SimPoint',Priority=0)
            #I = db.Tasks.insert().values(Point=None,Encounter=2740,Task='SimEncounter',Priority=-100000)
            db.conn.execute(I)
            db.session.commit()
        #print(db.Reality)
        for a in range(300000):
            A = self.NextTask()
            #print(A)
            pass
        return
        Live = ['https://chess24.com/en/watch/download/meltwater-champions-chess-tour-ftx-crypto-cup-2021-knockout']
        self.Games,self.Players,self.Competitions,self.Savedate = PGNExtractorToSQL.ExtractPGNs(False,self)
        print(self.Competitions['Name'])
        self.DrawishR = 0.0
        self.SigPerDay = SigPerDay
        #print(self.Games)
        #print(self.Games.duplicated(subset=['Moves'],keep='first'))
        RepeatedMoves = self.Games.duplicated(subset=['Moves'],keep='first')
        Draw = self.Games['Result'] == 0.5
        Rapid10p3 = self.Games['TimeControl'] == '10+3'
        Rapid10p5 = self.Games['TimeControl'] == '10+5'
        Rapid15p5 = self.Games['TimeControl'] == '15+5'
        Rapid15p10 = self.Games['TimeControl'] == '15+10'
        Rapid25p5 = self.Games['TimeControl'] == '25+5'
        Rapid25p10 = self.Games['TimeControl'] == '25+10'
        Rapid = Rapid10p3 | Rapid10p5 | Rapid15p5 | Rapid15p10 | Rapid25p5 |Rapid25p10
        Exclude = (RepeatedMoves & Draw) | ~Rapid
        self.Games['Exclude'] = Exclude
        #self.Games['Exclude'] = RepeatedMoves & Draw
        #print(self.Games)
        #self.CompsFormats = pandas.read_csv(filepath_or_buffer=currentdir+'\\DCompFormat.csv')
        self.Field = {}
        for index,row in self.Players.iterrows():
            NewPl = ChessPlayer(row)
            #self.session.add(NewPl)
            self.Field[index]=NewPl
        InSeeds = {}
        Seeds = {}
        Startdate = {}
        Enddate = {}
        Seediter = {}
        for index, row in self.Games.iterrows():
            Comp = row['Competition']
            Dat = dt(row['Date'])
            if Comp not in Seeds:
                Seeds[Comp] = {}
                InSeeds[Comp] = []
                Startdate[Comp] = Dat
                Enddate[Comp] = Dat
                Seediter[Comp] = 0
            W,B = row['White'],row['Black'] #Hier weitermachen.
            Startdate[Comp] = min(Startdate[Comp],Dat)
            Enddate[Comp] = max(Enddate[Comp],Dat)
            for WB in [W,B]:
                if WB not in InSeeds[Comp]:
                    Seeds[Comp][Seediter[Comp]] = [None,'First',numpy.array([WB])]
                    InSeeds[Comp].append(WB)
                    Seediter[Comp] += 1
        self.Tournamentmap = {}
        NameToComp = {}
        self.Children = []
        #print(InSeeds[103])
        #A = csv.reader(open(currentdir+'\\DCompetitions.csv','r'),delimiter=',')
        #next(A)
        A = self.Competitions
        #print(A)
        for index,a in A.iterrows():
            #print('a',a)
            #print(dt(a[2]),Heute)
            if dt(a[2]) <= datetime.date(2022,5,18):
                continue
            if a[5] != 'Other':
                print(a)
                SdsId = self.Competitions[self.Competitions['Name']==a[1]]['ID'].values[0]
                self.Children.append(ChessSet(a[5],Seeds[SdsId],dt(a[2]),dt(a[3]),a[1]))
                NameToComp[a[1]] = self.Children[-1]
                self.Tournamentmap[SdsId]=NameToComp[a[1]]
        self.run(N,Heute,Speichern,SigPerDay)

    def Extract(self,t,Today,Round,Plyr,N,s,Us):
        Group = t.Children[0]
        KO = t.Children[1]
        f = self.OutputDict[t]
        f.write(str(N)+','+t.Name+','+str(Today)+','+str(Round)+','+str(Plyr.Name)+','+str(Us)+',')
        for i in range(31):
            f.write(str(numpy.count_nonzero(2*Group.PW[:,s]==i))+',')
        for i in range(16):
            f.write(str(numpy.count_nonzero(Group.RankNo[:,s]==i+1))+',')
        f.write(str(numpy.count_nonzero(Group.RankNo[:,s]<9))+',')
        SF = numpy.count_nonzero(KO.Children[4].Ranks == Group.Seeds[s][0]) + numpy.count_nonzero(KO.Children[5].Ranks == Group.Seeds[s][0])
        f.write(str(SF)+',')
        f.write(str(numpy.count_nonzero(KO.Children[6].Ranks[:,:] == Group.Seeds[s][0]))+',')
        f.write(str(numpy.count_nonzero(KO.Children[6].Ranks[:,0] == Group.Seeds[s][0]))+'\n')
            
class ChessPlayer(oo3.Player):
    __tablename__ = "chessplayer"
    id = Column(Integer, ForeignKey("player.id"), primary_key=True)
    ShortName = Column(String)
    FideID = Column(Integer, index=True)
    Nationality = Column(String(3), index=True)
    __mapper_args__ = {"polymorphic_identity": "chess_player"}
    def __init__(self,p):
        oo3.Player.__init__(self,p[1])
        self.Nationality = p[3]
        self.ShortName = p[2]
        self.FideID = p[6]
        self.curdir = currentdir

class ChessGame(oo3.Game):
    __tablename__ = "chessgame"
    id = Column(Integer, ForeignKey("game.id"), primary_key=True)
    TimeControl = Column(String)
    PGNEvent = Column(Integer, ForeignKey("chessset.id"))
    Result = Column(Float)
    Moves = Column(LargeBinary)
    #RepeatedMoves = Column(Boolean)
    __mapper_args__ = {"polymorphic_identity": "chessgame"}
    def __init__(self,Seeds,TimeControl,Date,EndDate,World,Round,Name='',PGNEvent=None,Moves=bytearray(),Result=None):
        oo3.Game.__init__(self,Seeds,Date,EndDate,World,Round,Name)
        self.Played = False
        self.curdir = currentdir
        self.TimeControl = TimeControl
        if self.TimeControl in ['25+5','15+10','15+5','10+5','10+3']:
            self.Included = True
        else:
            self.Included = False
        self.PGNEvent = PGNEvent
        self.Moves = Moves
        self.Result = Result
        self.Simulate2(db.Reality, 0)

    def DumpGame(self,Date,Round,White,Black,Result,Timecontrol,Moves,IDE):
        if self.Dumped != False:
            return False
        dtDate = dt(Date)
        if None in [dtDate,self.StartDate,self.EndDate]:
            return False
        if dtDate < self.StartDate:
            return False
        if dtDate > self.EndDate:
            return False
        if Timecontrol not in ['Unknown',self.TimeControl]:
            return False
        if self.Seeds[0][0] != White or self.Seeds[1][0] != Black:
            return False
        if Result in [0.0,0.5,1.0]:
            Sco = numpy.empty((1,2,1))
            Sco[0,0,0] = Result
            Sco[0,1,0] = 1.0-Result
            self.Score = Sco
            self.Result = Result
            self.Played = True
        self.StartDate = dtDate
        self.EndDate = dtDate
        self.Moves = Moves
        self.TimeControl = Timecontrol
        self.Dumped = self
        self.PGNEvent = IDE.id
        return self
    
    def SwapHAGame(self,Date,Round,White,Black,Result,Timecontrol,Moves,WB):
        if self.Swapchecked != False:
            return False
        dtDate = dt(Date)
        if None in [dtDate,self.StartDate,self.EndDate]:
            return False
        if dtDate < self.StartDate:
            return False
        if dtDate > self.EndDate:
            return False
        if Timecontrol not in ['Unknown',self.TimeControl]:
            return False
        if len(numpy.unique(self.Seeds[0])) > 1 or len(numpy.unique(self.Seeds[1])) > 1:
            #print('NO',Date,Round,self.Seeds[0][0],White,self.Seeds[1][0],Black,Result,len(numpy.unique(self.Seeds[0])),len(numpy.unique(self.Seeds[1])))
            return False
        if self.Seeds[0][0] == White and self.Seeds[1][0] == Black:
            self.ToRev = False  
            #print('FOR',Date,Round,self.Seeds[0][0],White,self.Seeds[1][0],Black,Result)       
        elif WB == False and self.Seeds[1][0] == White and self.Seeds[0][0] == Black:
            #print('REV',Date,Round,self.Seeds[0][0],White,self.Seeds[1][0],Black,Result)
            self.ToRev = True
        else:
            return False
        #self.Round = Round
        self.Swapchecked = self
        return self
            
def td(no):
    return datetime.timedelta(no)


            
class ChessSet(oo3.Set):
    __tablename__ = "chessset"
    id = Column(Integer, ForeignKey("set.id"), primary_key=True)
    __mapper_args__ = {"polymorphic_identity": "chessset"}
    def __init__(self,Type,Seeds,Date,EndDate,Name,World,RootFo,RootNa): #split up
        oo3.Set.__init__(self,Type,Seeds,Date,EndDate,Name,World)
    #    if Seeds != {}:
    #        self.propagate(Type,Seeds,Date,EndDate,Name,World)
    #def propagate(self,Type,Seeds,Date,EndDate,Name,World):
        Type = self.Type
        print(Type,Name)
        P = len(Seeds)
        if P == 0:
            return
        self.curdir = currentdir
        self.ToRev = False
        self.TieBreakers = []
        self.DrawishR = 0.0
        self.RootFo = RootFo
        self.RootNa = RootNa
        if Type == 'Meltwater Tournament':
            self.Children.append(ChessSet('Meltwater Prelims',self.Seeding,Date,Date+td(2),'Prelims',World,self.RootFo,self.RootNa))
            NSeeds = {}
            for i in range(8):
                NSeeds[i] = [self.Children[0],'Ranks',i]
            self.Children.append(ChessSet('Meltwater Knockout',NSeeds,Date+td(3),Date+td(8),'Knockout',World,self.RootFo,self.RootNa))
        elif Type == 'Meltwater Prelims':
            self.TieBreakers = ['Points','DirectEncounter','Wins','SonnebornBerger','Koya','Random']
            self.ChiBerger(P,1,'ChessGame','15+10')
        elif Type == 'Meltwater Knockout':
            for i in range(int(P/2)):
                self.Children.append(ChessSet('Meltwater KO Match',{0:self.Seeding[i],1:self.Seeding[7-i]},Date,Date+td(1),'Quarterfinal '+str(i+1),World,self.RootFo,self.RootNa))
            for i in range(int(P/4)):
                self.Children.append(ChessSet('Meltwater KO Match',{0:[self.Children[0+i],'Ranks',0],1:[self.Children[3-i],'Ranks',0]},Date+td(2),Date+td(3),'Semifinal '+str(i+1),World,self.RootFo,self.RootNa))
            self.Children.append(ChessSet('Meltwater KO Match',{0:[self.Children[4],'Ranks',0],1:[self.Children[5],'Ranks',0]},Date+td(4),Date+td(5),'Final',World,self.RootFo,self.RootNa))
            self.Children.append(ChessSet('Meltwater KO Match',{0:[self.Children[4],'Ranks',1],1:[self.Children[5],'Ranks',1]},Date+td(4),Date+td(5),'3rd place match',World,self.RootFo,self.RootNa))
        elif Type == 'Meltwater KO Match':
            self.TieBreakers = [0,1,2]
            self.Children.append(ChessSet('Meltwater KO Sets',self.Seeding,self.StartDate,self.EndDate,'Rapid',World,self.RootFo,self.RootNa))
            self.Children.append(ChessSet('Meltwater 2 Blitz',self.Seeding,self.EndDate,self.EndDate,'Blitz tie-breaker',World,self.RootFo,self.RootNa))
            self.Children.append(ChessSet('Meltwater Armageddon',self.Seeding,self.EndDate,self.EndDate,'Armageddon tie-breaker',World,self.RootFo,self.RootNa))
            self.HMatrix = numpy.array([[0,0,0]])
            self.AMatrix = numpy.array([[1,1,1]])      
        elif Type == 'Meltwater KO Sets':         
            self.TieBreakers = ['Points']
            Dts = [self.StartDate,self.EndDate]
            for i in range(2):
                self.Children.append(ChessSet('Meltwater KO Set',self.Seeding,Dts[i],Dts[i],'Set '+str(i+1),World,self.RootFo,self.RootNa))
            self.HMatrix = numpy.array([[0,0]])
            self.AMatrix = numpy.array([[1,1]])
        elif Type == 'Meltwater KO Set':           
            self.TieBreakers = ['Points']
            self.ChiBerger(2,4,'ChessGame','15+10')
        elif Type == 'Meltwater 2 Blitz':          
            self.TieBreakers = ['Points']
            self.ChiBerger(2,2,'ChessGame','5+3')
        elif Type == 'Meltwater Armageddon':         
            self.TieBreakers = ['Points','moreBlack']       
            self.ChiBerger(2,1,'ChessGame','Armageddon 5&4')
        elif Type == 'MCI2020':
            self.Children.append(ChessSet('MCI2020 Prelims',self.Seeding,Date,Date+td(12),'Prelims',World,self.RootFo,self.RootNa))
            NSeeds = {}
            for i in range(4):
                NSeeds[i] = [self.Children[0],'Ranks',i]
            self.Children.append(ChessSet('MCI2020 Final 4',NSeeds,Date+td(13),Date+td(15),'Final 4',World,self.RootFo,self.RootNa))
        elif Type == 'MCI2020 Prelims':
            self.TieBreakers = ['Points','SubPoints','DirectEncounter','Random']
            self.ChiBerger(8,1,'ChessSet','','MCI2020 Match')
        elif Type == 'MCI2020 Final 4':
            for i in range(int(P/2)):
                self.Children.append(ChessSet('MCI 2020 KO Match',{0:self.Seeding[i],1:self.Seeding[3-i]},Date+td(i),Date+td(i),'Semifinal '+str(i+1),World,self.RootFo,self.RootNa))
            self.Children.append(ChessSet('MCI 2020 KO Match',{0:[self.Children[0],'Ranks',0],1:[self.Children[1],'Ranks',0]},Date+td(2),Date+td(2),'Final',World,self.RootFo,self.RootNa))
        elif Type == 'MCI 2020 KO Match':
            self.TieBreakers = [0,1,2,3]
            self.Children.append(ChessSet('Meltwater KO Set',self.Seeding,self.StartDate,self.EndDate,'Rapid',World,self.RootFo,self.RootNa))
            self.Children.append(ChessSet('Meltwater 2 Blitz',self.Seeding,self.EndDate,self.EndDate,'Blitz tie-breaker',World,self.RootFo,self.RootNa))
            self.Children.append(ChessSet('Meltwater 2 Blitz',self.Seeding,self.EndDate,self.EndDate,'Blitz tie-breaker',World,self.RootFo,self.RootNa))
            self.Children.append(ChessSet('Meltwater Armageddon',self.Seeding,self.EndDate,self.EndDate,'Armageddon 5&4',World,self.RootFo,self.RootNa))
            self.HMatrix = numpy.array([[0,0,0,0]])
            self.AMatrix = numpy.array([[1,1,1,1]])           
        elif Type == 'MCI2020 Match':
            self.TieBreakers = [0,1]
            self.Children.append(ChessSet('Meltwater KO Set',self.Seeding,self.StartDate,self.EndDate,'Rapid',World,self.RootFo,self.RootNa))
            self.Children.append(ChessSet('Meltwater Armageddon',self.Seeding,self.StartDate,self.EndDate,'Armageddon 5&4',World,self.RootFo,self.RootNa))
            self.HMatrix = numpy.array([[0,0]])
            self.AMatrix = numpy.array([[1,1]])
        elif Type == 'Meltwater Tournament 2022':
            self.Children.append(ChessSet('Meltwater Prelims 2022',self.Seeding,Date,Date+td(3),'Prelims',World,self.RootFo,self.RootNa))
            NSeeds = {}
            for i in range(8):
                NSeeds[i] = [self.Children[0],'Ranks',i]
            self.Children.append(ChessSet('Meltwater Knockout 2022',NSeeds,Date+td(4),Date+td(7),'Knockout',World,self.RootFo,self.RootNa))
            self.Children[0].Children[-1].CalcTrigger = self.Children[0]
            self.Children[1].Children[3].Children[-1].Children[-1].CalcTrigger = self.Children[1]
            self.Children[1].Children[5].Children[-1].Children[-1].CalcTrigger = self.Children[1]
        elif Type == 'Meltwater Prelims 2022':
            self.DrawishR = 0.2
            self.TieBreakers = ['Points','DirectEncounter','Wins','SonnebornBerger','Koya','Random']
            self.ChiBerger(P,1,'ChessGame','15+10')
        elif Type == 'Meltwater Knockout 2022':
            for i in range(int(P/2)):
                self.Children.append(ChessSet('Meltwater KO Match 2022',{0:self.Seeding[i],1:self.Seeding[7-i]},Date,Date,'Quarterfinal '+str(i+1),World,self.RootFo,self.RootNa))
            for i in range(int(P/4)):
                self.Children.append(ChessSet('Meltwater KO Match 2022',{0:[self.Children[0+i],'Ranks',0],1:[self.Children[3-i],'Ranks',0]},Date+td(1),Date+td(1),'Semifinal '+str(i+1),World,self.RootFo,self.RootNa))
            self.Children.append(ChessSet('Meltwater KO Match',{0:[self.Children[4],'Ranks',0],1:[self.Children[5],'Ranks',0]},Date+td(2),Date+td(3),'Final',World,self.RootFo,self.RootNa))
        elif Type == 'Meltwater KO Match 2022':
            self.TieBreakers = [0,1,2]
            self.Children.append(ChessSet('Meltwater KO Set',self.Seeding,self.StartDate,self.EndDate,'Rapid',World,self.RootFo,self.RootNa))
            self.Children.append(ChessSet('Meltwater 2 Blitz',self.Seeding,self.EndDate,self.EndDate,'Blitz tie-breaker',World,self.RootFo,self.RootNa))
            self.Children.append(ChessSet('Meltwater Armageddon',self.Seeding,self.EndDate,self.EndDate,'Armageddon tie-breaker',World,self.RootFo,self.RootNa))
            self.HMatrix = numpy.array([[0,0,0]])
            self.AMatrix = numpy.array([[1,1,1]])
        if P == 2:
            self.RDi = [1]
            self.QforW = ['Rating','Pts','To Win','To Tie','To Lose']
            self.PrioOne = 'To win'
        elif 'Knockout' in Type or 'Final 4' in Type:
            Pz = int(8-math.log(P,2))
            self.QforW = ['Rating']+['➔R128','➔R64','➔R32','➔R16','➔QF','➔SF','Win SF','➔F','Win F','Win'][Pz:]
            self.PrioOne = 'To win'
        elif 'Prelims' in Type:
            self.QforW = ['Pts','xPts','➔QF','|18Pts','|19Pts','|20Pts','|21Pts','|22Pts']
            self.PrioOne = 'To qualify'
            self.Spots = 8
        elif 'Tournament' in Type:
            #self.QforW = ['Rating','Pts_','Top 8','Win QF','➔SF','Win SF','➔F','Win F','Win']
            self.QforW = ['Rating W','Rating B','Prelims xPts','Top 8','➔SF','➔F','Win']
            self.PrioOne = 'To win'
        elif 'MCI2020' == Type:
            self.QforW = ['Rating W','Rating B','Prelims xPts','➔SF','➔F','Win']
            self.PrioOne = 'To win'
        else:
            print(Type,self.Name,P)
        #print(self.Name,self.Type)
        self.fill()

    def SpecialRanks(self,N,P):
        self.Ranks = numpy.zeros(shape=(N,P),dtype=intDT)
        for c in self.Children:
            c.BuildRanks(db.Reality.id)
        if 'Meltwater Knockout' in self.Type:
            self.Ranks[:,:2] = self.Children[6].Ranks
            if '2022' in self.Type:
                self.Ranks[:,2] = self.Children[4].Ranks[:,1]
                self.Ranks[:,3] = self.Children[5].Ranks[:,1]
            else:
                self.Ranks[:,2:4] = self.Children[7].Ranks
            for i in range(4):
               self.Ranks[:,4+i] = self.Children[3-i].Ranks[:,1]
            self.SeedRank = self.FromIDToSeed(self.Ranks)
            self.RankNo = numpy.argsort(self.SeedRank)#+1      
            self.RankNo = numpy.where(self.RankNo<4,self.RankNo+1,5)
        elif 'Meltwater Tournament' in self.Type:
            self.Ranks[:,:8] = self.Children[1].Ranks
            self.Ranks[:,8:] = self.Children[0].Ranks[:,8:]
            self.SeedRank = self.FromIDToSeed(self.Ranks)
            self.RankNo = numpy.argsort(self.SeedRank)#+1
            self.RankNo = numpy.where(numpy.abs(5.5 -self.RankNo)<2 ,5,self.RankNo+1)
        elif 'Final 4' in self.Type:
            self.Ranks[:,:2] = self.Children[2].Ranks
            self.Ranks[:,2] = self.Children[0].Ranks[:,1]
            self.Ranks[:,3] = self.Children[1].Ranks[:,1]
            self.SeedRank = self.FromIDToSeed(self.Ranks)
            self.RankNo = numpy.argsort(self.SeedRank)      
            self.RankNo = numpy.where(self.RankNo<2,self.RankNo+1,3)
        elif 'MCI2020' == self.Type:
            self.Ranks[:,:4] = self.Children[1].Ranks
            self.Ranks[:,4:] = self.Children[0].Ranks[:,4:]
            self.SeedRank = self.FromIDToSeed(self.Ranks)
            self.RankNo = numpy.argsort(self.SeedRank)
            self.RankNo = numpy.where(numpy.abs(2.5 -self.RankNo)<1 ,3,self.RankNo+1)    
        if '2022' in self.Type:
            self.RankNo = numpy.where(numpy.abs(3.5 -self.RankNo)<1 ,3,self.RankNo) 

    
    def ResArrays(self,N,Extra,SP):
        maxN = N + Extra
        LC = len(self.Children)
        if self.Type in ['Meltwater Prelims 2022']:
            ResultArrH = numpy.zeros(shape=(maxN,LC,1),dtype=floatDT)
            ResultArrA = numpy.zeros(shape=(maxN,LC,1),dtype=floatDT)
            PointsKey = numpy.array([0,1,3])
            for chi in range(LC):
                chiPts = (2*self.Children[chi].Score).astype(intDT)
##                if type(self.PointsKey) == str:
##                    UnSco = numpy.unique(chiPts[:,:,0])
##                    PointsKey = numpy.arange(numpy.max(UnSco)+1) * 0.5
##                    self.PointsKey = numpy.arange(numpy.max(UnSco)+1) * 0.5
##                else:
##                    PointsKey = self.PointsKey
                ResultArrH[:,chi,0] = PointsKey[chiPts[:,0,0]]
                ResultArrA[:,chi,0] = PointsKey[chiPts[:,1,0]]
            ResultsArr = ResultArrH - ResultArrA
            #print('hier')
            return ResultArrH,ResultArrA,ResultsArr
        elif self.Type in ['MCI2020 Prelims']:
            P = len(self.Seeds)
            ResultArrH = numpy.zeros(shape=(maxN,LC,2),dtype=floatDT)
            ResultArrA = numpy.zeros(shape=(maxN,LC,2),dtype=floatDT)
            for chi in range(LC):
                self.Children[chi].reestablish(SP)
                Rapid = self.Children[chi].Children[0]
                Rapid.reestablish(SP)
                Armag = self.Children[chi].Children[1]
                #Played = (numpy.max(Rapid.Score[N,:,0])>2.0) | (numpy.sum(Armag.Score[N,:,0])>0.0)
                Win = (Rapid.Score[:,0,0]>Rapid.Score[:,1,0])
                Draw = (Rapid.Score[:,0,0]==Rapid.Score[:,1,0])
                Loss = (Rapid.Score[:,0,0]<Rapid.Score[:,1,0])
                DWin = Draw & (Armag.Score[:,0,0]>Armag.Score[:,1,0])
                DLoss = Draw & (Armag.Score[:,0,0]<Armag.Score[:,1,0])
                ResultArrHc, ResultArrAc, ResultsArrc = Rapid.ResArrays(self.N, 0, SP)
                Rapid.PW = oo3.PointsPT(Rapid.HMatrix, ResultArrHc, Rapid.AMatrix, ResultArrAc, P)
                ResultArrH[:,chi,1] += Rapid.PW[:,0,0]
                ResultArrA[:,chi,1] += Rapid.PW[:,1,0]   
                Points = numpy.array([[3,0],[0,3],[2,1],[1,2]])
                res = [Win,Loss,DWin,DLoss] #Add Not Played
                for i in range(4):
                    ResultArrH[:,chi,0] += res[i]*Points[i,0]
                    ResultArrA[:,chi,0] += res[i]*Points[i,1]
            ResultArr = (ResultArrH>1.5).astype(intDT)-(ResultArrA<1.5).astype(intDT)
            #print(ResultArrH[:,:,0])
            return ResultArrH,ResultArrA,ResultArr
        else:
            return self.DefaultResArrays(N,Extra)

    def WritePage(self,World,N):
        LC = len(self.Children)
        for chi in range(LC):
            self.Children[chi].WritePage(World,N)
        #print(self.Name,self.RoundsSum)
        Sz = self.Sims['Rank|Pts'][0][0].size
        sS = self.Sims['Rank|Pts']
        NPts = sS[0][2].shape[0]
        NRanks = sS[0][2].shape[1]
        OutpMat = numpy.zeros(shape=(Sz,len(self.RoundsArray)),dtype=floatDT)
        OutpMatPl = {}
        for nr in range(NRanks):
            OutpMatPl[nr] = numpy.zeros(shape=(sS[0][2].shape[1],len(self.RoundsArray)),dtype=floatDT)
        n = 0
        for rr in self.Sims['Rank|Pts']:
            r = sS[rr]
            Mat = r[2]
            RkPl = numpy.sum(Mat[:,:8,:],axis=(0,1))/numpy.sum(Mat[:,:,:],axis=(0,1))
            OutpMat[:,n] = RkPl
            RkPts = {}
            for nr in range(NRanks):
                RkPts[nr] = numpy.sum(Mat[:,:,nr],axis=0)/numpy.sum(Mat[:,:,nr],axis=(0,1))
                OutpMatPl[nr][:,n] = RkPts[nr]
            n+=1
        indi = []
        for pl in range(Sz):
            indi.append(World.Field[self.Seeds[pl][0]].Name)
        PD = pandas.DataFrame(OutpMat, index=indi, columns=self.RoundsArray) #numpy.arange(1,n+1)
        PDPl = {}
        for nr in range(NRanks):
            PDPl[nr] = pandas.DataFrame(OutpMatPl[nr], index=numpy.arange(1,NRanks+1), columns=self.RoundsArray) #numpy.arange(1,n+1)
        #HT = PD.to_html()
        HT = viz.TimeChart(self,OutpMat,indi,World,list(sS),N)
        HT += '<h1>'+self.Name+' Forecast</h1>'+viz.OverviewTable(self,World)
        HT += '<h1>'+self.Name+' Games</h1>'+viz.Fixtures(self,World)
        #HT += '<h1>'+self.Name+' Forecast</h1>'+viz.OverviewTableBak(self,World,sS[self.Progress])
        HT += viz.Overview(self,World,sS[self.Progress])
        sra = self.RoundsArray[-1]
        Colours = ['#000']
        for i in range(8):
            Colours += ['#'+2*(hex(4)[2:])+(hex(15*16-6*i)[2:])+2*(hex(4)[2:])]
        for i in range(8):
            Colours += ['#'+(hex(15*16-6*i)[2:])+2*(hex(4)[2:])+2*(hex(4)[2:])]
        #print(Colours)
        if self.Type in ['Meltwater Prelims','Meltwater Prelims 2022']:
            for nr in range(NRanks):
                HT += '<h2>'+World.Field[self.Seeds[nr][0]].Name+'</h2>'
                #HT += '<div>'+PDPl[nr].to_html()+'</div>'
                HT += '<h3>how to not qualify</h3>'+viz.OverviewTable(self,World,nr)
                HT += viz.PlayerFixtures(self,self.Seeds[nr][0],World)
                #HT += viz.Permutsi(self,nr,World)
                #print(self.Name,Colours)
                HT += viz.SVGCondProb(sS[sra][0],sS[sra][1],sS[sra][2][:,:,nr],Colours)
        self.HTMLPage(HT)

    def CountRounds(self,RoundTree):
        RoundsSum = self.Rounds
        r = 0
        maxR = 0
        for chi in range(len(self.Children)):
            c = self.Children[chi]
            if c.IsGame() == True:
                c.RoundTree = (RoundTree+[c.Round]).copy()
                c.PrintRoundTree()
                maxR = max(maxR,c.Round)
                continue
            if self.Type in ['Meltwater Knockout','Meltwater Knockout 2022'] and chi in [1,2,3,5,7]:
                c.CountRounds(RoundTree+[r])
            else:
                r+=1
                maxR = max(maxR,r)
                RoundsSum += c.CountRounds(RoundTree+[r])
        self.MaxRoundTree = RoundTree+[maxR]
        self.RoundsSum = RoundsSum
        self.RoundTree = RoundTree
        #print('test',self.Folder,self.RoundTree,self.MaxRoundTree)
        return RoundsSum

    def AppendChild(self,Seeding,Timecontrol,SD,ED,W,R,Name):
        #if such a game exists, take that else

        # look in encounterranks join seeding who should be here.
        # then look for these.
        ToLookFor = [None,None]
        for sd in [0,1]: #Seeding[p] = [None,'First',Pl]
            if Seeding[sd][1] == 'First':
                ToLookFor[sd] = Seeding[sd][2].id
            elif Seeding[sd][1] == 'Ranks':
                ToLookFor[sd] = Seeding[sd][0].Ranks[0,Seeding[sd][2]]
        Home = aliased(db.FirstSeeds)
        Away = aliased(db.FirstSeeds)
        CG = db.session.query(ChessGame)\
             .join(dbc.PGN,dbc.PGN.c.ChessGameID==ChessGame.id)\
             .join(Home,Home.c.encounter==ChessGame.id)\
             .join(Away,Away.c.encounter==ChessGame.id)\
             .outerjoin(db.SetStructure,db.SetStructure.c.child==ChessGame.id)\
             .filter(ChessGame.StartDate<=ED,ChessGame.EndDate>=SD)\
             .filter(Home.c.spot==0,Away.c.spot==1)\
             .filter(Home.c.player==ToLookFor[0],Away.c.player==ToLookFor[1])\
             .filter(db.SetStructure.c.child==None)\
             .filter(dbc.PGN.c.format == self.RootFo)\
             .filter(dbc.PGN.c.tournament == self.RootNa)\
             .order_by(ChessGame.EndDate)\
             .all()
        print(CG)
        #if CG != []:
        #    self.Children.append(CG[0])
        for cg in range(len(CG)):
            if CG[cg] not in self.Children:
                for s in [0,1]:
                    if Seeding[s][1] == 'Ranks':
                        CG[cg].Priority = max(CG[cg].Priority,Seeding[s][0].Priority+1)
                self.Children.append(CG[cg])
                break
        else:
            self.Children.append(ChessGame(Seeding,Timecontrol,SD,ED,W,R,Name))
        

    def ChiBerger(self,P,S,GoS,TimeControl,Type=''):
        Rounds = self.ChiBerg(P,S)
        R = range(len(Rounds))
        for i in R:
            if GoS == 'ChessGame':
                if 'Armageddon' in self.Name:
                    self.AppendChild({1:self.Seeding[self.HMatrix[0,i]],0:self.Seeding[self.AMatrix[0,i]]},TimeControl,self.StartDate,self.EndDate,self.World,Rounds[i],'Game '+str(Rounds[i])+'/'+str(len(Rounds)))
                    #self.Children.append(ChessGame({1:self.Seeding[self.HMatrix[0,i]],0:self.Seeding[self.AMatrix[0,i]]},TimeControl,self.StartDate,self.EndDate,self.World,Rounds[i],'Game '+str(Rounds[i])+'/'+str(len(Rounds))))
                    self.HMatrix[0,i] = 1
                    self.AMatrix[0,i] = 0
                else:
                    self.AppendChild({0:self.Seeding[self.HMatrix[0,i]],1:self.Seeding[self.AMatrix[0,i]]},TimeControl,self.StartDate,self.EndDate,self.World,Rounds[i],'Game '+str(Rounds[i])+'/'+str(len(Rounds)))
            elif GoS == 'ChessSet':
                self.Children.append(ChessSet(Type,{0:self.Seeding[self.HMatrix[0,i]],1:self.Seeding[self.AMatrix[0,i]]},self.StartDate,self.EndDate,'Match '+str(Rounds[i])+'/'+str(len(Rounds)),self.World,self.RootFo,self.RootNa))
                #self,Type,Seeds,Date,EndDate,Name=''


        
    def DumpGame(self,Date,Round,White,Black,Result,Timecontrol,Moves,IDE):
        for c in self.Children:
            Filled = c.DumpGame(Date,Round,White,Black,Result,Timecontrol,Moves,IDE)
            if Filled != False:
                Filled.IfR[self] = self.CalcOdds('Rank','PtsB',Filled,slice(0,-1))[2]
                if Result in [0.0,0.5,1.0]:
                    self.Progress+=1
                return Filled
        return False

    def SwapHAGame(self,Date,Round,White,Black,Result,Timecontrol,Moves,WB):
        self.n = 0
        #print(self.Name,self.Type)
        for c in self.Children:
            if self.Type in ['Meltwater KO Set','Meltwater 2 Blitz']:
                WB = True
            Filled = c.SwapHAGame(Date,Round,White,Black,Result,Timecontrol,Moves,WB)
            if Filled != False:
                if self.Type in ['Meltwater Prelims','Meltwater Prelims 2022']:
                    c.Round = Round
                if c.ToRev == True:
                    #print('Vorher',c.Seeds[0][0],c.Seeds[1][0])
                    #Temp = c.Seeds[0].copy()
                    #c.Seeds[0] = c.Seeds[1].copy()
                    #c.Seeds[1] = Temp
                    #if 'Armageddon' in Timecontrol:
                    for s in c.Seeding:
                        Ar = c.Seeding[s]
                        if Ar[1] == 'First':
                            c.Seeds[1-s] = Ar[2]
                        if Ar[1] == 'Ranks':
                            for ar0 in Ar[0].SeedTrigger:
                                if ar0[0] == c and ar0[1] == Ar[2] and ar0[2] == s:
                                    ar0[2] = 1-s

                            
                            #Ar[0].SeedTrigger.append([c,Ar[2],1-s])
                            #print(Ar[0].Name,c.Name,Ar[2],1-s)
##                    Temp = c.Seeding[0].copy()
##                    c.Seeding[0] = c.Seeding[1].copy()
##                    c.Seeding[1] = Temp
##                    Temp = c.Seeds[0][0]
##                    c.Seeds[0] = numpy.array([c.Seeds[1][0]])
##                    c.Seeds[1] = numpy.array([Temp])
                    #print('Nachher',c.Seeds[0][0],c.Seeds[1][0])
                    Temp = self.HMatrix[:,self.n].copy()
                    self.HMatrix[:,self.n] = self.AMatrix[:,self.n].copy()
                    self.AMatrix[:,self.n] = Temp
                    #print(self.Name,'swapped')
                return Filled
            self.n+=1
        return False
                
