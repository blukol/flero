import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
import csv
import datetime
import scipy.stats as ss
import pickle
import numpy
import pandas
#import viz
import math
import scipy.linalg as la
import configglobal as db
from scipy.stats import entropy
#import PGNExtractor
#engine = create_engine("sqlite+pysqlite:///"+flerotron.db", echo=True)
from sqlalchemy.orm import aliased, relationship
from sqlalchemy import Column, Integer, String, ForeignKey, Table, Date, MetaData, LargeBinary,Float,Boolean, asc, desc, text

boolDT = db.boolDT
intDT = db.intDT
floatDT = db.floatDT

def dt(a):
    if type(a)==float: return None
    elif type(a)==datetime.date: return a
    return datetime.date(int(a[0:4]),int(a[5:7]),int(a[8:10]))

#class Entity():
class Entity(db.Base):
    __tablename__ = 'entity'
    id = Column(Integer, primary_key=True)
    type = Column(String)
    Name = Column(String)
    __mapper_args__ = {
        "polymorphic_identity": "entity",
        "polymorphic_on": type}
    def __init__(self,Name):     
        self.Name = Name
        #db.Base.metadata.create_all(self.engine)
    def HTMLPage(self,Text):
        HTML = '<!DOCTYPE html><html><head>\
<meta charset="utf-8"><title>Flerotron</title>\
<LINK REL=StyleSheet HREF="/style.css" TYPE="text/css">\
<meta name = "author" content = "Lars Schiefler"></head>\
<body style=" font-family:MetricWeb,sans-serif;; \
background-color: #0ABAB5; text-align:center;">' + viz.Logo() + Text + '</body></html>'  #0AA
        NewFolder = self.curdir+self.Folder
        if not os.path.exists(NewFolder):
            os.makedirs(NewFolder)
        Page = open(NewFolder+'\\index.html','w')
        Page.write(HTML)
        Page.close()

class World(Entity):
    def __init__(self,Name):
        Entity.__init__(self,Name)
##        db.Base = Base
##        self.engine = engine
##        Session = sessionmaker(bind = engine)
##        self.session = Session()
##        self.conn = self.engine.connect()
        self.GroundLevel=2881.0
        self.SigPerDay = 5.0
        #self.metadata_obj = MetaData()
##    def Tables(self):
##        self.SetStructure = Table('SetStructure',db.Base.metadata,
##                       Column('parent',Integer,ForeignKey('set.id'), index=True),
##                       Column('child',Integer,ForeignKey('encounter.id'), index=True),
##                       Column('spot',Integer))
##        self.FirstSeeds = Table('FirstSeeds',db.Base.metadata,
##                       Column('encounter',Integer,ForeignKey('encounter.id'), index=True),
##                       Column('spot',Integer),
##                       Column('player',Integer,ForeignKey('player.id'), index=True))
##        self.RankSeeds = Table('RankSeeds',db.Base.metadata,
##                       Column('encounter',Integer,ForeignKey('encounter.id'), index=True),
##                       Column('spot',Integer),
##                       Column('rootencounter',Integer,ForeignKey('encounter.id'), index=True),
##                       Column('rank',Integer))
####        self.OutcomeRanks = Table('OutcomeRanks',db.Base.metadata,                          CONTINUE HERE.
####                       Column('encounter',Integer,ForeignKey('encounter.id'), index=True),
####                       Column('simpoint',Integer,ForeignKey('simpoint.id')),
####                       Column('rank',Integer),
####                       Column('player',LargeBinary)
##        self.SimGame = Table('SimGame',db.Base.metadata,
##                      Column('SimPoint',Integer,ForeignKey('simpoint.id')),
##                      Column('Game',Integer,ForeignKey('game.id')),
##                      Column('Score',LargeBinary))
##        self.Elo = Table('Elo',db.Base.metadata,
##                      Column('Point',Integer,ForeignKey('point.id')),
##                      Column('PlayerID',Integer,ForeignKey('player.id')),
##                      Column('MatrSeed',Integer),
##                      Column('Categories',String),
##                      Column('Bucket',Integer),
##                      Column('Elo',Float),
##                      Column('SimmedElo',LargeBinary))
##        Base.metadata.create_all(engine)
##        print(Base.metadata.tables.keys())
##        print(Base.metadata.tables['chessplayer'])

##    def AppendIGM(self,tA,tB,tC,tD,alpha,beta,mu,R1,R2,I,G):
##        RAB = R1*(1+R1/R2*alpha*(1-alpha))
##        RCD = R1*(1+R1/R2*beta*(1-beta))
##        print(tA,tB,tC,tD)
##        G = AppendG(tA,tB,RAB,G)
##        G = AppendG(tC,tD,RCD,G)
##        R1n = R1*(beta*(1-beta)+alpha*(1-alpha)+alpha*beta)
##        if alpha < 1 and beta < 1:
##            RAC = R1n+R2/((1-beta)*(1-alpha))
##            G = AppendG(tA,tC,RAC,G)
##        if alpha > 0 and beta > 0:
##            RBD = R1n+R2/(beta*alpha)
##            G = AppendG(tB,tD,RBD,G)
##        muR2 = mu/R2
##        I[tA]+=muR2 * (1-alpha)
##        I[tB]+=muR2 * alpha
##        I[tC]-=muR2 * (1-beta) 
##        I[tD]-=muR2 * beta
##        return I,G

    def NextTask(self):
        #try:
        Tasky = db.session.query(db.Tasks).order_by(db.Tasks.c.Priority.asc()).all()
        if len(Tasky) == 0: return
        Task = Tasky[0]
        #print(Task)
        #except:
        #    return False
        if Task[2] == 'SavePoint':
            SP = db.session.query(SavePoint).filter(SavePoint.id==Task[0]).one()
            SP.World = self
            ND = SP.cascade()
            if ND != None:
                NSP = SavePoint(str(ND),self,ND,None,SP)
                I = db.Tasks.insert().values(Point=NSP.id,Task='SavePoint',Priority=(ND-datetime.date.today()).days-1000)
                db.conn.execute(I)
                I = db.Tasks.insert().values(Point=NSP.id,Task='SimPoints',Priority=(datetime.date.today()-ND).days*100)
                db.conn.execute(I)
                Encs = db.session.query(Encounter).filter(Encounter.EndDate>SP.Date,Encounter.EndDate <= ND,Encounter.Priority!=None).all()
                print(len(Encs))
                for enc in Encs:
                    print(enc.id,enc.Priority)
                    I = db.Tasks.insert().prefix_with("OR REPLACE").values(Point=db.Reality.id, Encounter=enc.id,Task='SimEncounter', Priority=(SP.Date-datetime.date.today()).days-1000+enc.Priority)
                    db.conn.execute(I)
            D = db.Tasks.delete().where(db.Tasks.c.Point == SP.id,db.Tasks.c.Task == Task[2])
            db.conn.execute(D)
        elif Task[2] == 'SimPoints':
            SP = db.session.query(SavePoint).filter(SavePoint.id==Task[0]).one()
            SP.World = self
            NewPs = SP.cascade2()
            for SimP in NewPs:
                I = db.Tasks.insert().values(Point=SimP.id,Task='SimPoint',Priority=((datetime.date.today()-SimP.Date).days-50)*100)
                #calc all comps
                db.session.commit()
                db.conn.execute(I)
            D = db.Tasks.delete().where(db.Tasks.c.Point == SP.id,db.Tasks.c.Task == Task[2])
            db.session.commit()
            db.conn.execute(D)
        elif Task[2] == 'SimPoint':       
            SimP = db.session.query(SimPoint).filter(SimPoint.id==Task[0]).one()
            SimP.World = self
            SimP.PrepSim()
            Encs = db.session.query(Encounter).filter(Encounter.StartDate <= SimP.Date + datetime.timedelta(14),Encounter.EndDate > SimP.Date).all()
            #to do: get all games that just ended
            for enc in Encs:
                #priority is 1 above max priority of children and 1 above rankseed.
                if enc.Priority == None:
                    continue
                I = db.Tasks.insert().prefix_with("OR REPLACE").values(Point=SimP.id,Encounter=enc.id,Task='SimEncounter',Priority=(enc.Priority))
                db.conn.execute(I)
            db.session.commit()
            D = db.Tasks.delete().where(db.Tasks.c.Point == SimP.id,db.Tasks.c.Task == Task[2])
            db.session.execute(D)
        elif Task[2] == 'SimEncounter':
            if Task[0]==None:
                SP = None
            else:
                SP = db.session.query(SimPoint).filter(SimPoint.id==Task[0]).one()
            Enc = db.session.query(Encounter).filter(Encounter.id==Task[1]).one()
            Enc.Simulate2(SP,self)
            db.session.commit()
            D = db.Tasks.delete().where(db.Tasks.c.Point == Task[0],db.Tasks.c.Encounter == Enc.id,db.Tasks.c.Task == Task[2])
            db.conn.execute(D)
        db.session.commit()
        return True
            
    def AppendIG(self,t1,t2,mu,sigma):
        Neu = numpy.true_divide(1.0,sigma) #Add the new game
        self.I[t1] -= mu*Neu
        self.I[t2] += mu*Neu
        self.G[t1,t2]+=Neu
        self.G[t2,t1]+=Neu
        self.G[t1,t1]-=Neu
        self.G[t2,t2]-=Neu

    def AppendG(self,t1,t2,sigma,G):
        Neu = numpy.true_divide(1.0,sigma) #Add the new game
        G[t1,t2]+=Neu
        G[t2,t1]+=Neu
        G[t1,t1]-=Neu
        G[t2,t2]-=Neu
        return G

    def AddOnePrior(self,t1,t2,UPrior1,sigma2):
        mu = UPrior1
        Neu = numpy.true_divide(1.0,sigma2) #Add the new game
        self.I2[t1] -= mu*Neu
        self.I2[t2] += mu*Neu
        self.G2[t1,t2]+=Neu
        self.G2[t2,t1]+=Neu
        self.G2[t1,t1]-=Neu
        self.G2[t2,t2]-=Neu
    
    def MoveClubsForward(self,clubs2,ThisDateRaw):
        ThisDate = dt(ThisDateRaw)
        sigma = 300.0
        #sigma = 0.01
        RSum,ISum = 0,0
        for u in range(self.TPos.shape[2]):
            for i in range(self.TPos.shape[0]):
                t1,t2 = self.TPos[i,0,u,0],self.TPos[i,-1,u,0]
                if 1000000000 not in [t1,t2] and max(t1,t2) < db.RankSeeds.shape[0]:
                    dU = self.U[t1] - self.U[t2]
                    dR = 1/db.RankSeeds[t1,t2]
                    RSum += dR
                    ISum += dU[0]*dR
            if ISum == 0:
                WbB = 30.0
            else:
                WbB = ISum/RSum
        for tlist in clubs2:
            for t in tlist:
                for u in range(self.TPos.shape[2]):
                    t1,t2 = self.TPos[t,0,u,0],self.TPos[t,-1,u,0]
                    if t1 != 1000000000:
                        LD1 = self.LastDate[t1]
                        self.MoveClubForward(t1,ThisDate)
                    if t2 != 1000000000:
                        LD2 = self.LastDate[t2]
                        self.MoveClubForward(t2,ThisDate)
                    if t1 != 1000000000 and t2 != 1000000000 and ThisDate!=LD1:
                        if None in [LD1,LD2]:
                            sigma2 = sigma
                            if LD1 == None:
                                self.U[t1] = self.U[t2]# + WbB
                            elif LD2 == None:
                                self.U[t2] = self.U[t1]# - WbB
                        else:
                            sigma2 = sigma*(sigma/(2*self.SigPerDay*(abs(ThisDate-LD1)).days)+1)
                        #self.G = self.AppendG(t1,t2,sigma2,self.G)
                        #self.AppendIG(t1,t2,WbB,sigma2)

    def AddGameI(self,t1,t2,res,r):
        U0 = 800.0/math.log(10)
        R0 = 250.0**2*(1+r)
        I0 =  U0/R0
        if self.Buckets[t1]==self.Buckets[t2]:
            #print(self.U[t1],self.U[t2])
            Ud = self.U[t1,0]-self.U[t2,0]
            #Ud = self.U2[t1]-self.U2[t2]
            W = 1/(1+10**(-Ud/400.0))
            if (W -0.5)**2 < 0.00001:
                Un = U0
            else:
                Un = Ud/(2.0*W-1.0)
        else:
            Ud = 0.0
            W = 1/(1+10**(-U0/400.0))
            Un = U0/(2.0*W-1.0)
        mu = Un*(2.0*res-1.0)
        sigma = abs(Un/I0)
        self.AppendIG(t1,t2,mu,sigma)
        #print(mu,sigma)
        #alpha = 0
        #beta = 0
        #R1 = 4000.0**2
        #R2 = sigma
        #I,G = AppendIGM(t1,t3,t2,t4,alpha,beta,mu,R1,R2,I,G)

    def LoadAll(self,DateRaw,Folder):
        Date = str(dt(DateRaw))
        Folder +='\\Data\\'
        self.G = numpy.load(Folder+str(Date)+'G.npy')
        self.U = numpy.load(Folder+str(Date)+'U.npy')
        self.UPrior = numpy.load(Folder+str(Date)+'Uext.npy')
        self.Buckets = numpy.load(Folder+str(Date)+'B.npy')
        with open(Folder+str(Date)+'LD.pkl', 'rb') as f:
            self.LastDate = pickle.load(f)
        with open(Folder+str(Date)+'IncrT.pkl', 'rb') as f:
            self.IncrTeamIDs = pickle.load(f)
            
    def SaveAll(self,DateRaw,Folder):
        Date = str(dt(DateRaw))
        Folder +='\\Data\\'
        numpy.save(Folder+str(Date)+'G.npy',self.G)
        numpy.save(Folder+str(Date)+'U.npy',self.U)
        numpy.save(Folder+str(Date)+'Uext.npy',self.UPrior)
        numpy.save(Folder+str(Date)+'B.npy',self.Buckets)
        with open(Folder+str(Date)+'LD.pkl', 'wb') as f:
            pickle.dump(self.LastDate,f)
        with open(Folder+str(Date)+'IncrT.pkl', 'wb') as f:
            pickle.dump(self.IncrTeamIDs,f)
       
    def RunOneRound(self,GamesToday):
        self.I = numpy.matmul(self.G,self.U) #5
        for g in GamesToday[['White','Black','Result']].values:
            g1 = self.TPos[int(g[0]),0,0,0]
            g2 = self.TPos[int(g[1]),-1,0,0]
            g1b= self.TPos[int(g[0]),0,-1,0] 
            g2b= self.TPos[int(g[1]),-1,-1,0] 
            self.AddGameI(g1,g2,g1b,g2b,g[2],self.DrawishR) #2
            self.Buckets = RefineBuckets(self.Buckets.copy(),g1,g2)
        #print(self.Buckets)
        self.U,db.RankSeeds = CompCalc(self.G,self.I,self.Buckets)
        #self.CovM = CovMatrix(db.RankSeeds)[:,:]

    def AddNextPlayers(self,GamesToday):
            CompEntrants = pandas.unique(GamesToday[["White"]].values.ravel('K'))
            for f in CompEntrants:
                if self.TPos[f,0,0,0] == 1000000000:
                    for u in range(self.TPos.shape[2]):
                        #print('Add',f,'W')
                        self.IncrTeamIDs.append([f,0,u,0])
                        self.TPos[f,0,u,0] = len(self.IncrTeamIDs)-1
                        self.LastDate[self.TPos[f,0,u,0]] = None
            CompEntrants = pandas.unique(GamesToday[["Black"]].values.ravel('K'))
                #print('Black',CompEntrants)
            for f in CompEntrants:
                if self.TPos[f,-1,0,0] == 1000000000:
                    for u in range(self.TPos.shape[2]):
                        #print('Add',f,'B')
                        self.IncrTeamIDs.append([f,-1,u,0])
                        self.TPos[f,-1,u,0] = len(self.IncrTeamIDs)-1
                        self.LastDate[self.TPos[f,-1,u,0]] = None
            numpy.set_printoptions(suppress=True)
            Today = GamesToday["Date"].min()
            Count = len(self.U)
            Mt = max(len(self.IncrTeamIDs),Count)
            if Mt > Count:
                Gn = numpy.zeros((Mt,Mt),dtype=floatDT) #
                Gn[:-(Mt-Count),:-(Mt-Count)] = self.G
                Un = numpy.zeros(shape=(Mt,1),dtype=floatDT, order='F')
                Un[:-(Mt-Count),:] = self.U
                UPriorn = numpy.nan * numpy.ones(shape=(Mt),dtype=floatDT, order='F')
                UPriorn[:-(Mt-Count)] = self.UPrior
                Bucketsn = numpy.arange(Mt)
                Bucketsn[:Count] = self.Buckets
                self.G,self.U,self.UPrior,self.Buckets = Gn,Un,UPriorn,Bucketsn
                Count = Mt
            self.I = numpy.matmul(self.G,self.U)
            self.MoveClubsForward([CompEntrants],Today)
            #self.U,db.RankSeeds = CompCalc(self.G,self.I,self.Buckets)

    def AddPrior(self):
        self.I2 = numpy.matmul(self.G,self.U) #5
        GG =[]
        for g in range(self.G.shape[0]-1):
            GG.append([0,g+1])
        self.G2 = self.G.copy()
        self.Buckets2 = self.Buckets.copy()
        for g in GG:
            g1 = g[0]
            g2 = g[1]
            UPr = self.UPrior[g2]
            if numpy.isnan(UPr):
                UPr = 2000.0
                Sig = 2*(250.0**2)
            else:
                Sig = 2*(150.0**2)
                #Sig = 0.01**2
            self.AddOnePrior(g1,g2,2881.0-UPr,Sig) #2
            self.Buckets2 = RefineBuckets(self.Buckets2,g1,g2)
        #print('Buck2',self.Buckets2)
        for t in range(self.TPos.shape[0]):
            for u in range(self.TPos.shape[2]):
                t1,t2 = self.TPos[t,0,u,0],self.TPos[t,-1,u,0]
                if t1 != 1000000000 and t2 != 1000000000:
                    self.AddOnePrior(t1,t2,25.0,45.0**2)
        self.U2,db.RankSeeds3 = CompCalc(self.G2,self.I2,self.Buckets2)

    def Uvalues(self,Today,aaa):
        self.CovM = CovMatrix(db.RankSeeds3)[:,:]
        rng = numpy.random.default_rng(10000)
        Usq = self.U2.squeeze()
        UsqEx = numpy.expand_dims(Usq,axis=0)
        ULiveR = rng.multivariate_normal(Usq,self.CovM,int((self.N-1)/2),method='cholesky')
        self.ULive = numpy.concatenate((UsqEx,ULiveR,2*UsqEx-ULiveR),axis=0)
        
    def run(self,N,Heute,Speichern,SigPerDay):
        self.N = N
        for c in self.Children:
            c.Stages = []
            c.Folder = '\\'+c.Name
            c.Folders = [c.Name]
            c.WriteStages(c.Folders)
            c.CountRounds([])
        if Heute < datetime.date(2020,6,1):
            Game1 = (self.Games[self.Games['Date']==self.Games['Date'].min()].values[0])
            #print(Game1[3],Game1[4])
            self.IncrTeamIDs = [[int(Game1[3]),0,0,0],[int(Game1[4]),1,0,0]]
            self.LastDate = {0:None,1:None}
            self.Buckets = numpy.arange(2)
            self.G = numpy.zeros((2,2),dtype=floatDT) #
            self.U = numpy.zeros(shape=(2,1),dtype=floatDT, order='F')
            self.UPrior = numpy.nan * numpy.ones(shape=(2),dtype=floatDT)
        else:
            self.LoadAll(str(Heute),self.currentdir)   
            self.Games = self.Games[self.Games["Date"] > str(Heute)]
        self.TPos = numpy.zeros(shape=(len(self.Players),2,1,1),dtype=intDT)
        self.TPos[:,:,:,:] = 1000000000
        for inc in range(len(self.IncrTeamIDs)):
            ARR = self.IncrTeamIDs[inc]
            a,b,c,d = ARR[0],ARR[1],ARR[2],ARR[3]
            self.TPos[a,-b,-c,-d] = inc
        self.TPosNP = numpy.array(self.TPos)
        self.Games.loc[self.Games["Round"].str.isnumeric()==False,"Round"] = 0
        self.Games["Round"] = pandas.to_numeric(self.Games["Round"])
        self.Games["SwapChecked"] = False
        WorldN = self.N
        self.N = 3**6
        for aaa in range(2):
        #for aaa in range(20000):
            Today = self.Games["Date"].min()
            GamesToday = self.Games[self.Games["Date"] == Today]
            Round = GamesToday["Round"].min()
            GamesToday = GamesToday[GamesToday["Round"] == Round]
            self.Games = self.Games[(self.Games["Date"] > Today) | (self.Games["Round"] > Round)]
            Tomorrow = self.Games["Date"].min()
            CompsToday = []
            MustCalc = []
            for index, row in GamesToday.iterrows(): #heutige ergebnisse auffüllen
                if row['Competition'] in self.Tournamentmap:
                    TTT = self.Tournamentmap[row['Competition']]
                    if TTT not in CompsToday:
                        CompsToday.append(TTT)
                    Retu = self.Tournamentmap[row['Competition']].DumpGame(row['Date'],row['Round'],row['White'],row['Black'],row['Result'],row['TimeControl'],row['Moves'])
                    if Retu == False:
                        print('Not filled:',TTT.Name,row['Date'],row['Round'],self.Field[row['White']].Name,self.Field[row['Black']].Name,row['Result'],row['TimeControl'])
                    elif row['Result'] in [0.0,0.5,1.0]:
                        MustCalc.append(TTT)
            GamesTodayCount = GamesToday[GamesToday['Exclude']==False]
            GamesTodayCount = GamesTodayCount[GamesTodayCount['Result'].notna()]
            if len(GamesTodayCount)>0: # Neue Elos.
                #print(GamesToday)
                self.RunOneRound(GamesTodayCount)
            CompsTomorrow = numpy.unique(self.Games[(self.Games["Date"] == Tomorrow)]['Competition'].values)
            GamesTomorrow = self.Games[(self.Games["Competition"].isin(CompsTomorrow))]
            self.AddNextPlayers(GamesTomorrow)
            for index, row in GamesTomorrow.iterrows():
                if row['TimeControl'] in ['10+3','10+5','15+5','15+10','25+5','25+10','Unknown']:
                    if row['WhiteElo'] != '':
                        self.UPrior[self.TPos[int(row['White']),0,0,0]] = float(row['WhiteElo'])
                    if row['BlackElo'] != '':
                        self.UPrior[self.TPos[int(row['Black']),-1,0,0]] = float(row['BlackElo'])
            if dt(Today) == Speichern:
                self.SaveAll(Today,self.currentdir)           
            self.IncrTeamIDsNP = numpy.array(self.IncrTeamIDs)
            self.TPosNP = numpy.array(self.TPos)
            print(Today,Round)
            #Ra = numpy.expand_dims(numpy.array(range(N)),axis=1)
            for nt in range(len(self.Children)): #ausgabe
                numpy.random.seed(nt)
                t = self.Children[nt]
                tOngoing = (t in CompsToday and dt(Today)>=t.StartDate and dt(Today)<=t.EndDate)
                tNext = (dt(Tomorrow)==t.StartDate and dt(Today)<t.StartDate)
                if tNext:
                    MustCalc.append(t)
                if tOngoing or tNext:   #dt(Tomorrow)>=t.StartDate and dt(Today)<=t.EndDate)
                    if tNext and not tOngoing:
                        Round = 0
                    Group = t.Children[0]
                    KO = t.Children[1]
                    if t in MustCalc:
                        self.AddPrior()
                        self.Uvalues(Today,aaa)
                        t.Ranking(self.DrawishR,int(Round),self)
                    if dt(Tomorrow)==t.StartDate and int(Round) == 1:
                        print('\n',t.Name,Today)
                if (pandas.isna(Tomorrow) or dt(Tomorrow)>t.EndDate) and dt(Today)<=t.EndDate:
                #if str(Round) == '14':
                    self.N = WorldN
                    self.AddPrior()
                    self.Uvalues(Today,aaa)
                    t.Ranking(self.DrawishR,int(Round),self)
                    t.WritePage(self,N)
            MustCalc = []
            for index, row in GamesTomorrow.iterrows():
                if self.Games.loc[index,"SwapChecked"] == False:
                    if row['Competition'] in self.Tournamentmap:
                        Retu = self.Tournamentmap[row['Competition']].SwapHAGame(row['Date'],row['Round'],row['White'],row['Black'],row['Result'],row['TimeControl'],row['Moves'],index,False)
                        if Retu != False:
                            self.Games.loc[index,"SwapChecked"] = True
                            if self.Tournamentmap[row['Competition']] not in MustCalc:
                                MustCalc.append(self.Tournamentmap[row['Competition']])
            for m in MustCalc:
                m.Ranking(self.DrawishR,int(Round),self)
            if pandas.isna(Tomorrow):
                break
            
    def AddNextPlayers2(self,CompEntrants):
            for f in CompEntrants:
                Col = ''
                if [f[0],Col] in self.IncrTeamIDs:
                    continue
                #if f[1]==1: Col = 'B'
                #elif f[1]==0: Col = 'W'
                self.IncrTeamIDs.append([f[0],Col])
                self.TPos[str(f[0])+'-'+Col] = len(self.IncrTeamIDs)-1
                #self.LastDate[len(self.IncrTeamIDs)-1] = None
            Count = len(self.U)
            Mt = max(len(self.IncrTeamIDs),Count)
            if Mt > Count:
                Gn = numpy.zeros((Mt,Mt),dtype=floatDT) #
                Gn[:-(Mt-Count),:-(Mt-Count)] = self.G
                Un = numpy.zeros(shape=(Mt,1),dtype=floatDT, order='F')
                Un[:-(Mt-Count),:] = self.U
                #UPriorn = numpy.nan * numpy.ones(shape=(Mt),dtype=floatDT, order='F')
                #UPriorn[:-(Mt-Count)] = self.UPrior
                Bucketsn = numpy.arange(Mt)
                Bucketsn[:Count] = self.Buckets
                self.G,self.U,self.Buckets = Gn,Un,Bucketsn
                Count = Mt
            self.I = numpy.matmul(self.G,self.U)
            #self.MoveClubsForward([CompEntrants],Today)
            #self.U,db.RankSeeds = CompCalc(self.G,self.I,self.Buckets)

    def MoveClubsForward2(self,LastDate,ThisDateRaw):
        ThisDate = dt(ThisDateRaw)
        for t1 in range(len(self.U)):
            self.MoveClubForward(t1,LastDate,ThisDate)

    def MoveClubForward(self,tea,LastDate,ThisDate): #Stern-Polygon transformation. Move club forward in time.
        if True: #self.LastDate[tea] not in [ThisDate,None]:
            GT = numpy.copy(self.G[tea,:])
            #print(type(self.G),self.G.shape)
            #print(type(GT),GT.shape)
            self.G[tea,:] = 0.0
            self.G[:,tea] = 0.0
            GT[tea] = numpy.divide(1.0,self.SigPerDay*(abs(ThisDate-LastDate)).days)
            SGa = numpy.sqrt(numpy.sum(GT))
            GT = numpy.divide(GT,SGa)        
            self.G += numpy.outer(GT,GT)
            self.G = ProperResMatrix(self.G)
        #self.LastDate[tea] = ThisDate

    def EliminatePlayer(self,tea):
        #print(tea)
        GT = numpy.copy(self.G[tea,:])
        SGa = numpy.sqrt(numpy.sum(GT)-GT[tea])
        if SGa > 0: GT = numpy.divide(GT,SGa)
        self.G += numpy.outer(GT,GT)
        self.G = numpy.delete(self.G,tea,0)
        self.G = numpy.delete(self.G,tea,1)
        self.G = ProperResMatrix(self.G)
        self.IncrTeamIDs = self.IncrTeamIDs[:tea]+self.IncrTeamIDs[tea+1:]
        #print('Vorher',self.Buckets)
        if tea in self.Buckets:
            #print('Es ist soweit')
            wh = numpy.where(self.Buckets==tea)[0]
            if wh.shape[0]>1:
                #print('Anpassung erfolgt!')
                self.Buckets[self.Buckets==tea] = wh[1]
        self.Buckets = numpy.delete(self.Buckets,tea,0)
        self.Buckets[self.Buckets>=tea]-=1      
        self.U = numpy.delete(self.U,tea,0)
        for tp in self.TPos:
            if self.TPos[tp] >= tea:
                self.TPos[tp] -= 1

class Point(Entity):
    __tablename__ = "point"
    id = Column(Integer, ForeignKey("entity.id"), primary_key=True)
    __mapper_args__ = {"polymorphic_identity": "Point"}
    Date = Column(Date)  
    RootSP = Column(Integer,ForeignKey("savepoint.id"))
    #rel_leaf_Elo = relationship("Elo", back_populates = "rel_root",cascade = "all, delete, delete-orphan" )
    def __init__(self,Name,World,date,RootSP):
        self.World = World
        self.Date = date
        if RootSP == None:
            self.RootSP = None
        else:
            self.RootSP = RootSP.id
        db.session.add(self)
        db.session.commit()
        Entity.__init__(self,Name)
        db.session.commit()
        if self.Date != None:
            for pl in range(1,len(self.World.IncrTeamIDs)):
                #print('WriteElo',self.World.IncrTeamIDs[pl])
                ID = self.World.IncrTeamIDs[pl][0]
                #print(self.World.IncrTeamIDs)
                Qu = db.Elo.insert().values(Point=self.id,PlayerID=ID,MatrSeed=pl,Bucket=int(self.World.Buckets[pl]),Categories='',Elo=self.World.U[pl])
                db.conn.execute(Qu)
            db.session.commit()



class SavePoint(Point):  ## Next: Clubelimination, Delete old calcs properly, 
    __tablename__ = "savepoint"
    id = Column(Integer, ForeignKey("point.id"), primary_key=True)
    __mapper_args__ = dict(polymorphic_identity="SavePoint",inherit_condition=(id == Point.id))
    Gstored = Column(LargeBinary)
    #rel_leaf = relationship("Point",order_by=Point.id, back_populates = "rel_root",foreign_keys=[id])
    def __init__(self,Name,World,Date,player=None,RootSP=None):
        self.World = World
        #print(self.id)
        #print(self.id)
        if player != None:
            self.World.U = numpy.zeros(shape=(2,1),dtype=floatDT, order='F')
            self.World.U[:,0] = self.World.GroundLevel
            self.World.G = numpy.zeros((2,2),dtype=floatDT) #actually infinity
            self.World.G[0,0]  = -1000000
            self.World.G[1,1]  = -1000000
            self.World.G[0,1]  = 1000000
            self.World.G[1,0]  = 1000000
            self.World.Rs = numpy.zeros((2,2),dtype=floatDT)
            self.World.CovM = numpy.ones((10,10),dtype=floatDT)
            self.World.IncrTeamIDs = [[0,''],[player.id,'']]
            self.World.Buckets = numpy.zeros(shape=(2),dtype=intDT)
            db.session.commit()
        self.Gstored = self.World.G.tobytes()
        #Delete all Savepoints between old SP and this SP.
        Point.__init__(self,Name,World,Date,RootSP)
        db.session.commit()
        #self.cascade() 
        #Get All Elo with this SP.
        
        #print('Elos',Elos)
        
    def loadpoint(self):
        Elos = db.conn.execute('SELECT PlayerID,MatrSeed,Elo,Bucket,Categories FROM Elo \
                            WHERE Point = '+str(self.id)).all()
        z = len(Elos)+1
        self.World.G = numpy.fromstring(self.Gstored,dtype=floatDT).reshape((z,z))
        #print(self.G)
        self.World.IncrTeamIDs = [0] * z
        self.World.Buckets = numpy.zeros(shape=(z),dtype=intDT)
        self.World.IncrTeamIDs[0] = [0,'']
        self.World.TPos = {}
        self.World.U = numpy.zeros(shape=(z,1),dtype=floatDT, order='F')
        self.World.U[0,0] = self.World.GroundLevel
        TPos = {}
        for e in range(z-1):
            #print(e)
            self.World.U[Elos[e][1],0] = Elos[e][2]
            self.World.Buckets[Elos[e][1]] = Elos[e][3]
            self.World.IncrTeamIDs[Elos[e][1]] = [Elos[e][0],Elos[e][4]]
            self.World.TPos[str(Elos[e][0])+'-'+Elos[e][4]] = Elos[e][1] 
        NextDate = db.conn.execute('SELECT P.ID, MAX(E.Enddate) \
            FROM Player P \
            JOIN FirstSeeds F ON P.ID = F.Player \
            JOIN Encounter E ON E.ID = F.Encounter \
            WHERE E.Enddate > \''+str(self.Date) +'\' \
            GROUP BY P.ID \
            ORDER BY MAX(E.Enddate) ASC LIMIT 1').all()
       # print('nextd',self.Date,NextDate[0][1])      
        LeavingPlayers = db.conn.execute('SELECT P.ID,MAX(E.Enddate) \
            FROM Player P \
            JOIN FirstSeeds F ON P.ID = F.Player \
            JOIN Encounter E ON E.ID = F.Encounter \
            GROUP BY P.ID \
            HAVING MAX(E.Enddate) = \''+str(self.Date)+'\'').all()  #+str(NextDate[0][1])+
        for l in LeavingPlayers:
            pass
            self.World.EliminatePlayer(self.World.TPos[str(l[0])+'-'])
        self.World.I = numpy.matmul(self.World.G,self.World.U)
        if len(NextDate) == 0:
            return None
        NewPlayers = db.conn.execute('SELECT P.ID \
            FROM Player P \
            JOIN FirstSeeds F ON P.ID = F.Player \
            JOIN Encounter E ON E.ID = F.Encounter \
            WHERE E.Startdate <= \''+str(NextDate[0][1])+'\' \
            GROUP BY P.ID \
            HAVING MIN(E.Startdate) > \''+str(self.Date)+'\'').all()
        #print(self.Date,NewPlayers)
        self.World.AddNextPlayers2(NewPlayers)
        HomeS = aliased(db.FirstSeeds)
        AwayS = aliased(db.FirstSeeds)
        self.GamesBetwSPs = db.session.query(Game)\
                       .join(HomeS) \
                       .join(AwayS) \
                       .add_columns(HomeS.c.player)\
                       .add_columns(AwayS.c.player)\
                       .filter(Game.Included == True,HomeS.c.spot == 0,AwayS.c.spot == 1, Game.EndDate > self.Date,Game.EndDate <= NextDate[0][1]).order_by(asc(Game.EndDate),asc(Game.Round)).all()
        return NextDate

    def cascade(self):
        NextDate = self.loadpoint()
        if NextDate == None:
            return None
        CurDat = self.Date
        for g in self.GamesBetwSPs:
            if dt(g[0].EndDate) > CurDat:
                if CurDat > self.Date:       
                    self.World.U,self.World.G = CalcNewU(self.World.G,self.World.I,self.World.Buckets)
                    self.World.I = numpy.matmul(self.World.G,self.World.U) #5
                self.World.MoveClubsForward2(CurDat,g[0].EndDate)
                self.World.I = numpy.matmul(self.World.G,self.World.U) #5
                CurDat = dt(g[0].EndDate)
            if g[0].Result == None:
                continue
            g1 = self.World.TPos[str(int(g[1]))+'-']
            g2 = self.World.TPos[str(int(g[2]))+'-']
            self.World.AddGameI(g1,g2,g[0].Result,self.World.DrawishR) #2
            self.World.Buckets = RefineBuckets(self.World.Buckets.copy(),g1,g2)
        self.World.U,self.World.G = CalcNewU(self.World.G,self.World.I,self.World.Buckets)
        return dt(NextDate[0][1])

    def cascade2(self):
        NextDate = self.loadpoint()
        if NextDate == None:
            NextDate = [[None,'2037-12-31']]
        StayingPlayers = db.conn.execute('SELECT DISTINCT P.ID \
            FROM Player P \
            JOIN FirstSeeds F ON P.ID = F.Player \
            JOIN Encounter E ON E.ID = F.Encounter \
            WHERE E.Included = 1 \
            AND E.Enddate > \''+str(self.Date)+'\'\
            AND E.Startdate <= \''+str(NextDate[0][1])+'\'').all()
        if len(StayingPlayers) == 0:
            return []
        for l in self.World.IncrTeamIDs.copy()[1:]:
            if (l[0],) not in StayingPlayers:
                self.World.EliminatePlayer(self.World.TPos[str(l[0])+'-'])
        CurRound = '-1'
        CurDat = dt(self.Date)
        #ToDel = db.session.query(Point).filter(Point.Date > self.Date,Point.Date <= NextDate[0][1]).all()
        #db.session.commit()
        #for todel in ToDel:
        #    db.session.flush()
        #    self.World.conn.execute('DELETE FROM Elo WHERE Point = '+str(todel.id))
        #    db.session.delete(todel)
        #print('ToDel',ToDel)
        self.World.I = numpy.matmul(self.World.G,self.World.U) #5
        NewPs = []        
        for g in self.GamesBetwSPs:
            if g[0].Result == None:
                continue
            if dt(g[0].EndDate) > CurDat:
                if CurRound != g[0].Round:       
                    self.World.U,self.World.Rs = CompCalc(self.World.G,self.World.I,self.World.Buckets)
                    #self.World.CovM = CovMatrix(self.World.Rs)[:,:]
                    self.World.I = numpy.matmul(self.World.G,self.World.U) #5
                    NewP = SimPoint('tttt',self.World,CurDat,CurRound,self)
                    NewPs.append(NewP)
                    NewP.Rstored = self.World.Rs.tobytes()
                self.World.MoveClubsForward2(CurDat,g[0].EndDate)
                CurDat = dt(g[0].EndDate)
                self.World.I = numpy.matmul(self.World.G,self.World.U) #5
                CurRound = g[0].Round
            g1 = self.World.TPos[str(int(g[1]))+'-']
            g2 = self.World.TPos[str(int(g[2]))+'-']
            self.World.AddGameI(g1,g2,g[0].Result,self.World.DrawishR) #2
            self.World.Buckets = RefineBuckets(self.World.Buckets.copy(),g1,g2)
            #CalcOneGame
        #print(self.World.G,self.World.I,self.World.Buckets)
        self.World.U,self.World.Rs = CompCalc(self.World.G,self.World.I,self.World.Buckets)
        self.World.I = numpy.matmul(self.World.G,self.World.U) #5
        self.World.CovM = CovMatrix(self.World.Rs)[:,:]
        return NewPs
        #print(self.World.U[:,0])
        return dt(NextDate[0][1])
        #Calc all games. Always store Elo, delete existing.
        #white True:

Point.rel_root = relationship("SavePoint",foreign_keys=[Point.RootSP],single_parent=True,cascade = "all, delete, delete-orphan")

class SimPoint(Point):
    __tablename__ = "simpoint"
    id = Column(Integer, ForeignKey("point.id"), primary_key=True)
    __mapper_args__ = dict(polymorphic_identity="SimPoint",inherit_condition=(id == Point.id))
    Round = Column(Integer)
    Rstored = Column(LargeBinary)
    N = Column(Integer)
    #rel_leaf_Enc = relationship("SimEncounter", back_populates = "rel_root",cascade = "all, delete, delete-orphan" )
    def __init__(self,Name,World,Date,Round,RootSP):
        self.Round = Round
        Point.__init__(self,Name,World,Date,RootSP)
        if Date == None:
            self.N = 1
        else:
            self.N = 3**6

    def PrepSim(self):
        Elos = db.conn.execute('SELECT PlayerID,MatrSeed,Elo,Bucket,Categories FROM Elo \
                            WHERE Point = '+str(self.id)).all()
        z = len(Elos)
        db.RankSeeds = numpy.asmatrix(numpy.fromstring(self.Rstored,dtype=floatDT).reshape((z+1,z+1)))
        self.CovM = CovMatrix(db.RankSeeds)[1:,1:]
        self.U = numpy.zeros(shape=(z,1),dtype=floatDT, order='F')
        for e in range(z):
            self.U[Elos[e][1]-1,0] = Elos[e][2]
        rng = numpy.random.default_rng(10000)      
        UsqEx = numpy.expand_dims(self.U[:,0],axis=0)
        ULiveR64 = rng.multivariate_normal(self.U[:,0],self.CovM,int((self.N-1)/2),method='cholesky')
        ULiveR = numpy.float32(ULiveR64)
        #self.EL = db.Base.metadata.tables['Elo']
        for e in range(z):
            db.session.commit()
            Q = db.Elo.update().values(SimmedElo=ULiveR[:,e].tobytes()).where(db.Elo.c.MatrSeed==e+1,db.Elo.c.Point==self.id)
            db.conn.execute(Q)
            #self.World.conn.execute('Update Elo SET SimmedElo = x\''+ULiveR[:,e].tobytes().hex()+'\'\
            #                WHERE MatrSeed = '+str(e+1)+' AND Point = '+str(self.id))
        db.session.commit()

class Player(Entity):
    __tablename__ = "player"
    id = Column(Integer, ForeignKey("entity.id"), primary_key=True)
    __mapper_args__ = {"polymorphic_identity": "player"}
    def __init__(self,Name):
        Entity.__init__(self,Name)
    def __lt__(self,other):
        return self.Name<other.Name
  

class Encounter(Entity):
    __tablename__ = "encounter"
    id = Column(Integer, ForeignKey("entity.id"), primary_key=True)
    StartDate = Column(Date)
    EndDate = Column(Date)
    Included = Column(Integer)
    __mapper_args__ = {"polymorphic_identity": "encounter"}
    StartDate = Column(String)
    EndDate = Column(String)
    Priority = Column(Integer)
    def __init__(self,Seeding,Date,EndDate,Name):
        Entity.__init__(self,Name)
        db.session.add(self)
        self.StartDate = Date
        self.EndDate = EndDate
        self.SeedTrigger = []
        self.Seeding = Seeding
        self.Seeds = {}
        #self.PW = numpy.zeros(shape=(1,len(self.Seeds),1))-1
        self.Odds = {}
        self.Sims = {}
        #print('Seeding',self.Seeding)
        for s in Seeding:
            Ar = Seeding[s]
            db.session.commit()
            if Ar[1] == 'First':
                self.Seeds[s] = Ar[2]
                Q = db.FirstSeeds.insert().values(encounter=self.id,spot=s,player=Ar[2].id)
                db.conn.execute(Q)
            elif Ar[1] == 'Ranks':
                Ar[0].SeedTrigger.append([self,Ar[2],s])
                #print('RankSeeds',self.id,s,Ar[0].id,Ar[2])
                Q = db.RankSeeds.insert().values(encounter=self.id,spot=s,rootencounter=Ar[0].id,rank=Ar[2])
                db.conn.execute(Q)
            db.session.commit()
               
    def __lt__(self,other):
        return self.EndDate<other.EndDate
    
    def PrintRoundTree(self):
        Str = ''
        for i in self.RoundTree:
            Str+=str(i)+'.'
        self.RoundTreeStr = Str[:-1]

    def PrepArrays(self,SimP):
        #Check first in RankSeeds
        #Check then in FirstSeeds
        if SimP.id == db.Reality.id:
            SimPid = '= '+str(SimP.id)
            ElosPl = []
            Elos = db.conn.execute('SELECT S.spot,S.player,NULL,NULL \
                            FROM FirstSeeds S  \
                            WHERE S.Encounter = '+str(self.id)).all()
        else:
            SimPid = '= '+str(SimP.id)
            ElosPl = db.session.query(db.Elo).filter(db.Elo.c.Point == SimP.id).all()
            Elos = db.conn.execute('SELECT S.spot,P.PlayerID,P.Elo,P.SimmedElo \
                            FROM Elo P JOIN FirstSeeds S ON P.PlayerID = S.player \
                            WHERE P.Point = '+str(SimP.id)+' AND S.Encounter = '+str(self.id)).all()
        RkMax = db.conn.execute('SELECT MAX(spot) FROM RankSeeds WHERE RankSeeds.Encounter = '+str(self.id)).one()
        FSMax = db.conn.execute('SELECT MAX(spot) FROM FirstSeeds WHERE FirstSeeds.Encounter = ' + str(self.id)).one()
        if RkMax[0] == None: LenS = FSMax[0]+1
        elif FSMax[0] == None: LenS = RkMax[0]+1
        else: LenS = max(FSMax[0],RkMax[0])+1
        self.Seeds = {}
        if SimP != None:
            self.ULive = numpy.zeros(shape=(SimP.N, LenS), dtype=floatDT)
        Taken = []
        RankPlNone = db.conn.execute('SELECT S.spot, ER.IDs \
                                FROM RankSeeds S JOIN EncounterRanks ER ON S.rootencounter = ER.Encounter AND ER.Ranks = S.rank \
                                WHERE S.encounter = '+str(self.id)+' AND ER.SimPoint == '+str(db.Reality.id)).all()
        RankPlSimP = db.conn.execute('SELECT S.spot, ER.IDs \
                                FROM RankSeeds S JOIN EncounterRanks ER ON S.rootencounter = ER.Encounter AND ER.Ranks = S.rank \
                                WHERE S.encounter = '+str(self.id)+' AND ER.SimPoint '+SimPid).all()
        for RankPl in [RankPlSimP,RankPlNone]:
            for rp in RankPl:
                if rp[0] in Taken: continue
                self.Seeds[rp[0]] = numpy.fromstring(rp[1],dtype=intDT)
                for e in ElosPl:
                    Ind = (self.Seeds[rp[0]]==e[1])
                    el = numpy.fromstring(e[6],dtype=floatDT)
                    elc = numpy.concatenate((numpy.array([e[5]]),el,2*e[5]-el),axis=0).astype(floatDT)
                    if len(Ind) == 1:
                        if Ind[0] == True:
                            self.ULive[:, rp[0]] = elc
                        elif Ind[0] == False: pass
                    else:
                        self.ULive[Ind,rp[0]] = elc[Ind]
                Taken.append(rp[0])
        #if(len(Elos)) == 0:
        #    self.ULive = None
        #    return
        z = len(Elos)
        for zz in range(z):
            if Elos[zz][0] in Taken: continue
            self.Seeds[Elos[zz][0]] = numpy.array([Elos[zz][1]])
            if SimP == None:
                pass
            elif Elos[zz][3] == None:
                self.ULive[:,zz] = numpy.array(numpy.nan)
            else:
                HalfU = numpy.fromstring(Elos[zz][3],dtype=floatDT)
                EinU = numpy.array([numpy.float32(Elos[zz][2])])
                #print(HalfU.shape,EinU.shape,EinU)
                self.ULive[:,zz] = numpy.concatenate((EinU,HalfU,2*EinU-HalfU),axis=0)
        #print(self.ULive)

class Game(Encounter):
    __tablename__ = "game"
    id = Column(Integer, ForeignKey("encounter.id"), primary_key=True)
    Round = Column(String)
    #Include  = Column(Boolean)
    __mapper_args__ = {"polymorphic_identity": "game"}
    def __init__(self,Seeding,Date,EndDate,World,Round,Name,NSc=1):
        Encounter.__init__(self,Seeding,Date,EndDate,Name)
        self.World = World
        self.Round = Round #0
        self.RankNo = numpy.ones(shape=(1,2),dtype=intDT)
        self.Dumped = False
        self.Swapchecked = False
        self.IfR = {}
        self.Include = True
        self.Priority = 0
        for s in [0,1]:
            if Seeding[s][1] == 'Ranks':
                self.Priority = max(self.Priority,Seeding[s][0].Priority+1)

    def reestablish(self,SimP):
        pass

    def Simulate(self,ULive,TPosNP,DrawishR,Rand,Extra):
        N = ULive.shape[0]
        Wh = TPosNP[self.Seeds[0].astype(int),0,0,0][:N]
        Wa = TPosNP[self.Seeds[1].astype(int),-1,0,0][:N]
        #print(Wh,self.Seeds[0].astype(int),Wa,self.Seeds[1].astype(int))
        Elodiff = numpy.expand_dims(ULive[numpy.arange(N),Wh.astype(int)]-ULive[numpy.arange(N),Wa.astype(int)],axis=1)
        self.Elo1 = numpy.average(ULive[numpy.arange(N),Wh.astype(int)])
        self.Elo2 = numpy.average(ULive[numpy.arange(N),Wa.astype(int)])
        Eloproba = 1/(1+10**(-Elodiff/400.0))
        Winpr = Eloproba * (Eloproba+DrawishR-DrawishR*Eloproba)
        Drawpr = 2*(Eloproba-Winpr)
        NoLossPr = Winpr + Drawpr
        #Rand = numpy.expand_dims(numpy.random.uniform(0.0,1.0,N),axis=1)
        self.Score = numpy.zeros(shape=(N+Extra,2,1),dtype=floatDT) #* None
        self.Score[:N,0,0] = 0.5*numpy.squeeze(1+(Winpr>Rand)-(Rand>NoLossPr))
        self.Score[:N,1,0] = 1- self.Score[:N,0,0]
        self.PW = self.Score
        #print('Score',self.Score[:,0,0])
        self.CalcOdds(N)

    def Simulate2(self,Smpnt,World):
        self.PrepArrays(Smpnt)
        if Smpnt.id == db.Reality.id:
            self.Score = numpy.zeros(shape=(1,2,1),dtype=floatDT) #* None
            if self.Result == None:
                self.Score[0, :, 0] = 0.0
            else:
                self.Score[0,0,0] = self.Result
                self.Score[0,1,0] = 1-self.Result
            Qu = db.SimEncounter.insert().prefix_with("OR REPLACE").values(SimPoint=db.Reality.id, Encounter=self.id,Score=self.Score[:, :, 0].tobytes())

        elif Smpnt.Date >= db.dt(self.EndDate):
            self.Score = numpy.zeros(shape=(1,2,1),dtype=floatDT) #* None
            if self.Result == None:
                self.Score[0, :, 0] = 0.0
            else:
                self.Score[0,0,0] = self.Result
                self.Score[0,1,0] = 1-self.Result
            Qu = db.SimEncounter.insert().prefix_with("OR REPLACE").values(SimPoint=db.Reality.id, Encounter=self.id,Score=self.Score[:, :, 0].tobytes())
        else:
            DrawishR = 0.8
            Extra = 0
            #print('Ushape',self.ULive.shape)
            if type(self.ULive) == type(None):
                Eloproba = 0.5
            elif self.ULive.shape[1] < 2:
                Eloproba = 0.5
            else:
                Elodiff = self.ULive[:,0] - self.ULive[:,1]
                Eloproba = 1/(1+10**(-Elodiff/400.0))
            #print(Eloproba)
            Winpr = Eloproba * (Eloproba+DrawishR-DrawishR*Eloproba)
            Drawpr = 2*(Eloproba-Winpr)
            NoLossPr = Winpr + Drawpr
            Rand = numpy.random.default_rng(self.id).uniform(0.0,1.0,Smpnt.N)
            self.Score = numpy.zeros(shape=(Smpnt.N+Extra,2,1),dtype=floatDT) #* None
            self.Score[:Smpnt.N,0,0] = 0.5*numpy.squeeze(1+(Winpr>Rand)-(Rand>NoLossPr))
            self.Score[:Smpnt.N,1,0] = 1- self.Score[:Smpnt.N,0,0]
        #print(len(self.Score[:Smpnt.N,:,0].tobytes()))
        db.session.commit()
        Qu = db.SimEncounter.insert().prefix_with("OR REPLACE").values(SimPoint=Smpnt.id,Encounter=self.id,Score=self.Score[:,:,0].tobytes())
        #db.session.commit()
        db.conn.execute(Qu)
        db.session.commit()
        self.PW = self.Score
        
        
##        Sds = self.World.conn.execute('SELECT Spot,Player FROM FirstSeeds \
##                    WHERE Encounter = '+str(self.id)).all()
##        self.Seeds = {}
##        for s in Sds:
##            self.Seeds[s[0]]=s[1]
##        print(self.Seeds)
##        Elos1 = self.World.conn.execute('SELECT PlayerID,Elo,SimmedElo FROM Elo \
##                    WHERE PlayerID = '+str(self.Seeds[0])+' AND Point = '+str(Smpnt.id)).all()
##        Elos1b = numpy.fromstring(Elos1[0][2],dtype=floatDT)
##        Elos2 = self.World.conn.execute('SELECT PlayerID,Elo,SimmedElo FROM Elo \
##                    WHERE PlayerID = '+str(self.Seeds[1])+' AND Point = '+str(Smpnt.id)).all()
##        Elos2b = numpy.fromstring(Elos2[0][2],dtype=floatDT)
##        #

    def IsGame(self):
        return True

    def GetGameList(self):
        return [self]

    def WritePage(self,World,N):
        pass

    def CalcOdds(self,N):
        self.Odds['W'] = numpy.count_nonzero(self.Score[:N,0,0] == 1.0)/N
        self.Odds['D'] = numpy.count_nonzero(self.Score[:N,0,0] == 0.5)/N
        self.Odds['L'] = numpy.count_nonzero(self.Score[:N,0,0] == 0.0)/N

    def Ranking(self,SP,DrawishR,Round,World,Extra):
        N = World.ULive.shape[0]
        Rand = numpy.expand_dims(numpy.random.uniform(0.0,1.0,N),axis=1)
        if self.Played == False:
            self.Simulate(World.ULive,World.TPosNP,DrawishR,Rand,Extra)

    def BuildRanks(self,SimPid):
        pass


##    def GetLatestStage(self):
##        if self.Dumped == True:
##            return self.Round
##        else:
##            return 0

class Set(Encounter):
    __tablename__ = "set"
    id = Column(Integer, ForeignKey("encounter.id"), primary_key=True)
    Type = Column(String)
    TiebreakersStr = Column(String)
    HMatrixS = Column(LargeBinary)
    AMatrixS = Column(LargeBinary)
    __mapper_args__ = {"polymorphic_identity": "set"}
    def __init__(self,Type,Seeding,Date,EndDate,Name,World):
        self.World = World
        Encounter.__init__(self,Seeding,Date,EndDate,Name)
        self.Children = []
        self.Type = Type
        self.Rounds = 0
        self.RoundsArray = []
        self.Progress = 0
        self.FullStage = Name
                
        #Qu = db.SimGame.insert().values(SimPoint=Smpnt.id,Game=self.id,Score=self.Score[:N,0,0].tobytes())

    def fill(self):
        self.TiebreakersStr = str(self.TieBreakers)
        try:
            self.HMatrixS = self.HMatrix.tobytes()
            self.AMatrixS = self.AMatrix.tobytes()
        except:
            pass
        db.session.commit()
        self.Priority = 1
        for c in range(len(self.Children)):
            self.Priority = max(self.Priority,self.Children[c].Priority+1)
            print(self.id,self.Name,c,self.Children[c].Name)
            db.session.commit()
            Qu = db.SetStructure.insert().values(parent=self.id,child=self.Children[c].id,spot=c)
            db.conn.execute(Qu)
            db.session.commit()
        self.Simulate2(db.Reality,0)

    def BuildRanks(self,SimPid):
        SE = db.session.query(db.EncounterRanks).filter(db.EncounterRanks.c.Encounter==self.id,
            db.EncounterRanks.c.SimPoint == SimPid).all()
        if len(SE) > 0:
            N = numpy.fromstring(SE[0][3],dtype=intDT).shape[0]
            self.Ranks = numpy.zeros(shape=(N,len(SE)),dtype=intDT)
            for se in SE:
                self.Ranks[:,se[2]] = numpy.fromstring(se[3],dtype=intDT)

    def reestablish(self,SimP):
        print('doing my thing')
        self.Children = []
        A = db.session.query(db.SetStructure).filter(db.SetStructure.c.parent==self.id).order_by(asc(db.SetStructure.c.spot)).all()
        for a in A:
            c1 = db.session.query(Encounter).filter(Encounter.id==a[1]).all()[0]
            #c1.PrepArrays(SimP)
            try:
                SE = db.session.query(db.SimEncounter).filter(db.SimEncounter.c.SimPoint == SimP.id,
                                                                  db.SimEncounter.c.Encounter == a[1]).one()
                c1.PrepArrays(SimP)
            except:
                SE = db.session.query(db.SimEncounter).filter(db.SimEncounter.c.SimPoint == db.Reality.id,
                                                                  db.SimEncounter.c.Encounter == a[1]).one()
                c1.PrepArrays(db.Reality)
            c1Score = numpy.fromstring(SE[2],dtype=floatDT)
            N = int(c1Score.shape[0]/len(c1.Seeds))
            c1.Score = c1Score.reshape(N,len(c1.Seeds),1)
            #c1.reestablish()
            self.Children.append(c1)
        if self.HMatrixS != None:
            self.HMatrix = numpy.expand_dims(numpy.fromstring(self.HMatrixS,dtype=intDT),0)
        if self.AMatrixS != None:
            self.AMatrix = numpy.expand_dims(numpy.fromstring(self.AMatrixS,dtype=intDT),0)
        self.Tiebreakers = eval(self.TiebreakersStr)
        #print(self.Children)
        
    def IsGame(self):
        return False

    def GetGameList(self):
        GameList = []
        for c in self.Children:
            GameList += c.GetGameList()
        return GameList
    
    def ChiBerg(self,P,S):
        HMatrix,AMatrix,Rounds = Berger(P,S)
        self.HMatrix = HMatrix
        self.AMatrix = AMatrix
        self.Rounds = (P-1)*S
        return Rounds

##    def GetLatestStage(self):
##        Stage = 0
##        for c in self.Children:
##            if c.GetLatestStage() != 0:
##                Stage = c.GetLatestStage()
##        return Stage

    def WriteStages(self,UpFolders=[]):
        self.GamesUnder = []
        for c in range(len(self.Children)):
            chi = self.Children[c]
            chi.Folders = UpFolders+[chi.Name]
            if chi.IsGame() == True:
                #chi.Stages = UpStages+[chi.Round]
                self.GamesUnder.append(chi)
            else:
                #chi.Stages = UpStages+[c+1]
                self.GamesUnder += chi.WriteStages(chi.Folders)
            chi.Folder = ''
            chi.FullStage = ''
            for f in chi.Folders:
                chi.Folder+= '\\'+f
                chi.FullStage+= f + ' | '
        return self.GamesUnder

    def CalcOdds(self,PA,PB,DistB,Ns):
        P = len(self.Seeds)
        #print('Ns',Ns)
        #print(self.Name,'Pts',self.PW.shape)
        #Ns = slice(0,-1)
        #print('Name',self.Name,DistB.Name)
        #print('RankNo',self.RankNo.shape,DistB.RankNo.shape)
        #print('Ns',Ns)
        if self.PW.shape[0] == 1:
            Ns = [0]
        Pts = self.PW[Ns,:,0]
        PtsB = numpy.expand_dims(DistB.PW[Ns,0,0],axis=1)
        M1 = {'Rank':numpy.arange(1,P+1),'Pts':numpy.unique(Pts),'RankB':numpy.arange(1,P+1),'PtsB':numpy.unique(PtsB)}
        M2 = {'Rank':self.RankNo[Ns,:],'Pts':Pts,'RankB':DistB.RankNo[Ns,:],'PtsB':PtsB}
        MatA = (numpy.expand_dims(M1[PA],axis=(0,2,3)) == M2[PA])
        MatB = (numpy.expand_dims(M1[PB],axis=(1,2,3)) == M2[PB])
        ABAnd = numpy.logical_and(MatA,MatB)
        Q = int(P/2)
        ABAndQ = ABAnd[:,:Q,:,:]
        ABAndnQ= ABAnd[:,Q:,:,:]
        if DistB.IsGame() == True:
            NecSuf = numpy.stack((numpy.any(ABAnd[:,:Q,:,:],axis=(1,2)),numpy.any(ABAnd[:,Q:,:,:],axis=(1,2))),axis=1)
            Impossible = ~numpy.any(ABAndQ,axis=(0,1,2))
            Certain = ~numpy.any(ABAndnQ,axis=(0,1,2))
            NecessaryN = ~numpy.any(ABAnd[:,:Q,:,:],axis=(1,2)) * numpy.expand_dims(~Impossible,axis=0)
            Sufficient = ~numpy.any(ABAnd[:,Q:,:,:],axis=(1,2)) * numpy.expand_dims(~Certain,axis=0)
            #print('Impossible',numpy.nonzero(Impossible))
            #print('Certain',numpy.nonzero(Certain))
            #print('NecessaryN',numpy.nonzero(NecessaryN))
            #print('Sufficient',numpy.nonzero(Sufficient))
        #print(self.PW.shape,PB,MatA.dtype,MatB.dtype,MatA.shape,MatB.shape)
        #return M1[PA],M1[PB],numpy.count_nonzero(MatA*MatB,axis=2)
        return M1[PA],M1[PB],numpy.count_nonzero(ABAnd,axis=2)

    def RecurProb(yQ,DecGms,Arr,Sli):
        DecGms[Arr]
        R1 = Sli * self.GamesUnder[DecGms[len(Arr)]].Score[:N,0,0] == 1.0
        R2 = Sli * self.GamesUnder[DecGms[len(Arr)]].Score[:N,0,0] == 0.5
        R3 = Sli * self.GamesUnder[DecGms[len(Arr)]].Score[:N,0,0] == 0.0
        if len(numpy.unique(yQ*R1)) == 1:
            pass
            
        E1,F1 = RecurProb(DecGms,Arr.copy().append(0),R1)
        E2,F2 = RecurProb(DecGms,Arr.copy().append(1),R2)
        E3,F3 = RecurProb(DecGms,Arr.copy().append(2),R3)
        return [E1,E2,E3],[F1,F2,F3]

    def Permut(self,p,World):
        if self.Name == 'Prelims':
            rQ = 8
            yQ = self.RankNo[:World.N,p] <= rQ
            nQ = self.RankNo[:World.N,p] > rQ
            Q = [numpy.sum(yQ)/World.N,numpy.sum(nQ)/World.N]
            QEnt =  entropy(Q,base=2)
            #Res = numpy.zeros(shape=(World.N,len(self.GamesUnder)),dtype=)
            TransEnts = numpy.zeros(shape=(len(self.GamesUnder)))
            for gi in range(len(self.GamesUnder)):
                g = self.GamesUnder[gi]
                if len(g.Score) > 1:
                    if self.PrioOne == 'To win':
                        PrRk = [1,1]
                    elif self.PrioOne == 'To qualify':
                        PrRk = [1,self.Spots]
                    #EntOd = g.IfR[self]
                    EntOd = self.CalcOdds('Rank','PtsB',g,slice(0,-1))[2]
                    EntOd2 = numpy.sum(EntOd[:,PrRk[0]-1:PrRk[1],p],axis=1)/ World.N
                    EntOd2b = numpy.sum(EntOd[:,PrRk[1]:,p],axis=1)/ World.N
                    EntOd3 = numpy.stack((EntOd2,EntOd2b),axis=0)
                    #print(EntOd3)
                    GameEnt = entropy([g.Odds['W'],g.Odds['D'],g.Odds['L']],base=2)
                    BothEnt = entropy(EntOd3.flatten(),base=2)
                    TransEnt = GameEnt+QEnt-BothEnt
                    TransEnts[gi] = TransEnt
            DecGms = numpy.flip(numpy.argsort(TransEnts))[:5]
            #print(DecGms[:10],TransEnts[DecGms[:10]])
            toQ = []
            N = World.N
            Permuts = numpy.ones(shape=(N,3,3,3,3,3),dtype=bool)
            print(self.GamesUnder[DecGms[1]].Score[:N,0,0].shape)
            Permuts[:,0,:,:,:,:] *= numpy.expand_dims((self.GamesUnder[DecGms[0]].Score[:N,0,0] == 1.0),axis=(1,2,3,4))
            Permuts[:,1,:,:,:,:] *= numpy.expand_dims((self.GamesUnder[DecGms[0]].Score[:N,0,0] == 0.5),axis=(1,2,3,4))
            Permuts[:,2,:,:,:,:] *= numpy.expand_dims((self.GamesUnder[DecGms[0]].Score[:N,0,0] == 0.0),axis=(1,2,3,4))
            Permuts[:,:,0,:,:,:] *= numpy.expand_dims((self.GamesUnder[DecGms[1]].Score[:N,0,0] == 1.0),axis=(1,2,3,4))
            Permuts[:,:,1,:,:,:] *= numpy.expand_dims((self.GamesUnder[DecGms[1]].Score[:N,0,0] == 0.5),axis=(1,2,3,4))
            Permuts[:,:,2,:,:,:] *= numpy.expand_dims((self.GamesUnder[DecGms[1]].Score[:N,0,0] == 0.0),axis=(1,2,3,4))
            Permuts[:,:,:,0,:,:] *= numpy.expand_dims((self.GamesUnder[DecGms[2]].Score[:N,0,0] == 1.0),axis=(1,2,3,4))
            Permuts[:,:,:,1,:,:] *= numpy.expand_dims((self.GamesUnder[DecGms[2]].Score[:N,0,0] == 0.5),axis=(1,2,3,4))
            Permuts[:,:,:,2,:,:] *= numpy.expand_dims((self.GamesUnder[DecGms[2]].Score[:N,0,0] == 0.0),axis=(1,2,3,4))
            Permuts[:,:,:,:,0,:] *= numpy.expand_dims((self.GamesUnder[DecGms[3]].Score[:N,0,0] == 1.0),axis=(1,2,3,4))
            Permuts[:,:,:,:,1,:] *= numpy.expand_dims((self.GamesUnder[DecGms[3]].Score[:N,0,0] == 0.5),axis=(1,2,3,4))
            Permuts[:,:,:,:,2,:] *= numpy.expand_dims((self.GamesUnder[DecGms[3]].Score[:N,0,0] == 0.0),axis=(1,2,3,4))
            Permuts[:,:,:,:,:,0] *= numpy.expand_dims((self.GamesUnder[DecGms[4]].Score[:N,0,0] == 1.0),axis=(1,2,3,4))
            Permuts[:,:,:,:,:,1] *= numpy.expand_dims((self.GamesUnder[DecGms[4]].Score[:N,0,0] == 0.5),axis=(1,2,3,4))
            Permuts[:,:,:,:,:,2] *= numpy.expand_dims((self.GamesUnder[DecGms[4]].Score[:N,0,0] == 0.0),axis=(1,2,3,4))
            PermutsToQ = Permuts * numpy.expand_dims(yQ,axis=(1,2,3,4,5))
            PermutsSum = numpy.count_nonzero(PermutsToQ,axis=0) / numpy.count_nonzero(Permuts,axis=0)
            #print(numpy.count_nonzero(PermutsToQ,axis=0))
            #print(numpy.count_nonzero(Permuts,axis=0))
            #print(PermutsSum)
            return PermutsSum,DecGms

##    def Necessary(self,MaxRank):
##        A = numpy.where(self.RankNo[:,0]<MaxRank)
##        print(A)

    def SpecialRanks(self):
        pass

    def FromIDToSeed(self,Ranks):
        N,P = Ranks.shape
        NRanks = numpy.zeros(shape=(N,P),dtype=intDT)
        for p in range(P):
            NRanks += p * (Ranks == numpy.expand_dims(self.Seeds[p],axis=1))
        return NRanks

    def FromRankNoToID(self,RankNo):
        N,P = RankNo.shape
        NRanks2 = numpy.zeros(shape=(N,P),dtype=intDT)
        SeedsMem = numpy.zeros(shape=(N,P),dtype=intDT)
        for s in range(P):
            SeedsMem[:,s] = self.Seeds[s]
        Ra = numpy.expand_dims(numpy.array(range(N)),axis=1)
        NRanks2[Ra,RankNo-1] = SeedsMem[:,:]
        return NRanks2

    def WinLoseAll(self,n,s,win):
        for c in self.Children:
            if c.IsGame() == True:
                if numpy.sum(c.Score[n,:,0],axis=1) != 1.0:
                    if c.Seeds[0] == s:
                        c.Score[n,0,0] = win
                        c.Score[n,1,0] = 1.0-win
                    if c.Seeds[1] == s:
                        c.Score[n,1,0] = win
                        c.Score[n,0,0] = 1.0-win
            else:
                c.WinLoseAll(n,s,win)
        
    def InTheMix(self,n,s,win):
        for c in self.Children:
            if c.IsGame() == True:
                if numpy.sum(c.Score[n,:,0],axis=1) != 1.0:
                    c.Score[n,:,0] = win
        

    def DefaultResArrays(self,N,Extra):
        LC = len(self.Children)
        maxN = N + Extra
        ResultArrH = numpy.zeros(shape=(N,LC,1),dtype=floatDT)
        ResultArrA = numpy.zeros(shape=(N,LC,1),dtype=floatDT)
        for chi in range(LC):
            chiPts = self.Children[chi].Score
            ResultArrH[:,chi,0] = chiPts[:,0,0]
            ResultArrA[:,chi,0] = chiPts[:,1,0]
        ResultsArr = ResultArrH - ResultArrA
        return ResultArrH,ResultArrA,ResultsArr

    def ResArrays(self,N,Extra):
        return DefaultResArrays(self,N,Extra)        

    def Simulate2(self,SP,World):
        self.PrepArrays(SP)
        if SP == None:
            self.N = 1
            Spid = db.Reality.id
        else:
            self.N = SP.N
            Spid = SP.id
        self.reestablish(SP)
        P = len(self.Seeds)
        Extra = 1
        maxN = self.N# + Extra
        ResultArrH,ResultArrA,ResultsArr = self.ResArrays(self.N,Extra,SP)
        self.ResultsArr = ResultsArr
        if self.Tiebreakers != []:
            self.PW = PointsPT(self.HMatrix,ResultArrH,self.AMatrix,ResultArrA,P)
            Ra = numpy.expand_dims(numpy.array(range(maxN)),axis=1)
            self.RankNo = numpy.ones(shape=(maxN,P),dtype=intDT)
            self.NsToCalc = numpy.arange(maxN)
            for tb in self.Tiebreakers:
                if tb == 'Points':
                    Pts = self.PW[:,:,0]
                elif tb == 'SubPoints':
                    Pts = self.PW[:,:,1]
                    #print('2',Pts,numpy.sum(Pts,axis=1))
                elif tb == 'DirectEncounter':
                    OnSamePoints = numpy.nonzero(self.RankNo[Ra,self.HMatrix] == self.RankNo[Ra,self.AMatrix])
                    ResultArrHDE = numpy.zeros(shape=ResultArrH.shape,dtype=floatDT)
                    ResultArrADE = numpy.zeros(shape=ResultArrA.shape,dtype=floatDT)
                    ResultArrHDE[OnSamePoints[0],OnSamePoints[1],:] = ResultArrH[OnSamePoints[0],OnSamePoints[1],:]
                    ResultArrADE[OnSamePoints[0],OnSamePoints[1],:] = ResultArrA[OnSamePoints[0],OnSamePoints[1],:]
                    PtsD = PointsPT(self.HMatrix,ResultArrHDE,self.AMatrix,ResultArrADE,P)
                    Pts = PtsD[:,:,0]
                elif tb == 'Wins':
                    PtsD = PointsPT(self.HMatrix,ResultsArr>0,self.AMatrix,ResultsArr<0,P)
                    Pts = PtsD[:,:,0]
                elif tb == 'SonnebornBerger':
                    Pts = SonnebornBerger(ResultsArr,self.HMatrix,self.AMatrix,self.PW[:,:,0])
                elif tb == 'Koya':   
                    Pts = Koya(self.PW[:,:,0],self.HMatrix,self.AMatrix,ResultArrH,ResultArrA)
                elif tb == 'moreBlack':
                    Pts = numpy.zeros(shape=(maxN,P))
                    numpy.add.at(numpy.swapaxes(Pts,0,1),self.AMatrix,1)
                elif tb == 'Random':
                    #numpy.random.seed(10)
                    Pts = numpy.argsort(numpy.random.rand(maxN,P))
                elif type(tb) == int:
                    Pts = self.Children[tb].Score[:,:,0]
                else:
                    print(tb+' is not implemented, skipping')
                    continue
                #print('Ptsshape',Pts.shape,Pts[self.NsToCalc,:].shape)
                #print('Ranks',self.RankNo.shape)
                self.RankNo[self.NsToCalc,:] = RanksFromValues(Pts[self.NsToCalc,:],self.RankNo[self.NsToCalc,:]) #Points
                #if int(numpy.sum(self.RankNo))==maxN*P*(P+1)/2:
                #    break
                self.NsToCalc = numpy.nonzero(numpy.sum(self.RankNo,axis=1) < int(P*(P+1)/2))[0]
                if len(self.NsToCalc) == 0:
                    break
            self.Ranks = self.FromRankNoToID(self.RankNo)
            #self.SeedRank = self.FromIDToSeed(self.Ranks)
            #for st in self.SeedTrigger:
            #    st[0].Seeds[st[2]]=self.Ranks[:,st[1]]
        else:
            self.PW = numpy.zeros(shape=(1,len(self.Seeds),1))-1
            self.PW = numpy.zeros(shape=(maxN,1,1))-1
            self.SpecialRanks(maxN,P)
            print(self.id,'special')
        for z in range(P):
            #print('typeranks',type(self.Ranks))
            Qu = db.EncounterRanks.insert().prefix_with("OR REPLACE").values(SimPoint=Spid,Encounter=self.id,Ranks=z,IDs=self.Ranks[:,z].tobytes())
            db.conn.execute(Qu)
            db.session.commit()
        self.Scoring()
        db.session.commit()
        Qu = db.SimEncounter.insert().prefix_with("OR REPLACE").values(SimPoint=Spid,Encounter=self.id,Score=self.Score.tobytes())
        db.conn.execute(Qu)
        db.session.commit()
        #self.FromIDToSeed(self.Ranks)
        #self.LogSims(Round,numpy.arange(N))
        #if len(self.Seeds) > 2:
        #    viz.Necessary(self,World,0)
        #self.PlayerPerStage()

    def ScoringDefault(self):
        if len(self.Seeds) == 2:
            self.Score = numpy.expand_dims(- self.RankNo + 0.5 + 0.5 * numpy.expand_dims(numpy.sum(self.RankNo,axis=1),axis=1),axis=2)
        else:
            self.Score = numpy.expand_dims(numpy.max(self.RankNo)-self.RankNo,axis=2)
        self.Score = self.Score.astype(floatDT)

    def Scoring(self):
        self.ScoringDefault()
    
    def PlayerPerStage(self):
        self.UniqueStages = []
        for g in self.GamesUnder:
            if g.RoundTree not in self.UniqueStages:
                self.UniqueStages.append(g.RoundTree)
        print(self.UniqueStages)

    def LogSims(self,Round,Ns):
        Arr1,Arr2,E = self.CalcOdds('Rank','Pts',self,slice(0,-1))
        if 'Rank|Pts' not in self.Sims:
            self.Sims['Rank|Pts']={}
        self.Sims['Rank|Pts'][self.Progress] = [Arr1,Arr2,E]
        if self.Progress not in (self.RoundsArray):
            self.RoundsArray.append(self.Progress)


            

def Berger(P,S):
    TotalGames = int(P*(P-1)/2)*S
    HMatrix = numpy.zeros(shape=(1,TotalGames),dtype=intDT)
    AMatrix = numpy.zeros(shape=(1,TotalGames),dtype=intDT)
    Rounds = numpy.zeros(shape=(TotalGames),dtype=intDT)
    n = 0
    for s in range(S):
        for i in range(1,P):
            for j in range(i+1,P):
                WB = (bool((i+j+s)%2) == (i<j))
                BW = not WB
                HMatrix[0,n] = i*WB+j*BW-1
                AMatrix[0,n] = j*WB+i*BW-1
                #Rounds[n] = (i+j-2)%(P-1) + (P-1)*s
                if i+j-1 < P:
                    Rounds[n] = i+j-1 + (P-1)*s
                else:
                    Rounds[n] = i+j-P + (P-1)*s
                n+=1
            WB = (2*i>P) == bool((s+1)%2)
            BW = not WB
            HMatrix[0,n] = P*WB+i*BW-1
            AMatrix[0,n] = i*WB+P*BW-1
            if 2*i <= P:
                Rounds[n] = 2*i-1 + (P-1)*s
            else:
                Rounds[n] = 2*i-P + (P-1)*s
            #Rounds[n] = (2*i-2)%(P-1) + (P-1)*s + 1
            n+=1
    Order = numpy.argsort(Rounds)
    #print(Order)
    return HMatrix,AMatrix,Rounds
    return numpy.expand_dims(HMatrix[0,Order],axis=0),numpy.expand_dims(AMatrix[0,Order],axis=0),Rounds[Order]

def PointsPT(HMatrix,ResultArrH,AMatrix,ResultArrA,P): 
    #PWn = numpy.zeros(shape=(ResultArrH.shape[0],P,1),dtype=floatDT)
    #for p in range(P):
    #    PWn[:,p] += numpy.sum(numpy.expand_dims(HMatrix==p,axis=2)*ResultArrH + numpy.expand_dims(AMatrix==p,axis=2)*ResultArrA,axis=1)
    RHx,RAx = numpy.expand_dims(ResultArrH,axis=1),numpy.expand_dims(ResultArrA,axis=1)
    HMx,AMx = numpy.expand_dims(HMatrix,axis=(1,3)),numpy.expand_dims(AMatrix,axis=(1,3))
    PP = numpy.expand_dims(numpy.arange(P),axis=(0,2,3))
    PWn2 = numpy.sum((PP==HMx)*RHx+(PP==AMx)*RAx,axis=2)
    #if numpy.allclose(PWn, PWn2) == False:
    #    print('RHx',RHx.shape)
    #    print('HMx',HMx.shape)
    #    print(PWn.shape,PWn2.shape)
    return PWn2             


def RanksFromValues(Pts,RanksUp):
    N,Pl = Pts.shape
    PtsX = numpy.expand_dims(Pts,axis=2)
    PtsY = numpy.expand_dims(Pts,axis=1)
    Ranks = numpy.sum(PtsY>PtsX,axis=2)+1
    if numpy.max(RanksUp)==1:
        return Ranks
    else:
        TB = -Pl*RanksUp-Ranks
        Ranks[:,:]=1
        return RanksFromValues(TB,Ranks) 

def RanksFromValuesBak(Pts,Ranks):
    RanksUp = numpy.copy(Ranks)
    N,Pl = Pts.shape
    PtsRange = numpy.flip(numpy.unique(Pts))
    RanksAssigned = numpy.zeros(shape=(Ranks.shape[0]),dtype=intDT)
    Ranks[:,:] = 0# = numpy.zeros(shape=(N,Pl),dtype=intDT)
    for i in PtsRange:
        Ptsi = (Pts==i)
        Tied = numpy.sum(Ptsi,axis=1)
        Who = numpy.where(Ptsi) #all with certain number of points (pl,n)
        try: Ranks[Who[0],Who[1]]=RanksAssigned[Who[0]]+1 #assign rank to above
        except: print(Pts.shape,PtsRange,Ranks.shape,RanksAssigned.shape)
        RanksAssigned += Tied
    if numpy.max(RanksUp)==1:
        return Ranks
    else:
        TB = -Pl*RanksUp-Ranks
        Ranks[:,:]=1
        return RanksFromValues(TB,Ranks) 

def Koya(PW,HMatrix,AMatrix,ResultsArrHR,ResultsArrAR):
    ResultsArrH = ResultsArrHR[:,:,0]
    ResultsArrA = ResultsArrAR[:,:,0]
    N2,G = ResultsArrH.shape
    N,P = PW.shape
    Av = numpy.expand_dims(numpy.average(PW,axis=1),axis=1)
    Pl=(PW>=Av)
    Ra = numpy.expand_dims(numpy.array(range(N)),axis=1)
    ResArrHKoya = numpy.expand_dims(Pl[Ra,AMatrix] * ResultsArrH,axis=2)
    ResArrAKoya = numpy.expand_dims(Pl[Ra,HMatrix] * ResultsArrA,axis=2)
    Koyascore = PointsPT(HMatrix,ResArrHKoya,AMatrix,ResArrAKoya,P)
    return numpy.squeeze(Koyascore,axis=2)
    
def SonnebornBerger(ResultsArrR,HMatrix,AMatrix,PW):
    ResultsArr = ResultsArrR[:,:,0]
    N,G = ResultsArr.shape
    Crosstable = numpy.zeros(shape=(ResultsArr.shape[0],PW.shape[1],PW.shape[1]))
    for g in range(G):
        Crosstable[:,HMatrix[:,g],AMatrix[:,g]] += numpy.expand_dims(ResultsArr[:,g],axis=1)
        Crosstable[:,AMatrix[:,g],HMatrix[:,g]] -= numpy.expand_dims(ResultsArr[:,g],axis=1)
    PWt = numpy.expand_dims(PW,axis=2)
    Res = numpy.matmul(Crosstable,PWt)[:,:,0]
    return(Res)

def Allto1600(G,Buckets):
    T = len(Buckets)
    SHMat = numpy.zeros(shape=(T,T),dtype=floatDT, order='F')
    SHMat[Buckets,Buckets]= 1.000
    return(SHMat)

def ProperResMatrix(G):
    numpy.fill_diagonal(G,0.0)
    TSSum = numpy.sum(G,axis=0) #Set matrix diagonal to negative su7m of lines (Knotenpotentialverfahren)
    numpy.fill_diagonal(G,-TSSum)
    return G

def CovMatrix(D):
    CovMat = -0.5*D
    CovMat -= CovMat[0,:]
    CovMat -= CovMat[:,0]
    CovMat[0,0]  =1.0
##    A = numpy.sqrt(D[1:,1:])
##    print(A)
##    u, s, v = numpy.linalg.svd(A)
##    print(u.shape,s.shape,v.shape)
##    #print(type(A),type(Asq))
##    ur,sr,vr = u[:,:10],s[:10],v[:10,:]
##    print(ur.shape,sr.shape,vr.shape)
##    for x in range(1,100):
##        B = u[:,:x]@la.diagsvd(s[:x],x,x)[:x]@v[:x,:]
##        print(x,numpy.sqrt(numpy.sum(numpy.power(A-B,2))/A.shape[0]**2))
##        if numpy.allclose(B,A, rtol=0, atol=1e+00) == True:
##            print('good from: ',x)
##            break
    #print(numpy.allclose(a, numpy.dot(u, numpy.dot(smat, vh))))
    return CovMat

def CalcNewU(G,I,Buckets):
    SHMat = Allto1600(G,Buckets)
    #print(G.shape,SHMat.shape)
    C = numpy.asmatrix(numpy.add(G,SHMat))
    U = numpy.linalg.solve(C.astype(floatDT),numpy.array(I).astype(floatDT))
    U[Buckets==0,0]+=2881.0
    U[Buckets>0,0]+=2300.0
    #print(U[:7,0])
    return U,numpy.asarray(C)

def CompCalc(G,I,Buckets):  
##    #print('G2',G)
##    G2 = numpy.zeros(G.shape,dtype=floatDT)
##    for b in numpy.unique(Buckets):
##        G2[0,b] += 0.000016
##        G2[b,0] += 0.000016
##        G2[0,0] -= 0.000016
##        G2[b,b] -= 0.000016
##    U2,G3 = CalcNewU(G+G2,I,numpy.zeros(G.shape[0],dtype=intDT))
##    y = numpy.linalg.inv(G3)
    U,G = CalcNewU(G,I,Buckets)
    y = numpy.linalg.inv(G) ##
    yt = numpy.diag(y)
    ytv = numpy.expand_dims(yt,axis=0)
    yth = numpy.expand_dims(yt,axis=1)
    Rs = y-ytv-yth+(y.transpose())
    for j in numpy.unique(Buckets): ##
        Js = numpy.array(numpy.where(Buckets==j)[0]) ##
        nonJs = numpy.array(numpy.where(Buckets!=j)[0]) ##
        Rs[Js[:,None],nonJs] = numpy.Inf ##
    #print(Rs[:5,:5])
    return U,Rs

def RefineBuckets(Buckets,t1,t2):
    b1 = Buckets[t1]
    b2 = Buckets[t2]
    if b1!=b2:
        Buckets[Buckets==max(b1,b2)]=min(b1,b2)
    return Buckets

