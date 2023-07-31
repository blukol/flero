from django.http import HttpResponse
from sqlalchemy import desc
import numpy

def page(request, ppp):
    return HttpResponse("Hello worldie!" +str(ppp))

def page2(request, sport,pp):
    import configglobal as db
    db.init()
    import oo3
    if sport == 'chess':
        import Chess.configchess as dbc
        import Chess.oo2chess as oo2
        dbc.tables()
    db.tables()
    #Magn = db.session.query(oo2.ChessPlayer).filter(oo2.ChessPlayer.Name == 'Carlsen, Magnus').all()
    #Magn = db.session.query(oo2.ChessPlayer).join(oo2.Elo).all() #,PlayerID = oo2.ChessPlayer.id
    if pp == 'Ranking':
        Elos2 = db.session.query(oo2.ChessPlayer, db.Elo).join(db.Elo).filter(db.Elo.c.Point == 805).order_by(desc(db.Elo.c.Elo)).all()
        Str = '<table><tr><th>Nation</th><th>Name</th><th>Elo</th></tr>'
        for i in range(len(Elos2)):
            Str += '<tr><td>'+ Elos2[i][0].Nationality +'</td>\
            <td>'+Elos2[i][0].Name +'</td>\
            <td>'+ str(round(Elos2[i][6],2))+  '</td></tr>'
        Str+='</table>'
    elif pp == 'Results':
        #Gams = db.session.query(db.SimEncounter, oo2.ChessGame).join(oo2.ChessGame).order_by(oo2.ChessGame.EndDate).all() #.join(oo3.SimPoint,id==db.SimEncounter.c.SimPoint) #,oo3.SimPoint <td>' + str(gam[8].Date)+ '</td>\
        #Gams = db.session.query(oo2.ChessGame,db.SimEncounter,db.FirstSeeds,oo2.ChessPlayer).join(db.SimEncounter)\
        #    .join(db.FirstSeeds).join(oo2.ChessPlayer).filter(db.SimEncounter.c.SimPoint==801).order_by(oo2.ChessGame.EndDate).all()
        Gams = db.conn.execute('SELECT * FROM Gamedetail')
        Str = '<table><tr><th></th></tr>'
        for gam in Gams:
            #for inv in gam.SimEncounter:
                Res = numpy.fromstring(gam[7], dtype=db.floatDT)
                Res = Res.reshape(int(Res.shape[0]/2),2)
                try:
                    Prob = numpy.average(Res[:,gam[6]]) #<td>' + str(gam[0].EndDate)+ '</td>\
                except:
                    Prob = -1.0
                Str += '<tr>\
                <td>' + str(gam[0])+ '</td>\
                <td>' + str(gam[1])+ '</td>\
                <td>' + str(gam[2])+ '</td>\
                <td>' + str(gam[3])+ '</td>\
                <td>' + str(gam[4])+ '</td>\
                <td>' + str(gam[5])+ '</td>\
                <td>' + str(gam[6])+ '</td>\
                <td>' + str(round(100.0*Prob,1))+ '%</td></tr>'
                continue
                Str += '<tr><td>A ' + str(gam[0])+ '</td>\
                            <td>B' + str(gam[1])+ '</td>\
                            <td>C' + str(len(gam[2]))+ '</td>\
                            <td>D' + str(gam[3].EndDate)+ '</td></tr>'
                           # <td>' + str(len(gam[3]))+ '</td>\
                            #<td>' + str(Res.shape[0])+ '</td>\
                            #<td>' + str(Prob)+ '</td>'
                #            <td>' + Elos2[i][0].Name + '</td>\
                #            <td>' + str(round(Elos2[i][6], 2)) + '</td></tr>'
        Str += '</table>'
    return HttpResponse(Str)
    #return HttpResponse("Result:" +str(len(Magn))+'\n LaLaZ '+str(Magn[0].Name))

def chess(request):
    pass
# Create your views here.

#import os, sys
#currentdir = os.path.dirname(os.path.realpath(__file__))
#parentdir = os.path.dirname(currentdir)
#sys.path.append(parentdir)
#import configglobal as db
#db.init()
#import Chess.configchess as dbc
#import Chess.oo2chess as oo2
#dbc.tables()
#db.tables()
##EElo = aliased(db.Elo)
#Magn = db.session.query(oo2.ChessPlayer).join(db.Elo,db.Elo.c.PlayerID == oo2.ChessPlayer.id).all() #,PlayerID = oo2.ChessPlayer.id
#Elos = db.session.query(db.Elo).join(oo2.ChessPlayer).all()
#Elos2 = db.session.query(oo2.ChessPlayer,db.Elo).join(db.Elo).filter(db.Elo.c.Point == 805).all()
#rr = 2 #