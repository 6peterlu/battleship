from django.shortcuts import render, redirect
from django.http import HttpResponse
from board.models import Board, Game, User, UserName
from . import gamestate

gameEngine = gamestate.GameState()
gameEngine.randomInitialization([2,3,4], 10)
name = ""

randomAgent = gamestate.RandomCPU()
# Create your views here.
def index(request):
    return render(request, 'board/index.html')

def game(request):
	# TODO: add to db
	user = UserName(user_name=request.POST['name'])
	user.save()
	global name
	name = request.POST['name']
	playerrendergrid = []
	for i in range(len(gameEngine.playerKnowledge['p1'])):
		playerrendergrid.append([])
		for j in range(len(gameEngine.playerKnowledge['p1'])):
			playerrendergrid[i].append((gameEngine.playerKnowledge['p1'][i, j], 'white'))
	return render(request, 'board/game.html', {"name": request.POST['name'], 'enemyBoard': gameEngine.playerKnowledge['p1'], 'userBoard': playerrendergrid, 'enemyMove': "No move yet"})

def move(request):
	x = int(request.POST['x'])
	y = int(request.POST['y'])
	gameEngine.move('p1', (x,y))
	if gameEngine.didWin('p1'):
		return render(request, 'board/win.html')
	randomAgentAction = randomAgent.getAction(gameEngine, 'p2')
	gameEngine.move('p2', randomAgentAction)
	if gameEngine.didLose('p1'):
		return render(request, 'board/lose.html')
	playerrendergrid = []
	for i in range(len(gameEngine.playerKnowledge['p1'])):
		playerrendergrid.append([])
		for j in range(len(gameEngine.playerKnowledge['p1'])):
			if gameEngine.playerKnowledge['p2'][i, j] == 1:
				playerrendergrid[i].append((gameEngine.playerBoards['p1'][i, j], 'red'))
			elif gameEngine.playerKnowledge['p2'][i, j] == 0:
				playerrendergrid[i].append((gameEngine.playerBoards['p1'][i, j], 'green'))
			else:
				playerrendergrid[i].append((gameEngine.playerBoards['p1'][i, j], 'white'))
	return render(request, 'board/game.html', {"name": name, 'enemyBoard': gameEngine.playerKnowledge['p1'], 'userBoard': playerrendergrid, "enemyMove": randomAgentAction})

# def moveClick(i, j):
