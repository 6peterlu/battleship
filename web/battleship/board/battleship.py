from gamestate import GameState, RandomCPU, GradDescentCPU
import numpy as np


def HvHRepl():
    print("starting human vs human game.")
    gamestate = GameState()
    gamestate.manualRandomInitialization()
    curPlayer = 'p1'
    while not gamestate.didWin(curPlayer):
        print('%s\'s turn: \n %s' % (
                curPlayer,
                str(gamestate.playerKnowledge[curPlayer])
            )
        )
        x_coord = int(input('Enter x_coord: '))
        y_coord = int(input('Enter y_coord: '))
        coordinate = (x_coord, y_coord)
        while gamestate.move(curPlayer, coordinate) == False:
            print('repeated entry. enter another location.')
            x_coord = int(input('Enter x_coord: '))
            y_coord = int(input('Enter y_coord: '))
            coordinate = (x_coord, y_coord)
        print('%s\'s turn complete. Resulting board: %s' %
            (
                curPlayer,
                str(gamestate.playerKnowledge[curPlayer])
            )
        )
        curPlayer = gamestate.getEnemyPlayerID(curPlayer)
    print('Human v Human game complete!')

def HvCRepl(agent):
    print("starting human vs computer game.")
    gamestate = GameState()
    gamestate.manualRandomInitialization()
    humanPlayer = 'p1'
    cpuPlayer = 'p2'
    while not (gamestate.didWin(humanPlayer) or gamestate.didLose(humanPlayer)):
        print('%s\'s turn: \n %s' % (
                humanPlayer,
                str(gamestate.playerKnowledge[humanPlayer])
            )
        )
        x_coord = int(input('Enter x_coord: '))
        y_coord = int(input('Enter y_coord: '))
        coordinate = (x_coord, y_coord)
        while not gamestate.move(humanPlayer, coordinate) == False:
            print('repeated entry. enter another location.')
            x_coord = int(input('Enter x_coord: '))
            y_coord = int(input('Enter y_coord: '))
            coordinate = (x_coord, y_coord)
        print('%s\'s turn complete. Resulting board: %s' %
            (
                humanPlayer,
                str(gamestate.playerKnowledge[humanPlayer])
            )
        )
        cpuAction = agent.getAction(gamestate, cpuPlayer)
        print('cpu plays: %s' %
            str(cpuAction)
        )
        gamestate.move(cpuPlayer, cpuAction)
        print('cpu turn complete. Resulting board: %s' %
            str(gamestate.playerKnowledge[cpuPlayer])
        )
    print('Human v Computer game complete!')


def CvCSimulation(agent1, agent2, verbose = False):
    gamestate = GameState()
    gamestate.randomInitialization([1,2,3], 10)
    agent1name = 'p1'
    agent2name = 'p2'
    moves = 0
    while True:
        moves += 1
        agent1action = agent1.getAction(gamestate, agent1name)
        gamestate.move(agent1name, agent1action)
        if (gamestate.didWin(agent1name)): break
        agent2action = agent2.getAction(gamestate, agent2name)
        gamestate.move(agent2name, agent2action)
        if (gamestate.didWin(agent2name)): break
    winner = agent1name if gamestate.didWin(agent1name) else agent2name  # random agent 1 wins 55-60% of the time
    if verbose:
        print('Computer v computer game complete!')
        print('Statistics:')
        print('Num moves: %d' % moves)
        print('Winner: %s' % winner)
        print('Final board agent 1: %s' % str(gamestate.playerKnowledge[agent1name]))
        print('Final board agent 2: %s' % str(gamestate.playerKnowledge[agent2name]))
    return winner, moves

def CvAISimulation(randomAgent, AIAgent, verbose=False):
    gamestate = GameState()
    gamestate.randomInitialization([2,3,4], 10)
    moves = 0
    randomAgentName = 'p1'
    AIAgentName = 'p2'
    while True:
        moves += 1
        randomAgentAction = randomAgent.getAction(gamestate, randomAgentName)
        gamestate.move(randomAgentName, randomAgentAction)
        if (gamestate.didWin(randomAgentName)): break
        AIAgentAction, prediction, features = AIAgent.getAction(gamestate, AIAgentName)
        reward = gamestate.move(AIAgentName, AIAgentAction)
        if (gamestate.didWin(AIAgentName)): break
        if verbose:
            print("features: %s" % str(features))
            print("weights: %s" % str(AIAgent.feature_weights))
            print("agent chose (%d, %d)" % (AIAgentAction[0], AIAgentAction[1]))
            print("board: %s" % str(gamestate.playerKnowledge))
        AIAgent.updateWeights(reward, prediction, features)
    winner = randomAgentName if gamestate.didWin(randomAgentName) else AIAgentName
    if verbose:
        print('Computer v AI game complete!')
        print('Statistics:')
        print('Num moves: %d' % moves)
        print('Winner: %s' % winner)
        print('Final board agent 1: %s' % str(gamestate.playerKnowledge[randomAgentName]))
        print('Final board agent 2: %s' % str(gamestate.playerKnowledge[AIAgentName]))
    return winner, moves

def testInitialization(boatSizes, boardSize, n):
    p1boards = np.zeros((boardSize, boardSize))
    for i in range(n):
        gamestate = GameState()
        gamestate.randomInitialization(boatSizes, boardSize)
        p1boards += gamestate.playerBoards['p1']
    print(p1boards / n)


def CvCSimulationRunner(agent1, agent2, num_simulations, is_training=False, verbose=False):
    p1wins = 0
    p2wins = 0
    average_moves = 0
    for i in range(num_simulations):
        winner, moves = None, None
        if is_training:
            winner, moves = CvAISimulation(agent1, agent2, verbose)
        else:
            winner, moves = CvCSimulation(agent1, agent2, verbose)
        if winner == 'p1':
            p1wins += 1
        else:
            p2wins += 1
        average_moves += moves
        if i % 50 == 0:
            print('iteration %d complete' % i)
    print('Final statistics:')
    print('p1 win percentage: %g' % (p1wins * 100 / (p1wins + p2wins)))
    print('average moves: %g' % (average_moves / num_simulations))


if __name__ == "__main__":
    randomAgent = RandomCPU()
    # agent1 = RandomCPU()
    # agent2 = RandomCPU()
    aiAgent = GradDescentCPU()
    CvCSimulationRunner(randomAgent, aiAgent, 1000, is_training=True)
    # CvCSimulationRunner(randomAgent, aiAgent, 1, is_training=True, verbose=True)
    # CvCSimulationRunner(randomAgent, aiAgent, 500, is_training=True)
    # HvCRepl(randomAgent)
    # CvCSimulationRunner(agent1, agent2, 1000)
    # testInitialization([2,3,4], 10, n=1000)

