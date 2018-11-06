import numpy as np
import random
import math
import collections

BLANK = 0
SHIP = 1
UNKNOWN = -1

# creating some sample boards to bootstrap
RANDOM_BOARDS = [
    np.array([
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]),
    np.array([
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]),
    np.array([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0]
    ]),
    np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ]),
    np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ]),
]


class GameState:
    def __init__(self, n=5):  # skeleton initialization
        self.playerBoards = {
            'p1': np.zeros((n, n)),
            'p2': np.zeros((n, n))
        }
        self.playerKnowledge = {  # knowledge of opponent's board
            'p1': np.full((n, n), -1),
            'p2': np.full((n, n), -1)
        }
        self.boatSizes = []

    def manualRandomInitialization(self):
        self.playerBoards = {
            'p1': random.choice(RANDOM_BOARDS),
            'p2': random.choice(RANDOM_BOARDS)
        }
        self.playerKnowledge = {  # knowledge of opponent's board
            'p1': np.full((5, 5), -1),
            'p2': np.full((5, 5), -1)
        }
        self.boatSizes = np.array([1, 2, 3])

    # TODO: actual random initialization
    def getActions(self, playerID):
        action_matrix = np.vstack(np.where(self.playerKnowledge[playerID] == -1)).T
        return [(x[0], x[1]) for x in action_matrix]

    def getScore(self, playerID):
        ones = np.vstack(np.where(self.playerKnowledge[playerID] == 1)).T
        return np.sum([self.playerKnowledge[playerID][x[0], x[1]] for x in ones])

    def didWin(self, playerID):
        return self.getScore(playerID) == np.sum(self.boatSizes)

    def getEnemyPlayerID(self, playerID):
        return 'p1' if playerID == 'p2' else 'p2'

    def didLose(self, playerID):
        return self.getScore(self.getEnemyPlayerID(playerID)) == np.sum(self.boatSizes)

    def move(self, playerID, coordinate):  # TODO: test this
        if self.playerKnowledge[playerID][coordinate] != -1:
            return False  # illegal move
        updateValue = self.playerBoards[self.getEnemyPlayerID(playerID)][coordinate]
        self.playerKnowledge[playerID][coordinate] = updateValue
        return updateValue

    def __str__(self):
        return "playerBoards:\n%s,\nplayerKnowledge:\n%s,\nplayerOneScore: %d,\nplayerTwoScore: %d" % (
                str(self.playerBoards),
                str(self.playerKnowledge),
                self.getScore('p1'),
                self.getScore('p2')
            )

class RandomCPU:
    def getAction(self, gamestate, playerID):
        actions = gamestate.getActions(playerID)
        return random.choice(actions)

class GradDescentCPU:
    def __init__(self):
        self.feature_weights = np.random.normal(0, 1, 9)
        self.learning_rate = 0.01

    def scorePrediction(self, gamestate, playerID, action):
        features = self.generateFeatures(gamestate, playerID, action)
        predicted_score = 1 / (1 + math.exp(-self.feature_weights.dot(features)))
        return predicted_score, features

    def generateFeatures(self, gamestate, playerID, coordinate):
        features = []  # 1s in row, 0s in row, -1s in row, neighbor 1s, neighbor 0s, neighbor -1s
        gameboard = gamestate.playerKnowledge[playerID]
        c_row = collections.Counter(gameboard[coordinate[0], :])
        c_col = collections.Counter(gameboard[:, coordinate[1]])
        c_neighbor = collections.Counter(
            gameboard[max(0, coordinate[0] - 1): min(len(gameboard) - 1, coordinate[0] + 1), coordinate[1]]
        )
        for i in range(-1, 2):
            features.extend([c_row[i], c_col[i], c_neighbor[i]])
        return np.array(features)

    def getAction(self, gamestate, playerID):
        actions = gamestate.getActions(playerID)
        bestScore = -1
        bestAction = None
        bestScoreFeatures = None
        for action in actions:
            curScore, features = self.scorePrediction(gamestate, playerID, action)
            if curScore > bestScore:
                bestAction = action
                bestScore = curScore
                bestScoreFeatures = features
        return bestAction, bestScore, features

    def updateWeights(self, reward, prediction, features):
        loss = (prediction - reward) ** 2
        gradient = 2 * (prediction - reward) * prediction * (1 - prediction) * features
        self.feature_weights -= self.learning_rate * gradient


