# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Calculate the distance to the nearest food
        foodDistances = [
            util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()
        ]
        minFoodDistance = min(foodDistances) if foodDistances else 0

        # Calculate the distance to the nearest ghost
        ghostDistances = [
            util.manhattanDistance(newPos, ghostState.getPosition())
            for ghostState in newGhostStates
        ]
        minGhostDistance = min(ghostDistances) if ghostDistances else 0

        # Encourage eating food by using the reciprocal of the food distance
        foodScore = 1.0 / (minFoodDistance + 1)

        # Discourage getting too close to ghosts by using the reciprocal of the ghost distance
        ghostScore = 0
        if minGhostDistance > 0:
            ghostScore = -1.0 / minGhostDistance

        # Encourage eating scared ghosts by considering the scared time
        scaredGhostScore = 0
        for scaredTime, ghostDistance in zip(newScaredTimes, ghostDistances):
            if scaredTime > ghostDistance:
                scaredGhostScore += scaredTime - ghostDistance

        return successorGameState.getScore() + foodScore + ghostScore + scaredGhostScore


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def minimax(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Maximizer (Pacman)
                maxValue = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    maxValue = max(maxValue, minimax(successor, agentIndex + 1, depth))
                return maxValue
            else:  # Minimizer (Ghosts)
                minValue = float("inf")
                nextAgentIndex = (
                    agentIndex + 1 if agentIndex + 1 < state.getNumAgents() else 0
                )
                nextDepth = depth + 1 if nextAgentIndex == 0 else depth
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    minValue = min(
                        minValue, minimax(successor, nextAgentIndex, nextDepth)
                    )
                return minValue

        bestAction = None
        maxScore = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = minimax(successor, 1, 0)
            if score > maxScore:
                maxScore = score
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def alpha_beta(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Maximizer (Pacman)
                maxValue = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    maxValue = max(
                        maxValue,
                        alpha_beta(successor, agentIndex + 1, depth, alpha, beta),
                    )
                    alpha = max(alpha, maxValue)
                    if beta < alpha:
                        break
                return maxValue
            else:  # Minimizer (Ghosts)
                minValue = float("inf")
                nextAgentIndex = (
                    agentIndex + 1 if agentIndex + 1 < state.getNumAgents() else 0
                )
                nextDepth = depth + 1 if nextAgentIndex == 0 else depth
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    minValue = min(
                        minValue,
                        alpha_beta(successor, nextAgentIndex, nextDepth, alpha, beta),
                    )
                    beta = min(beta, minValue)
                    if beta < alpha:
                        break
                return minValue

        bestAction = None
        maxScore = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = alpha_beta(successor, 1, 0, alpha, beta)
            if score > maxScore:
                maxScore = score
                bestAction = action
            alpha = max(alpha, maxScore)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def expectimax(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Maximizer (Pacman)
                maxValue = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    maxValue = max(
                        maxValue, expectimax(successor, agentIndex + 1, depth)
                    )
                return maxValue
            else:  # Chance node (Ghosts)
                sumValue = 0
                legalActions = state.getLegalActions(agentIndex)
                numActions = len(legalActions)
                nextAgentIndex = (
                    agentIndex + 1 if agentIndex + 1 < state.getNumAgents() else 0
                )
                nextDepth = depth + 1 if nextAgentIndex == 0 else depth
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    sumValue += expectimax(successor, nextAgentIndex, nextDepth)
                return sumValue / numActions

        bestAction = None
        maxScore = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = expectimax(successor, 1, 0)
            if score > maxScore:
                maxScore = score
                bestAction = action

        return bestAction


# New class variable to store the previous position
prevPacmanPosition = None


def betterEvaluationFunction(currentGameState):
    global prevPacmanPosition

    # Get the position of Pacman
    pacmanPosition = currentGameState.getPacmanPosition()

    # Reward for moving from the previous position
    moveReward = 0
    if prevPacmanPosition and prevPacmanPosition != pacmanPosition:
        moveReward = 1

    # Update the previous position
    prevPacmanPosition = pacmanPosition

    # Compute the reciprocal of the distance to the nearest food
    foodDistances = [
        util.manhattanDistance(pacmanPosition, food)
        for food in currentGameState.getFood().asList()
    ]
    if foodDistances and min(foodDistances) != 0:
        reciprocalFoodDistance = 1.0 / min(foodDistances)
    else:
        reciprocalFoodDistance = 1.0

    # Compute the number of remaining food pellets
    remainingFood = currentGameState.getNumFood()

    # Compute the reciprocal of the distance to the nearest ghost
    ghostDistances = [
        util.manhattanDistance(pacmanPosition, ghost.getPosition())
        for ghost in currentGameState.getGhostStates()
    ]
    minGhostDistance = min(ghostDistances) if ghostDistances else None
    if minGhostDistance and minGhostDistance != 0:
        reciprocalGhostDistance = 1.0 / minGhostDistance
    else:
        reciprocalGhostDistance = 0

    # Compute the current game score
    currentScore = currentGameState.getScore()

    # Combine the features with different weights
    evaluation = (
        (2.5 * reciprocalFoodDistance)
        - (2.0 * reciprocalGhostDistance)
        + (1.0 * currentScore)
        - (0.5 * remainingFood)
        + (2.2 * moveReward)
    )

    return evaluation


# Abbreviation
better = betterEvaluationFunction
