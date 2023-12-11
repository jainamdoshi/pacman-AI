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

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = bestIndices[0] # Pick randomly among the best

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
        newFoodList = newFood.asList()
        newCapsuleList = successorGameState.getCapsules()

        # Calculating all the distances between the pacman position and the food, and the pacman position and the ghosts
        foodDistances = [util.manhattanDistance(newPos, food) for food in newFoodList]
        ghostDistances = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]

        score = 0
        stopValue = 100
        ghostFactor = 10000
        foodEatenValue = 100
        capsuleEatenValue = 1000
        capsuleDistanceValue = 1000
        foodDistanceValue = 10

        # Pacman should try not to stop
        if action == Directions.STOP:
            score -= stopValue

        # Pacman will try be 2 spaces away from the ghosts if they are not scared
        nearestGhost = min(ghostDistances) + 0.01
        if min(newScaredTimes) == 0 and nearestGhost < 2:
            score -= ghostFactor / nearestGhost
        elif max(newScaredTimes) > 0 :
            score += ghostFactor / ghostDistances[newScaredTimes.index(max(newScaredTimes))]

        # Pacman will try to eat food and capsules
        if len(newFoodList) < len(currentGameState.getFood().asList()):
            score += foodEatenValue

        if len(newCapsuleList) < len(currentGameState.getCapsules()):
            score += capsuleEatenValue

        # Pacman will try to eat the where this is more food and capsules
        score += sum([capsuleDistanceValue / util.manhattanDistance(newPos, capsule) for capsule in newCapsuleList])
        score += sum([foodDistanceValue / foodDistance for foodDistance in foodDistances])
        return score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        return self.getActionsRecursion(gameState, self.depth, 0)[1]
            
    def getActionsRecursion(self, gameState, depth, turn):

        # Base case
        if gameState.isWin() or gameState.isLose() or depth <= 0:
            return (self.evaluationFunction(gameState), None)
        
        # Calculate the agent index, legal moves and successor states
        agentIndex = turn % gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(agentIndex)
        states = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]

        # Change depth if all agents have taken their turn
        if agentIndex + 1 == gameState.getNumAgents():
            depth -= 1

        # Recursively call the function for each successor state
        newMoves = [self.getActionsRecursion(state, depth, turn + 1)[0] for state in states]
        value = None
        
        # Get max or min value depending on the agent
        if agentIndex == 0:
            value = max(newMoves)
        else:
            value = min(newMoves)

        # Return the value and the action that corresponds to the value
        return (value, legalMoves[newMoves.index(value)])

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.max(gameState, self.depth, 0, float('-inf'), float('inf'))[1]

    # A recursive function that returns the max value and the action that corresponds to the max value
    def max(self, gameState, depth, agentIndex, alpha, beta):

        # Base case
        if gameState.isWin() or gameState.isLose() or depth <= 0:
            return (self.evaluationFunction(gameState), None)
        
        legalMoves = gameState.getLegalActions(agentIndex)
        bestMove = (float('-inf'), None)

        # Recursively call the function for each successor state
        for move in legalMoves:

            # Getting minimum value for successor state 
            state = gameState.generateSuccessor(agentIndex, move)
            score = self.min(state, depth, agentIndex + 1, alpha, beta)

            # Keep track of the best move
            if score[0] > bestMove[0]:
                bestMove = (score[0], move)

            # Pruning
            if bestMove[0] > beta:
                return bestMove

            # Update alpha
            alpha = max(alpha, bestMove[0])
        
        return bestMove
    
    # A recursive function that returns the min value and the action that corresponds to the min value
    def min(self, gameState, depth, agentIndex, alpha, beta):

        # Base case
        if gameState.isWin() or gameState.isLose() or depth <= 0:
            return (self.evaluationFunction(gameState), None)
        
        agent = agentIndex
        legalMoves = gameState.getLegalActions(agent)
        
        bestMove = (float('inf'), None)
        function = None

        # Determine if the next agent is pacman or a ghost, and change depth accordingly
        if agentIndex + 1 == gameState.getNumAgents():
            agentIndex = 0
            function = self.max
            depth -= 1
        else:
            function = self.min
            agentIndex += 1

        # Recursively call the function for each successor state
        for move in legalMoves:

            # Getting value for successor state
            state = gameState.generateSuccessor(agent, move)
            score = function(state, depth, agentIndex, alpha, beta)

            # Keep track of the best move
            if score[0] < bestMove[0]:
                bestMove = (score[0], move)

            # Pruning
            if bestMove[0] < alpha:
                return bestMove
            
            # Update beta
            beta = min(beta, bestMove[0])
        
        return bestMove

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.getActionsRecursion(gameState, self.depth, 0)[1]

    def getActionsRecursion(self, gameState, depth, turn):

        # Base case
        if gameState.isWin() or gameState.isLose() or depth <= 0:
            return (self.evaluationFunction(gameState), None)

        # Calculate the agent index, legal moves and successor states
        agentIndex = turn % gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(agentIndex)
        states = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]

        # Change depth if all agents have taken their turn
        if agentIndex + 1 == gameState.getNumAgents():
            depth -= 1

        # Recursively call the function for each successor state
        scores = [self.getActionsRecursion(state, depth, turn + 1)[0] for state in states]
        value = None
        index = None

        # Get max or average value depending on the agent
        if agentIndex == 0:
            value = max(scores)
            index = scores.index(value)
        else:
            value = sum(scores) / len(scores)
            index = scores.index(min(scores, key=lambda x:abs(x-value)))

        # Return the value and the action that corresponds to the value
        return (value, legalMoves[index])
        
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <write something here so we know what you did>
    """
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()
    foodDistancesManhattan = [util.manhattanDistance(pacmanPosition, food) for food in foodList]

    score = 0

    # For each ghost, calculate the distance between the pacman and the ghost
    for state in ghostStates:
        ghost = state.getPosition()
        scaredTime = state.scaredTimer
        distance = util.manhattanDistance(pacmanPosition, ghost)
        ghostFactor = -1000000
        
        # If the ghost is scared, the pacman will try to eat the ghost
        if scaredTime > 0:
            ghostFactor = 100
            score += 100

        # Ghost will only be considered if it is scared or if it is 2 spaces away from the pacman
        if scaredTime > 0 or distance < 3:
            score += ghostFactor / (distance + 0.01)

    # Pacman will try to go towards more food and capsules
    score += sum([100 / (util.manhattanDistance(pacmanPosition, capsule) + 0.01) for capsule in capsuleList])
    score += sum([1 / foodDistance for foodDistance in foodDistancesManhattan])
    score += 100 / (len(foodList) + 0.01)
    score += currentGameState.getScore() * 2

    return score

# Abbreviation
better = betterEvaluationFunction
