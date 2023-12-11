# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    fringe = util.Stack()
    startState = problem.getStartState()
    visited = [startState[0]]
    
    fringe.push((problem.getStartState(), []))

    # Iterate while no more states are left to explore
    while not fringe.isEmpty():
        state, actions = fringe.pop()
        
        #Check if goal node is found
        if problem.isGoalState(state):
            return actions
        
        # Add all successors state to the fringe if not visited
        for successor in problem.getSuccessors(state):
            if successor[0] not in visited:
                visited.append(successor[0])
                fringe.push((successor[0], actions + [successor[1]]))

    # No solution found
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    fringe = util.Queue()
    visited = []

    fringe.push((problem.getStartState(), []))

    # Iterate while no more states are left to explore
    while not fringe.isEmpty():
        state, actions = fringe.pop()
        
        # End the algorithm if the goal state is found
        if problem.isGoalState(state):
            return actions
        
        elif state not in visited:
            visited.append(state)

            # Add all successors state to the fringe
            for successor in problem.getSuccessors(state):
                fringe.push((successor[0], actions + [successor[1]]))

    # No solution found
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    fringe = util.PriorityQueue()
    visited = []

    fringe.push((problem.getStartState(), [], 0), 0)

    # Iterate while no more states are left to explore
    while not fringe.isEmpty():
        state, actions, cost = fringe.pop()
        
        # End the algorithm if the goal state is found
        if problem.isGoalState(state):
            return actions
        
        elif state not in visited:
            visited.append(state)

            # Add all successors state to the fringe
            for successor in problem.getSuccessors(state):
                newCost = cost + successor[2]
                fringe.push((successor[0], actions + [successor[1]], newCost), newCost)

    # No solution found
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    fringe = util.PriorityQueue()
    visited = []
    fringe.push((problem.getStartState(), [], 0), 0)

    # Iterate while no more states are left to explore
    while not fringe.isEmpty():
        state, actions, cost = fringe.pop()
        
        # End the algorithm if the goal state is found
        if problem.isGoalState(state):
            return actions
        
        elif state not in visited:
            visited.append(state)

            # Add all successors state to the fringe. The priority is the newCost + heuristic
            for successor in problem.getSuccessors(state):
                newCost = cost + successor[2]
                fringe.push((successor[0], actions + [successor[1]], newCost), newCost + heuristic(successor[0], problem))

    # No solution found
    return []


#####################################################
# EXTENSIONS TO BASE PROJECT
#####################################################

# Extension Q1e
def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""

    # Iterate over the depth until a solution is found
    for depth in range(9999999999999):
        result = depthLimitedSearch((problem.getStartState(), [], 0), problem, depth)
        
        # Return the solution if found
        if result is not None:
            return result[:-1]
    
    # No solution found
    return []

def depthLimitedSearch(state, problem, depth):
    # Recursive function to perform depth limited search

    # Base case
    if depth == 0:

        # Return the action if the goal state is found
        if problem.isGoalState(state[0]):
            return [state[1]]

        # Return cut off if the goal state is not found
        return None
        
    # Recursive case
    elif depth > 0:

        # Recursively call the function on all successors
        for successor in problem.getSuccessors(state[0]):
            result = depthLimitedSearch(successor, problem, depth - 1)
            
            # Returns the action if one of the children returns the action
            if result is not None:
                return result + [state[1]]
        
        # Return cut off if the goal state is not found
        return None
    

#####################################################
# Abbreviations
#####################################################
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
