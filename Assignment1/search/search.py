# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
import copy

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** Modified by Du Tianyi 3035149685 ***"
    from game import Directions

    frontier = util.Stack()
    explored = []
    solution = []
    path = [problem.getStartState()]

    """frontier[0] stores a list of paths, frontier[1] stores a list of solutions"""
    frontier.push((path,solution))

    while not frontier.isEmpty():
        cur_node = frontier.pop()
        #print "cur cur_node",cur_node[0][-1]

        if problem.isGoalState(cur_node[0][-1]):
            return cur_node[1]
        #print "cur_node:",cur_node[0]
        
        explored.append(cur_node[0][-1])
        #print "explored:", explored
        next_successor = problem.getSuccessors(explored[-1])

        #print "next_pos:"
        for successor in next_successor:
            #if successor[0] not in explored and successor[0] not in expand:
            if successor[0] not in explored:
                #print "cur_node:",cur_node
                new_node = copy.deepcopy(cur_node)
                new_node[0].append(successor[0])
                new_node[1].append(successor[1])
                #print "new_node:",new_node
                frontier.push(new_node)
                #print successor[0]

    return []
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** Modified by Du Tianyi 3035149685 ***"
    from game import Directions
    #frontier = util.Queue()
    expand = []
    explored = set()
    frontier = []

    solution = []
    path = [problem.getStartState()]
    frontier.append([path,solution])

    while frontier:
        curPath,curSolution = frontier.pop(0)
        curState = curPath[-1]
        #print "curState:",curState
        #print "curPath:",curPath

        if problem.isGoalState(curState):
        	return curSolution
        #print "curState:",curState
        explored.add(curState)
        successor = problem.getSuccessors(curState)
        for nextState,movement,cost in successor:
            if nextState not in explored and nextState not in expand:
                expand.append(nextState)
                newPath = curPath[:]
                newPath.append(nextState)
                newSolution = curSolution[:]
                newSolution.append(movement)
                frontier.append([newPath,newSolution])

    return []
    #util.raiseNotDefined()

def priorityFunc(frontier_para):
    return frontier_para[3]

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** Modified by Du Tianyi 3035149685 ***"
    from game import Directions

    def priorityFunc(frontier_para):
        return frontier_para[2]
    #frontier = util.PriorityQueueWithFunction(priorityFunc)
    frontier = []
    expand = dict()
    explored = []
    solution = []
    path = [problem.getStartState()]
    cost = 0

    """frontier[0] stores a list of paths, frontier[1] stores a list of solutions"""
    frontier.append([path,solution,cost])
    while frontier:
        cur_node = frontier.pop(0)
        if problem.isGoalState(cur_node[0][-1]):
            return cur_node[1]
        
        explored.append(cur_node[0][-1])
        next_successor = problem.getSuccessors(explored[-1])
        for successor in next_successor:
            if successor[0] not in explored:
                new_node = copy.deepcopy(cur_node)
                new_node[0].append(successor[0])
                new_node[1].append(successor[1])
                new_node[2] += successor[2]
                if successor[0] not in expand:
                    expand[successor[0]] = new_node[2]
                    frontier.append(new_node)
                    frontier.sort(key = priorityFunc)
                else:
                    if expand[successor[0]] > new_node[2]:
                        expand[successor[0]] = new_node[2]
                        for node in frontier:
                            if node[0][-1] == successor[0]:
                                frontier.remove(node)
                        frontier.append(new_node)
                        frontier.sort(key = priorityFunc)

    return []
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** Modified by Du Tianyi 3035149685 ***"
    from game import Directions

    frontier = []
    expand = dict()
    explored = []

    path = [problem.getStartState()]
    solution = []
    cost = 0
    fVal = 0

    frontier.append([path,solution,cost,fVal])
    while frontier:
        curPath,curSolution,curCost,curHVal = frontier.pop(0)
        curState = curPath[-1]

        if problem.isGoalState(curState):
        	return curSolution
        
        explored.append(curState)
        successor = problem.getSuccessors(explored[-1])
        for nextState,movement,cost in successor:
            if nextState not in explored:
                newPath = curPath[:]
                newPath.append(nextState)
                newSolution = curSolution[:]
                newSolution.append(movement)
                newCost = curCost + cost
                # print "newCost:",newCost
                # print "heuristic:",heuristic(nextState,problem)
                newFVal = newCost + heuristic(nextState,problem)
                if nextState not in expand:
                    expand[nextState] = newFVal
                    frontier.append([newPath,newSolution,newCost,newFVal])
                    frontier.sort(key = priorityFunc)
                else:
                    if expand[nextState] > newFVal:
                        expand[nextState] = newFVal
                        for tmpPath,tmpSolution,tmpCost,tmpFVal in frontier:
                            if tmpPath[-1] == nextState:
                                frontier.remove([tmpPath,tmpSolution,tmpCost,tmpFVal])
                        frontier.append([newPath,newSolution,newCost,newFVal])
                        frontier.sort(key = priorityFunc)

    return []
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
