# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to
control Pacman.  To select an agent, use the '-p' option
when running pacman.py.  Arguments can be passed to your agent
using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the
project description.

Please only change the parts of the file you are asked to.
Look for the lines that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the
project description for details.

Good luck and happy searching!
"""
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search algorithm for a
    supplied search problem, then returns actions to follow that path.

    As a default, this agent runs DFS on a PositionSearchProblem to find location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game board. Here, we
        choose a path to the goal.  In this phase, the agent should compute the path to the
        goal and store it in a local variable.  All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in registerInitialState).  Return
        Directions.STOP if there is no further action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test,
    successor function and cost function.  This search problem can be
    used to find paths to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1

        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** Modified by Du Tianyi 3035149685 ***"
        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0
        self.visualize = True
        #self.cornerState = {self.corners[0]:0,self.corners[1]:0,self.corners[2]:0,self.corners[3]:0}
        self.cornerDict = {self.corners[0]:0x1,self.corners[1]:0x2,self.corners[2]:0x4,self.corners[3]:0x8}

    def getStartState(self):
        "Returns the start state (in your state space, not the full Pacman state space)"
        "*** Modified by Du Tianyi 3035149685 ***"
        startPos = self.startingPosition
        startCornerVisited = 0
        return (startPos,startCornerVisited)
        #util.raiseNotDefined()

    def isGoalState(self, state):
        "Returns whether this search state is a goal state of the problem"
        "*** Modified by Du Tianyi 3035149685 ***"
        isGoal = False
        curPos,curCornerVisited = state
        if curPos in self.corners:
            #if 0 in curCornerVisited and 1 in curCornerVisited and 2 in curCornerVisited and 3 in curCornerVisited:
            if curCornerVisited == 0xF:
                #print "expanded:",self._expanded
                isGoal = True

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state[0])
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal
        #util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """
        successors = []
        #print "successor"
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** Modified by Du Tianyi 3035149685 ***"
            curPos,curCornerVisited = state
            x,y = curPos
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextPos = (nextx,nexty)
                nextCornerVisited = curCornerVisited
                if nextPos in self.corners:
                    nextCornerVisited |= self.cornerDict[nextPos]
                cost = 1
                successors.append(((nextPos,nextCornerVisited),action,cost))

        self._expanded += 1

        if state not in self._visitedlist:
            self._visited[state] = True
            self._visitedlist.append(state[0])

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        "*** Modified by Du Tianyi 3035149685 ***"
        #print "*FUNC:getCostOfActions"
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem; i.e.
    it should be admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** Modified by Du Tianyi 3035149685 ***"
    curPos,curCornerVisited = state
    cornerUnVisited = []
    hVal = 0

    bitPos = 0
    #print "curState:",state
    while bitPos < 4:
        if not curCornerVisited & 0x1:
            cornerUnVisited.append(corners[bitPos])
        bitPos += 1
        curCornerVisited = curCornerVisited >> 1
    #print "cornerUnVisited:",cornerUnVisited
    distFrontier = []
    while cornerUnVisited:
        for corner in cornerUnVisited:
            distFrontier.append([util.manhattanDistance(curPos,corner),corner])
        cost,corner = min(distFrontier,key = lambda para: para[0])
        #print "nearest corner:",corner,cost
        cornerUnVisited.remove(corner)
        del distFrontier[:]
        curPos = corner
        hVal += cost

    return hVal


class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

#added
# def euclideanDistance(position1, position2, info={}):
#     "The Euclidean distance"
#     xy1 = position1
#     xy2 = position2
#     return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

# def costNormalization(cost, minCost, maxCost, minDist, maxDist, info={}):
#     result = (cost - minCost)/(maxCost - minCost) * (maxDist - minDist) + minDist
#     return cost
def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a
    Grid (see game.py) of either True or False. You can call foodGrid.asList()
    to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, problem.walls gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use. For example,
    if you only want to count the walls once and store that value, try:
      problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** Modified by Du Tianyi 3035149685 ***"
    #1 number of food unvisited: 12517
    # hVal=0
    # for y in range(foodGrid.height):
    #     for x in range(foodGrid.width):
    #         if (foodGrid[x][y]):
    #             hVal+=1
    # return hVal

    #2 manhattan dist of closest and farest food: 8178
    # hVal=0
    # foodCount = 0
    # tempFoodGrid=foodGrid.deepCopy()
    # nextPosition=position[:]
    # foodCount = foodGrid.count()
    # distFrontier = []
    # getMaxFlag = False
    # #if foodCount > 2:
    # foodCount = 2
    # #print "curPos:",position
    # while foodCount:
    #     for i in range(foodGrid.width):
    #         for j in range(foodGrid.height):
    #             if tempFoodGrid[i][j]:
    #                 distFrontier.append([(i,j),util.manhattanDistance(nextPosition,(i,j))])
    #                 #distFrontier.append([(i,j),euclideanDistance(nextPosition,(i,j))])
    #     #print "nextPos:",nextPosition
    #     #print "distFrontier:",distFrontier
    #     if distFrontier:
    #         #if foodCount == 1 and getMaxFlag:
    #         if foodCount == 1:
    #             foodPosition,cost = max(distFrontier,key = lambda para: para[1])
    #         else:
    #             foodPosition,cost = min(distFrontier,key = lambda para: para[1])
    #             #getMaxFlag = True
    #         tempFoodGrid[foodPosition[0]][foodPosition[1]] = False
    #         nextPosition = (foodPosition[0],foodPosition[1])
    #         hVal += cost
    #         del distFrontier[:]
    #     foodCount -= 1
    # #print "hVal:",hVal
    # return hVal

    #3 manhattan dist of all foods through the path: 6126 (Not admissible)
    # hVal = 0
    # foodCount = foodGrid.count()
    # tempFoodGrid=foodGrid.deepCopy()
    # nextPosition=position[:]

    # distFrontier = []
    # #print "curPos:",position
    # while foodCount:
    #     for i in range(foodGrid.width):
    #         for j in range(foodGrid.height):
    #             if tempFoodGrid[i][j]:
    #                 distFrontier.append([(i,j),util.manhattanDistance(nextPosition,(i,j))])
    #                 #distFrontier.append([(i,j),euclideanDistance(nextPosition,(i,j))])
    #     #print "nextPos:",nextPosition
    #     #print "distFrontier:",distFrontier
    #     if distFrontier:
    #         foodPosition,cost = min(distFrontier,key = lambda para: para[1])
    #         tempFoodGrid[foodPosition[0]][foodPosition[1]] = False
    #         nextPosition = (foodPosition[0],foodPosition[1])
    #         hVal += cost
    #         foodCount -= 1
    #         del distFrontier[:]
    #     else:
    #         return 0
    # #if hVal > 1:
    # #    hVal /= 2
    # #print "hVal:",hVal
    # return hVal

    #4 average manhattan dist from unvisited foods: 
    # hVal = 0
    # sumCost = 0
    # sumDist = 0
    # foodCount = foodGrid.count()
    # tempFoodGrid=foodGrid.deepCopy()
    # nextPosition = position[:]
    # distFrontier = []
    # if foodCount == 1:
    #     for i in range(tempFoodGrid.width):
    #         for j in range(tempFoodGrid.height):
    #             if tempFoodGrid[i][j]:
    #                 return util.manhattanDistance((i,j),position)

    # while foodCount:
    #     for i in range(tempFoodGrid.width):
    #         for j in range(tempFoodGrid.height):
    #             if tempFoodGrid[i][j]:
    #                 #print "compute", (i,j)
    #                 cost = 0
    #                 leftFoodCount = 0
    #                 for x in range(tempFoodGrid.width):
    #                     for y in range(tempFoodGrid.height):
    #                         if tempFoodGrid[x][y]:
    #                             if x != i or y != j:
    #                                 #print "to",(x,y),":",util.manhattanDistance((i,j),(x,y))
    #                                 cost += util.manhattanDistance((i,j),(x,y))
    #                                 leftFoodCount += 1
    #                 if leftFoodCount:
    #                     if cost < leftFoodCount:
    #                         distFrontier.append([(i,j),1,util.manhattanDistance((i,j),(nextPosition[0],nextPosition[1]))])    
    #                     else:
    #                         distFrontier.append([(i,j),cost/leftFoodCount,util.manhattanDistance((i,j),(nextPosition[0],nextPosition[1]))])
    #     #print "curPos:",nextPosition
    #     #print "distFrontier:",distFrontier
    #     if distFrontier:
    #         #foodPosition,cost,dist = min(distFrontier,key = lambda para: para[1])
    #         foodPosition,cost,dist = min(distFrontier,key = lambda para: para[1])

    #         tempFoodGrid[foodPosition[0]][foodPosition[1]] = False
    #         nextPosition = foodPosition[:]
    #         #sumCost += cost
    #         #sumDist += dist
    #         #hVal += cost
    #         hVal += (cost + dist) / 2
    #         del distFrontier[:]
    #     foodCount -= 1
    # #hVal = sumDist/2 + (sumCost / 2) / tempFoodGrid.width
    # #print "hVal:",hVal
    # return hVal

    # 5 use manhattan dist and the dist from target node to other unvisited nodes as heuristic value
    hVal = 0
    sumCost = 0
    sumDist = 0
    foodCount = foodGrid.count()
    tempFoodGrid=foodGrid.deepCopy()
    nextPosition = position[:]
    distFrontier = []
    if foodCount == 1:
        for i in range(tempFoodGrid.width):
            for j in range(tempFoodGrid.height):
                if tempFoodGrid[i][j]:
                    return util.manhattanDistance((i,j),position)

    while foodCount:
        for i in range(tempFoodGrid.width):
            for j in range(tempFoodGrid.height):
                if tempFoodGrid[i][j]:
                    #print "compute", (i,j)
                    cost = 0
                    leftFoodCount = 0
                    for x in range(tempFoodGrid.width):
                        for y in range(tempFoodGrid.height):
                            if tempFoodGrid[x][y]:
                                if x != i or y != j:
                                    #print "to",(x,y),":",util.manhattanDistance((i,j),(x,y))
                                    cost += util.manhattanDistance((i,j),(x,y))
                                    leftFoodCount += 1
                    if leftFoodCount:
                        if cost < leftFoodCount:
                            distFrontier.append([(i,j),1,util.manhattanDistance((i,j),(nextPosition[0],nextPosition[1]))])    
                        else:
                            distFrontier.append([(i,j),cost/leftFoodCount,util.manhattanDistance((i,j),(nextPosition[0],nextPosition[1]))])
        #print "curPos:",nextPosition
        #print "distFrontier:",distFrontier
        if distFrontier:
            #foodPosition,cost,dist = min(distFrontier,key = lambda para: para[1])
            #maxFoodPosition,maxCost,maxDist = max(distFrontier,key = lambda para: para[1])
            #minFoodPosition,minCost,minDist = min(distFrontier,key = lambda para: para[1])
            distFrontier.sort(key = lambda para: para[1])
            finalFrontier = []
            costRank = 1
            for curPosition,curCost,curDist in distFrontier:
                #finalCost = (curDist + ((curCost - minCost)/(maxCost - minCost) * (maxDist - minDist) + minDist))/2
                costRank = (costRank * tempFoodGrid.width) / len(distFrontier)
                finalCost = (curDist + costRank) / 2
                costRank += 1
                finalFrontier.append([curPosition,finalCost])
            finalFoodPosition,finalCost = min(finalFrontier,key = lambda para: para[1])

            tempFoodGrid[finalFoodPosition[0]][finalFoodPosition[1]] = False
            nextPosition = finalFoodPosition[:]
            #sumCost += cost
            #sumDist += dist
            hVal += finalCost
            #hVal += (cost + dist) / 2
            #hVal += cost
            del distFrontier[:]
        foodCount -= 1
    #hVal = sumDist/2 + (sumCost / 2) / tempFoodGrid.width
    #print "hVal:",hVal
    return hVal

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        "Returns a path (a list of actions) to the closest dot, starting from gameState"
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** Modified by Du Tianyi 3035149685 ***"
        dist = []
        x,y = startPosition
        for i in range(food.width):
            for j in range(food.height):
                if food[i][j]:
                    dist.append([(i,j),mazeDistance(startPosition,(i,j),gameState)])
                    #print "from",startPosition,"to",(j,i)
        closestDot,closestDist = min(dist,key = lambda para: para[1])
        prob = PositionSearchProblem(gameState, start=startPosition, goal=closestDot, warn=False, visualize=False)
        return search.bfs(prob)
        #path = search.bfs(prob)
        #print "path:",path
        #return path
        #util.raiseNotDefined()

class AnyFoodSearchProblem(PositionSearchProblem):
    """
      A search problem for finding a path to any food.

      This search problem is just like the PositionSearchProblem, but
      has a different goal test, which you need to fill in below.  The
      state space and successor function do not need to be changed.

      The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
      inherits the methods of the PositionSearchProblem.

      You can use this search problem to help you fill in
      the findPathToClosestDot method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()
        #print "row:",self.food.height
        #print "col:",self.food.width
        #print "food:",self.food
        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test
        that will complete the problem definition.
        """
        x,y = state

        "*** Modified by Du Tianyi 3035149685 ***"
        """count = 0
        for i in range(self.food.height):
            for j in range(self.food.width):
                if self.food[i][j]:
                    count += 1
        if not count:
            return True
        else:
            return False"""
        return self.food[x][y]
        #util.raiseNotDefined()

##################
# Mini-contest 1 #
##################

class ApproximateSearchAgent(Agent):
    "Implement your contest entry here.  Change anything but the class name."

    def registerInitialState(self, state):
        "This method is called before any moves are made."
        "*** YOUR CODE HERE ***"
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        From game.py:
        The Agent will receive a GameState and must return an action from
        Directions.{North, South, East, West, Stop}
        """
        "*** YOUR CODE HERE ***"
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP
        #util.raiseNotDefined()

    def __init__(self, fn='aStarSearch', prob='FoodSearchProblem', heuristic='anotherFoodHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

def anotherFoodHeuristic(state, problem):
    position, foodGrid = state
    "*** Modified by Du Tianyi 3035149685 ***"
    #4 average manhattan dist from unvisited foods: 
    hVal = 0
    sumCost = 0
    sumDist = 0
    foodCount = foodGrid.count()
    tempFoodGrid=foodGrid.deepCopy()
    nextPosition = position[:]
    distFrontier = []
    if foodCount == 1:
        for i in range(tempFoodGrid.width):
            for j in range(tempFoodGrid.height):
                if tempFoodGrid[i][j]:
                    return util.manhattanDistance((i,j),position)

    while foodCount:
        for i in range(tempFoodGrid.width):
            for j in range(tempFoodGrid.height):
                if tempFoodGrid[i][j]:
                    #print "compute", (i,j)
                    cost = 0
                    leftFoodCount = 0
                    for x in range(tempFoodGrid.width):
                        for y in range(tempFoodGrid.height):
                            if tempFoodGrid[x][y]:
                                if x != i or y != j:
                                    #print "to",(x,y),":",util.manhattanDistance((i,j),(x,y))
                                    cost += util.manhattanDistance((i,j),(x,y))
                                    leftFoodCount += 1
                    if leftFoodCount:
                        if cost < leftFoodCount:
                            distFrontier.append([(i,j),1,util.manhattanDistance((i,j),(nextPosition[0],nextPosition[1]))])    
                        else:
                            distFrontier.append([(i,j),cost/leftFoodCount,util.manhattanDistance((i,j),(nextPosition[0],nextPosition[1]))])
        #print "curPos:",nextPosition
        #print "distFrontier:",distFrontier
        if distFrontier:
            #foodPosition,cost,dist = min(distFrontier,key = lambda para: para[1])
            foodPosition,cost,dist = min(distFrontier,key = lambda para: para[1])

            tempFoodGrid[foodPosition[0]][foodPosition[1]] = False
            nextPosition = foodPosition[:]
            #sumCost += cost
            #sumDist += dist
            #hVal += cost
            hVal += (cost + dist) / 2
            del distFrontier[:]
        foodCount -= 1
    #hVal = sumDist/2 + (sumCost / 2) / tempFoodGrid.width
    #print "hVal:",hVal
    return hVal

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built.  The gameState can be any game state -- Pacman's position
    in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
