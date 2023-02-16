# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #Cantidad de iteraciones que deben suceder al calcular para encontrar el valor opt
        for i in range(self.iterations):
            states = self.mdp.getStates()
            #Tabla de valores optimos de cada iteración/estado
            table = util.Counter()
            for s in states:
                mvtPos = self.mdp.getPossibleActions(s)
                #Cantidad de mvtos posibles
                if len(mvtPos) == 0:
                    maxValue = 0
                else:
                    maxValue = -999999
                    for acc in mvtPos:
                        #Mirar el mejor valor
                        qValue = self.computeQValueFromValues(s, acc)
                        if qValue > maxValue:
                            maxValue = qValue
                table[s] = maxValue
            self.values = table



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        ans = 0
        """Formula dada en el punto 1:
        T = T(s, a, s')
        R = R(s, a, s')
        gammaVk = gamma * Vk(s')
        """
        for sPrima, T in self.mdp.getTransitionStatesAndProbs(state, action):
            R = self.mdp.getReward(state, action, sPrima)
            gammaVk = self.discount * self.values[sPrima]
            ans += T * (R + gammaVk)
        return ans

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        maxValue = -999999
        bestAcc = 0
        for acc in self.mdp.getPossibleActions(state):
            qValue = self.computeQValueFromValues(state, acc)
            if qValue > maxValue:
                maxValue = qValue
                bestAcc = acc
        return bestAcc

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            #se usa el modulo porque actualiza un solo estado en cada iteración
            state = states[i % len(states)]
            if not self.mdp.isTerminal(state):
                mvtPos = self.mdp.getPossibleActions(state)
                #Cantidad de mvtos posibles
                if len(mvtPos) == 0:
                    maxValue = 0
                else:
                    maxValue = -999999
                    for acc in mvtPos:
                        qValue = self.computeQValueFromValues(state, acc)
                        if qValue > maxValue:
                            maxValue = qValue
                self.values[state] = maxValue
                
class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        queue = util.PriorityQueue()
        parents = dict()
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                for acc in self.mdp.getPossibleActions(s):
                    for sPrima, T in self.mdp.getTransitionStatesAndProbs(s, acc):
                        if T > 0:
                            if sPrima not in parents:
                                parents[sPrima] = {s}
                            else:
                                parents[sPrima].add(s)
        
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                mvtPos = self.mdp.getPossibleActions(s)
                listQ = []
                for acc in mvtPos:
                    qValue = self.getQValue(s, acc)
                    listQ.append(qValue)
                maxQ = max(listQ)
                diff = abs(self.values[s] - maxQ)
                queue.update(s, -diff)
        
        for i in range(self.iterations):
            if queue.isEmpty():
                break
            s = queue.pop()

            if not self.mdp.isTerminal(s):
                mvtPos = self.mdp.getPossibleActions(s)
                listQ = []
                for acc in mvtPos:
                    qValue = self.getQValue(s, acc)
                    listQ.append(qValue)
                maxQ = max(listQ)
                self.values[s] = maxQ
          
            for father in parents[s]:
                mvtPos = self.mdp.getPossibleActions(father)
                listQ = []
                for acc in mvtPos:
                    qValue = self.getQValue(father, acc)
                    listQ.append(qValue)
                maxQ = max(listQ)
                diff = abs(self.values[father] - maxQ)
                if diff > self.theta:
                    queue.update(father, -diff)

