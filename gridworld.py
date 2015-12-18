from pylab import *
import numpy
from time import sleep

class Gridworld:
    """
    A class that implements a quadratic NxN gridworld.

    Methods:

    learn(N_trials=100)  : Run 'N_trials' trials. A trial is finished, when the agent reaches the reward location.
    visualize_trial()  : Run a single trial with graphical output.
    reset()            : Make the agent forget everything he has learned.
    plot_Q()           : Plot of the Q-values .
    learning_curve()   : Plot the time it takes the agent to reach the target as a function of trial number.
    navigation_map()   : Plot the movement direction with the highest Q-value for all positions.
    """

    def __init__(self,N,reward_position=(0.8, 0.8),obstacle=False, lambda_eligibility=0.95):
        """
        Creates a quadratic NxN gridworld.

        Mandatory argument:
        N: size of the gridworld

        Optional arguments:
        reward_position = (x_coordinate,y_coordinate): the reward location
        obstacle = True:  Add a wall to the gridworld.
        """

        # gridworld size
        self.N = N

        self.worldWidth = 1.0

        # reward location
        self.reward_position = reward_position

        # reward administered t the target location and when
        # bumping into walls
        self.reward_at_target = 10.
        self.reward_at_wall   = -2

        # probability at which the agent chooses a random
        # action. This makes sure the agent explores the grid.
        self.start_epsilon = 0.5
        self.epsilon = self.start_epsilon
        # learning rate
        self.eta = 0.05

        # discount factor - quantifies how far into the future
        # a reward is still considered important for the
        # current action
        self.gamma = 0.95

        # the decay factor for the eligibility trace the
        # default is 0., which corresponds to no eligibility
        # trace at all.
        self.lambda_eligibility = lambda_eligibility

        # is there an obstacle in the room?
        self.obstacle = obstacle

        self.reward_radius = 0.1
        self.step_size = 0.03
        # initialize the Q-values etc.
        self._init_run()
        self.sigma = self.worldWidth / self.N
        print "Sigma: ", self.sigma

    def r(self, x, y, sx, sy):
        # First we transform grid coordinates into world coordinates
        x_temp = x * self.sigma
        y_temp = y * self.sigma
        #print "r: ", x_temp, " , ", y_temp , " , ", sx , " , ", sy
        val = numpy.exp(-(pow(x_temp - sx, 2.0) + pow(y_temp - sy, 2.0)) / (2.0 * pow(self.sigma, 2.0)))
        #print val
        #raw_input()
        return val

    def run(self,N_trials=10,N_runs=1):
        self.latencies = zeros(N_trials)

        for run in range(N_runs):
            self.epsilon = self.start_epsilon
            self._init_run()
            latencies = self._learn_run(N_trials=N_trials)
            #latencies = self._learn_run()
            self.latencies += latencies/N_runs
            print "Number of steps: ", latencies
            print "Neurons: ", self.w
            self.navigation_map()
            self.learning_curve()
        print "Mean number of steps: ", self.latencies     

    def visualize_trial(self):
        """
        Run a single trial with a graphical display that shows in
                red   - the position of the agent
                blue  - walls/obstacles
                green - the reward position

        Note that for the simulation, exploration is reduced -> self.epsilon=0.1

        """
        # store the old exploration/exploitation parameter
        epsilon = self.epsilon

        # favor exploitation, i.e. use the action with the
        # highest Q-value most of the time
        self.epsilon = 0.1

        self._run_trial(visualize=True)

        # restore the old exploration/exploitation factor
        self.epsilon = epsilon

    def learning_curve(self,log=False,filter=1.):
        """
        Show a running average of the time it takes the agent to reach the target location.

        Options:
        filter=1. : timescale of the running average.
        log    : Logarithmic y axis.
        """
        figure()
        xlabel('trials')
        ylabel('time to reach target')
        latencies = array(self.latency_list)
        # calculate a running average over the latencies with a averaging time 'filter'
        for i in range(1,latencies.shape[0]):
            latencies[i] = latencies[i-1] + (latencies[i] - latencies[i-1])/float(filter)

        if not log:
            plot(self.latencies)
        else:
            semilogy(self.latencies)

    def navigation_map(self):
        """
        Plot the direction with the highest Q-value for every position.
        Useful only for small gridworlds, otherwise the plot becomes messy.
        """
        print "Naviagation map"
        self.x_direction = numpy.zeros((self.N,self.N))
        self.y_direction = numpy.zeros((self.N,self.N))

        # First compute the table of Q's
        tempQ = 0.01 * numpy.random.rand(self.N, self.N, 8)
        print "-- Computing the Q's for the navigation map"
        for x in range(self.N):
            for y in range(self.N):
                for a in range(8):
                    tempQ[x, y, a] = self._compute_Q(x, y, a)

        print "-- Compute actions"
        tempQ = self.w
        self.actions = argmax(tempQ[:,:,:],axis=2)
        print self.actions
        self.x_direction[self.actions==0] = 1.
        self.x_direction[self.actions==1] = 0.5
        self.x_direction[self.actions==2] = 0.
        self.x_direction[self.actions==3] = -0.5
        self.x_direction[self.actions==4] = -1.
        self.x_direction[self.actions==5] = -0.5
        self.x_direction[self.actions==6] = 0.
        self.x_direction[self.actions==7] = 0.5


        self.y_direction[self.actions==0] = 0.
        self.y_direction[self.actions==1] = 0.5
        self.y_direction[self.actions==2] = 1.
        self.y_direction[self.actions==3] = 0.5
        self.y_direction[self.actions==4] = 0.
        self.y_direction[self.actions==5] = -0.5
        self.y_direction[self.actions==6] = -1.
        self.y_direction[self.actions==7] = -0.5

        print "-- figure:"
        figure()
        quiver(self.x_direction,self.y_direction)
        axis([-0.5, self.N - 0.5, -0.5, self.N - 0.5])

        show()
        print "--it's shown"

        raw_input()
        close()

    def reset(self):
        """
        Reset the Q-values (and the latency_list).

        Instant amnesia -  the agent forgets everything he has learned before
        """
        #self.Q = numpy.random.rand(self.N,self.N, 8)
        self.w = numpy.zeros.rand(self.N,self.N, 8)
        self.latency_list = []

    def plot_Q(self):
        """
        Plot the dependence of the Q-values on position.
        The figure consists of 4 subgraphs, each of which shows the Q-values
        colorcoded for one of the actions.
        """
        figure()
        for i in range(4):
            subplot(2,2,i+1)
            imshow(self.Q[:,:,i],interpolation='nearest',origin='lower',vmax=1.1)
            if i==0:
                title('Up')
            elif i==1:
                title('Down')
            elif i==2:
                title('Right')
            else:
                title('Left')

            colorbar()
        draw()

    ###############################################################################################
    # The remainder of methods is for internal use and only relevant to those of you
    # that are interested in the implementation details
    ###############################################################################################


    def _init_run(self):
        """
        Initialize the Q-values, eligibility trace, position etc.
        """
        # initialize the Q-values and the eligibility trace
        #self.Q = 0.01 * numpy.random.rand(self.N, self.N, 8) + 0.1
        self.e = numpy.zeros((self.N, self.N, 8))
        self.w = numpy.zeros((self.N, self.N, 8))

        # list that contains the times it took the agent to reach the target for all trials
        # serves to track the progress of learning
        self.latency_list = []

        # initialize the state and action variables
        self.x_position = None
        self.y_position = None
        self.action = None

    def _learn_run(self,N_trials=10):
        """
        Run a learning period consisting of N_trials trials.

        Options:
        N_trials :     Number of trials

        Note: The Q-values are not reset. Therefore, running this routine
        several times will continue the learning process. If you want to run
        a completely new simulation, call reset() before running it.

        """
        for trial in range(N_trials):
            # run a trial and store the time it takes to the target
            latency = self._run_trial()
            self.latency_list.append(latency)
            self.epsilon = max(0.996 * self.epsilon, 0.001)
            print "e: ", self.epsilon, " n of steps: ", latency
            #self.navigation_map()


        return array(self.latency_list)

    def _run_trial(self,visualize=False):
        """
        Run a single trial on the gridworld until the agent reaches the reward position.
        Return the time it takes to get there.

        Options:
        visual: If 'visualize' is 'True', show the time course of the trial graphically
        """

        print "Run trial"

        #print "Init position"
        self.x_position = 0.1
        self.y_position = 0.1
        
        # initialize the latency (time to reach the target) for this trial
        latency = 0.

        # start the visualization, if asked for
        if visualize:
            self._init_visualization()

        #print "Enter the while loop"
        # run the trial
        self._choose_action()
        while not self._arrived():
        #    print "Step: "
            self._update_state()
            self._choose_action()
        #    print "Action -> ", self.action
            self._update_W()
            if visualize:
                self._visualize_current_state()

            latency = latency + 1
            if latency > 3000:
                print "Position: (", self.x_position, ",", self.y_position, ")"
            if latency > 2900:
                break
            #sleep(0.2)

        if visualize:
            self._close_visualization()
        return latency

    def _compute_Q(self, sx, sy, a):
        """
        Q cannot be stored so we have to compute it online using the w's
        """
        acc = 0.0
        # iterates over every cells
        for x in range(self.N):
            for y in range(self.N):
                # Optimization, trying to skip some computations
                #if (self.w[x, y, a] > 0):
                if abs(x * self.sigma - sx) < 4 * self.sigma  and abs(y * self.sigma - sy) < 4 * self.sigma:
                   acc += self.r(x, y, sx, sy) * self.w[x, y, a]
                   #print x* self.sigma, y* self.sigma, sx, sy
        #print "Computed Q: ", acc, " pos: (", self.x_position, ",",self.y_position, ")"
        return acc       


    def _update_Q(self):
        """
        Update the current estimate of the Q-values according to SARSA.
        """
        # update the eligibility trace
        self.e = self.lambda_eligibility * self.e
        self.e[self.x_position_old, self.y_position_old,self.action_old] += 1.

        # update the Q-values
        if self.action_old != None:
            self.Q +=     \
                self.eta * self.e *\
                (self._reward()  \
                - ( self.Q[self.x_position_old,self.y_position_old,self.action_old] \
                - self.gamma * self.Q[self.x_position, self.y_position, self.action] )  )

    def _update_W(self):
        """
        Update the weights in the neural network.
        """
        
        # TOCHECK update the eligibility trace
        self.e = self.gamma * self.lambda_eligibility * self.e
        for x in range(self.N):
            for y in range(self.N): 
                self.e[x, y, self.action_old] += self.r(x , y, self.x_position_old, self.y_position_old)

        # Compute the detla
        delta = self._reward() \
         - self._compute_Q(self.x_position_old, self.y_position_old, self.action_old)\
         + self.gamma * self._compute_Q(self.x_position, self.y_position, self.action)

        # Update the weights
        self.w += self.eta * delta * self.e 

    def _choose_action(self):
        """
        Choose the next action based on the current estimate of the Q-values.
        The parameter epsilon determines, how often agent chooses the action
        with the highest Q-value (probability 1-epsilon). In the rest of the cases
        a random action is chosen.
        """
        self.action_old = self.action
        if numpy.random.rand() < self.epsilon:
            self.action = numpy.random.randint(8)
        else:
            maxExpR = -sys.maxint - 1
            bestAction = -1
            for a in range(8):
                temp = self._compute_Q(self.x_position, self.y_position, a)
                if temp > maxExpR:
                    maxExpR = temp
                    bestAction = a
            self.action = bestAction

    def _arrived(self):
        """
        Check if the agent has arrived.
        """
        distance = numpy.sqrt(pow(self.reward_position[0] - self.x_position, 2.0) + pow(self.reward_position[1] - self.y_position, 2.0))
        return distance <= self.reward_radius

    def _reward(self):
        """
        Evaluates how much reward should be administered when performing the
        chosen action at the current location
        """
        if self._arrived():
            #print "Got the cheese! -> 1.0 reward"
            return self.reward_at_target

        if self._wall_touch:
            #print "Bumped into the wall -> -2 reward"
            return self.reward_at_wall
        else:
            return 0.

    def _update_state(self):
        """
        Update the state according to the old state and the current action.
        """
        # remember the old position of the agent
        self.x_position_old = self.x_position
        self.y_position_old = self.y_position

        # update the agents position according to the action
        angle = 2.0 * 3.1417 * self.action / 8.0

        dx = self.step_size * cos(angle)
        dy = -self.step_size * sin(angle)
        
        if self.action >= 8:
            print "There must be a bug. This is not a valid action!"

        self.x_position = self.x_position + dx
        self.y_position = self.y_position + dy

        # check if the agent has bumped into a wall.
        if self._is_wall():
            self.x_position = self.x_position_old
            self.y_position = self.y_position_old
            self._wall_touch = True
        else:
            self._wall_touch = False

        #print "Action ",self.action, " X: ", self.x_position, " Y: ", self.y_position
        #raw_input()

    def _is_wall(self,x_position=None,y_position=None):
        """
        This function returns, if the given position is within an obstacle
        If you want to put the obstacle somewhere else, this is what you have
        to modify. The default is a wall that starts in the middle of the room
        and ends at the right wall.

        If no position is given, the current position of the agent is evaluated.
        """
        if x_position == None or y_position == None:
            x_position = self.x_position
            y_position = self.y_position

        # check of the agent is trying to leave the gridworld
        if x_position < 0 or x_position >= self.worldWidth or y_position < 0 or y_position >= self.worldWidth:
            return True

        # check if the agent has bumped into an obstacle in the room
        if self.obstacle:
            if y_position == self.worldWidth / 2 and x_position> worldWidth / 2:
                return True

        # if none of the above is the case, this position is not a wall
        return False

    def _visualize_current_state(self):
        """
        Show the gridworld. The squares are colored in
        red - the position of the agent - turns yellow when reaching the target or running into a wall
        blue - walls
        green - reward
        """

        # set the agents color
        self._display[self.x_position_old,self.y_position_old,0] = 0
        self._display[self.x_position_old,self.y_position_old,1] = 0
        self._display[self.x_position,self.y_position,0] = 1
        if self._wall_touch:
            self._display[self.x_position,self.y_position,1] = 1

        # set the reward locations
        self._display[self.reward_position[0],self.reward_position[1],1] = 1

        # update the figure
        self._visualization.set_data(self._display)
        #close()
        imshow(self._display,interpolation='nearest',origin='lower')
        show()
        print "it's shown"
        draw()
        print "it's drown"

        # and wait a little while to control the speed of the presentation
        sleep(0.2)
        print "it ends sleep"
        #close()
        print "it's closed"

    def _init_visualization(self):

        # create the figure
        figure()
        # initialize the content of the figure (RGB at each position)
        self._display = numpy.zeros((self.N,self.N,3))

        # position of the agent
        self._display[self.x_position,self.y_position,0] = 1
        self._display[self.reward_position[0],self.reward_position[1],1] = 1

        for x in range(self.N):
            for y in range(self.N):
                if self._is_wall(x_position=x,y_position=y):
                    self._display[x,y,2] = 1.

                self._visualization = imshow(self._display,interpolation='nearest',origin='lower')

    def _close_visualization(self):
        print "Press <return> to proceed..."
        raw_input()
        close()

if __name__ == '__main__':
    grid = Gridworld(20)
    print "Run the game"
    grid.run(200,1)


# J'ai fait une petite optimization, il prend en compte que les neurons proche 
# au lieux de tout le temps calculer la gaussienne, sinon ca prend trop de temp avec 400 neurons.

# Navigation map affiche pour chaque neuron la direction prefere, je crois pas que c'est encore top.