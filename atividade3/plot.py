import matplotlib.pyplot as plt
import util

def plotManipulability(time, manipulability):
    plt.scatter(time, manipulability, s = 1.5, c = 'k')
    plt.title('System manipulability')
    plt.grid(0.5)
    plt.xlabel('Time(s)')
    plt.ylabel('Manipulability')
    plt.show()

def plotCoordinates(time, effectors):
    plt.plot(time, effectors[0],'g', label = 'Position X', linewidth = 1.5)
    plt.plot(time, effectors[1],'b', label = 'Position Y', linewidth = 1.5)
    plt.plot(time, effectors[2],'r', label = 'Position Z', linewidth = 1.5)
    plt.legend(loc='best', framealpha=1)
    plt.title('Coordinates')
    plt.grid(0.5)
    plt.xlabel('Time(s)')
    plt.ylabel('Coordinates')
    plt.show() 

def plotJoins(time, thetas):
    plt.plot(time, thetas[0],'g', label = 'Θ 1', linewidth = 1.5)
    plt.plot(time, thetas[1],'b', label = 'Θ 2', linewidth = 1.5)
    plt.plot(time, thetas[2],'r', label = 'Θ 3', linewidth = 1.5)
    plt.plot(time, thetas[3],'k', label = 'Θ 4', linewidth = 1.5)
    plt.plot(time, thetas[4],'y', label = 'Θ 5', linewidth = 1.5)
    plt.legend(loc='best', framealpha=1)
    plt.title('Joints')
    plt.grid(0.5)
    plt.xlabel('Time(s)')
    plt.ylabel('Joints')
    plt.show()

def showAllPlots(q_inicial, q, time, manipulability, effectors, thetas):
    plotManipulability(time, manipulability)
    plotCoordinates(time, effectors)
    plotJoins(time, thetas)
    print("Initial jacobian Matrix:\n", util.generateJacobian(q_inicial))
    print("\nFinal jacobian Matrix:\n", util.generateJacobian(q))