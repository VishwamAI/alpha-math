import numpy as np
from scipy.integrate import odeint

class EvolutionaryGameTheory:
    def __init__(self, payoff_matrix):
        """
        Initialize the Evolutionary Game Theory model.

        :param payoff_matrix: A numpy array representing the payoff matrix
        """
        self.payoff_matrix = np.array(payoff_matrix)
        self.num_strategies = self.payoff_matrix.shape[0]

    def replicator_dynamics(self, x, t):
        """
        Define the replicator dynamics equations.

        :param x: Current population state
        :param t: Time (not used, but required for odeint)
        :return: Rate of change of population state
        """
        fitness = np.dot(self.payoff_matrix, x)
        mean_fitness = np.dot(x, fitness)
        return x * (fitness - mean_fitness)

    def simulate(self, initial_state, time_points):
        """
        Simulate the evolutionary dynamics.

        :param initial_state: Initial population state
        :param time_points: Array of time points for simulation
        :return: Array of population states over time
        """
        return odeint(self.replicator_dynamics, initial_state, time_points)

    def find_equilibrium(self, initial_state, time_points):
        """
        Find the equilibrium state of the system.

        :param initial_state: Initial population state
        :param time_points: Array of time points for simulation
        :return: Final population state (approximate equilibrium)
        """
        simulation = self.simulate(initial_state, time_points)
        return simulation[-1]

    def analyze_stability(self, equilibrium, perturbation=1e-6):
        """
        Analyze the stability of an equilibrium point.

        :param equilibrium: The equilibrium state to analyze
        :param perturbation: Size of perturbation to apply
        :return: Boolean indicating if the equilibrium is stable
        """
        perturbed_states = []
        for i in range(self.num_strategies):
            perturbed = equilibrium.copy()
            perturbed[i] += perturbation
            perturbed /= np.sum(perturbed)  # Normalize
            perturbed_states.append(perturbed)

        for state in perturbed_states:
            final_state = self.find_equilibrium(state, np.linspace(0, 100, 1000))
            if not np.allclose(final_state, equilibrium, atol=1e-3):
                return False
        return True

# Example usage
if __name__ == "__main__":
    # Example: Hawk-Dove game
    payoff_matrix = np.array([[0, 2], [1, 1]])
    egt = EvolutionaryGameTheory(payoff_matrix)

    initial_state = np.array([0.5, 0.5])
    time_points = np.linspace(0, 100, 1000)

    equilibrium = egt.find_equilibrium(initial_state, time_points)
    print("Equilibrium state:", equilibrium)

    is_stable = egt.analyze_stability(equilibrium)
    print("Is equilibrium stable?", is_stable)

    # Simulate and plot the dynamics
    import matplotlib.pyplot as plt

    simulation = egt.simulate(initial_state, time_points)
    plt.plot(time_points, simulation)
    plt.xlabel('Time')
    plt.ylabel('Population fraction')
    plt.legend(['Strategy 1', 'Strategy 2'])
    plt.title('Evolutionary Game Dynamics')
    plt.show()
