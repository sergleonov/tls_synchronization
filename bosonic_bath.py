from qutip import basis, sigmax, sigmaz
from qutip.solver.heom import DrudeLorentzBath
from qutip.solver.heom import DrudeLorentzPadeBath
from qutip.solver.heom import HEOMSolver
from qutip import QobjEvo
from qutip import *
import numpy as np
import matplotlib.pyplot as plt

# ------------------System and bath parameters------------------
Del = 0  # detuning term
omega0 = 1.0  # GHZ frequency of the bath modes (normalized)
Omega = 60*omega0  # GHz frequency of the driving field
omega_drive = Omega/2.4048  # GHz frequency of the driving field (resonant)

# Bath properties:
gamma = omega0/2 # GHz cut off frequency
lam = omega0  # coupling strength
T = 0.5  # K temperature

# solver parameters:
Nk = 2 # expansion terms
time_steps = 10000
solver_steps = 15000
max_depth = 5  # maximum hierarchy depth to retain

# initial state of the system
psi0 = basis(2, 0) # ground state

# time vector
tmax = 30/omega0

# ------------------Simulation------------------
def simulate_tls_dynamics(psi0=psi0, time_steps=time_steps, solver_steps=solver_steps, max_depth=max_depth):
    # time independent part of the Hamiltonian:
    H_sys = 0.5 * omega0 * sigmaz() + 0.5 * Del * sigmax() # static Hamiltonian

    # time dependent part of the Hamiltonian:
    def H_coeff(t, args):
        return 0.5 * args["Omega"] * np.cos(args["omega_drive"] * t)
    H_tot = [H_sys, [sigmax(), H_coeff]]

    # total Hamiltonian
    H_tot = QobjEvo(H_tot, args={'omega_drive': omega_drive, 'Omega': Omega})

    # define bath and solver
    Q = sigmax() # coupling operator
    bath = DrudeLorentzPadeBath(Q, lam=lam, gamma=gamma, T=T, Nk=Nk)
    options = {"nsteps": solver_steps, "progress_bar": ''}
    solver = HEOMSolver(H_tot, bath, max_depth=max_depth, options=options)

    # initial density matrix
    rho0 = psi0 * psi0.dag()

    # run simulation
    tlist = np.linspace(0, tmax, time_steps)
    result = solver.run(rho0, tlist)
    
    return result
    


# -----------------Plotting the results-----------------
def plot_bloch_sphere(result, tmax=tmax, filename="bloch_sphere.png"):
    # get the bloch vector components from the expectation values of the operators
    xs = expect(sigmax(), result.states)
    ys = expect(sigmay(), result.states)
    zs = expect(sigmaz(), result.states)    

    b = Bloch()
    b.view = [-45, 30]
    cmap = plt.get_cmap('inferno')  # choose a colormap
    colors = cmap(np.divide(result.times, tmax))  # normalize time to [0, 1] for color mapping

    for i in range(len(xs) - 1):
        segment = [[xs[i], xs[i+1]], 
                [ys[i], ys[i+1]], 
                [zs[i], zs[i+1]]]
        b.add_points(segment, meth='l') # 'l' tells QuTiP to render as a line  
    colors[0] = [0, 0.5, 1, 1] # color first point differently for better visibility
    b.point_color = colors
    b.render() 
    fig = b.fig
    ax = b.axes

    # add colorbar and save
    ax.set_position([0.0, 0.1, 0.7, 0.8]) 
    cax = fig.add_axes([0.85, 0.2, 0.03, 0.6]) 
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Time', rotation=270, labelpad=15)
    plt.subplots_adjust(right=0.8)

    plt.savefig(filename)


def main():
    result = simulate_tls_dynamics()
    plot_bloch_sphere(result)


if __name__ == "__main__":
    main()
