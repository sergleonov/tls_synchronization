from qutip import *
from qutip.solver.heom import DrudeLorentzBath
from qutip.solver.heom import DrudeLorentzPadeBath
from qutip.solver.heom import HEOMSolver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------System and bath parameters------------------
Del = 0  # detuning term
omega0 = 1.0  # GHZ frequency of the bath modes (normalized)
Omega = 60*omega0  # GHz frequency of the driving field
omega_drive = Omega/2.4048  # GHz frequency of the driving field (resonant)
freq_list = np.linspace(omega_drive - 3, omega_drive + 3, 800) # driving freqs 

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

# ------------------Simulation for 1 TLS------------------
def simulate_tls_dynamics(psi0=psi0, time_drive=tmax, time_steps=time_steps, solver_steps=solver_steps, max_depth=max_depth, Q=sigmax(), omega_drive=omega_drive):
    # check that driving time is less than total simulation time
    if time_drive > tmax:
        raise ValueError("Driving time must be less than total simulation time.")

    # time independent part of the Hamiltonian:
    H_sys = 0.5 * omega0 * sigmaz() + 0.5 * Del * sigmax() # static Hamiltonian

    # time dependent part of the Hamiltonian:
    def H_coeff(t, args):
        return 0.5 * args["Omega"] * np.cos(args["omega_drive"] * t)
    H_tot = [H_sys, [sigmax(), H_coeff]]

    # total Hamiltonian
    H_tot = QobjEvo(H_tot, args={'omega_drive': omega_drive, 'Omega': Omega})

    # define bath and solver
    bath = DrudeLorentzPadeBath(Q, lam=lam, gamma=gamma, T=T, Nk=Nk)
    options = {"nsteps": solver_steps, "progress_bar": ''}
    solver_drive = HEOMSolver(H_tot, bath, max_depth=max_depth, options=options)

    # initial density matrix
    rho0 = psi0 * psi0.dag()
    
    # run simulation
    drive_steps = int(time_drive/tmax * time_steps)
    tlist_drive = np.linspace(0, time_drive, drive_steps)
    result_drive = solver_drive.run(rho0, tlist_drive)

    # extract drive evolution
    xs = np.zeros((time_steps))
    ys = np.zeros((time_steps))
    zs = np.zeros((time_steps))
    t = result_drive.times

    for i in range(drive_steps):
        rho_t = result_drive.states[i]
        xs[i] = (rho_t * sigmax()).tr().real
        ys[i] = (rho_t * sigmay()).tr().real
        zs[i] = (rho_t * sigmaz()).tr().real

    # evolve post driving to observe relaxation
    if tmax > time_drive:
        tlist_relax = np.linspace(time_drive, tmax, time_steps - drive_steps)
        solver_relax = HEOMSolver(H_sys, bath, max_depth=max_depth, options=options) # use static Hamiltonian for relaxation    
        result_relax = solver_relax.run(result_drive.states[-1], tlist_relax)

        for i in range(drive_steps, time_steps):
            rho_t = result_relax.states[i - drive_steps]
            xs[i] = (rho_t * sigmax()).tr().real
            ys[i] = (rho_t * sigmay()).tr().real
            zs[i] = (rho_t * sigmaz()).tr().real
        t = np.concatenate((t, result_relax.times))

    return (xs, ys, zs, t)
    


# -----------------Plotting the Bloch sphere-----------------
def plot_bloch_sphere(result, tmax=tmax, filename="bloch_sphere.png"):
    
    xs, ys, zs, t = result

    b = Bloch()
    b.view = [-45, 30]
    cmap = plt.get_cmap('inferno')  # choose a colormap
    colors = cmap(np.divide(t, tmax))  # normalize time to [0, 1] for color mapping

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


#----------------------Plotting heatmap----------------------
def plot_freq_sweep_map(filename="freq_sweep.png",  freq_list=freq_list, psi0=psi0, time_drive = tmax, time_steps=time_steps, solver_steps=solver_steps, max_depth=max_depth, Q=sigmax()):
    # check that driving time is less than total simulation time
    if time_drive > tmax:
        raise ValueError("Driving time must be less than total simulation time.")
    
    # initialize arrays for plotting
    time = None
    xs_all = np.zeros((len(freq_list), time_steps))
    ys_all = np.zeros((len(freq_list), time_steps))
    zs_all = np.zeros((len(freq_list), time_steps))

    # iterate over drive frequencies
    for i in tqdm(range(len(freq_list)), desc="Rendering Subplots"): 
        xs, ys, zs, t = simulate_tls_dynamics(omega_drive=freq_list[i], psi0=psi0, time_drive=time_drive, time_steps=time_steps, 
                                                        solver_steps=solver_steps, max_depth=max_depth, Q=Q)
        if time is None: # initialize time array
            time = t

        # store data
        xs_all[i] = xs
        ys_all[i] = ys
        zs_all[i] = zs

    # transpose results
    xs_all = np.transpose(xs_all)
    ys_all = np.transpose(ys_all)
    zs_all = np.transpose(zs_all)
    
    # plot results
    xs_min, xs_max = -np.abs(xs_all).max(), np.abs(xs_all).max()
    ys_min, ys_max = -np.abs(ys_all).max(), np.abs(ys_all).max()
    zs_min, zs_max = -np.abs(zs_all).max(), np.abs(zs_all).max()
   
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    m0 = axes[0].pcolormesh(freq_list, time, xs_all, cmap ='inferno', shading='auto', vmin = xs_min, vmax = xs_max)
    m1 = axes[1].pcolormesh(freq_list, time, ys_all, cmap ='inferno', shading='auto', vmin = ys_min, vmax = ys_max)
    m2 = axes[2].pcolormesh(freq_list, time, zs_all, cmap ='inferno', shading='auto', vmin = zs_min, vmax = zs_max)
    plt.colorbar(m0)
    plt.colorbar(m1)
    plt.colorbar(m2)

    # add labels
    axes[0].set_title("X expectation")
    axes[1].set_title("Y expectation")
    axes[2].set_title("Z expectation")
    axes[0].set_ylabel('Time')
    axes[1].set_xlabel('Frequency')

    plt.savefig(filename) # save


def main():
    result = simulate_tls_dynamics(time_drive=10/omega0)
    plot_bloch_sphere(result)
    # plot_freq_sweep_map(time_drive=10/omega0)

if __name__ == "__main__":
    main()
