import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os


# Load data from .mat files
btom = sio.loadmat('btom_results_complete.mat')
people = sio.loadmat('data/human_data.mat')

# Set exclusion list and inclusion set
excl = [11, 12, 22, 71, 72]
incl = list(set(range(78)) - set(excl))

# Set indices and plotting parameters
btom['beta_ind'] = 5

desire_axis = [1, 7.5, 1, 7.5]
belief_axis = [0, 1, 0, 0.8]
markersize = 10  # Increased marker size

# Individual trial analyses
desire_people = people['des_inf_mean'][:, incl]
belief_people = people['bel_inf_mean_norm'][:, incl]

# BToM individual model data
desire_model = btom['desire_model'][:, incl, btom['beta_ind'] - 1]
belief_model = btom['belief_model'][:, incl, btom['beta_ind'] - 1]

# Flatten the data for plotting and fitting
x_desire = desire_model.flatten()
y_desire = desire_people.flatten()

x_belief = belief_model.flatten()
y_belief = belief_people.flatten()

savedir = f'output/plots'
os.makedirs(savedir, exist_ok=True)

print('Desire model shape:', desire_model.shape)
print('Desire people shape:', desire_people.shape)

# Plot desire
plt.figure(1)
plt.plot(x_desire, y_desire, 'k.', markersize=markersize)

# Compute linear fit for desire data
coefficients_desire = np.polyfit(x_desire, y_desire, 1)
slope_desire, intercept_desire = coefficients_desire
print(f"Desire linear fit: y = {slope_desire:.2f}x + {intercept_desire:.2f}")

# Create fit line data
x_fit_desire = np.linspace(desire_axis[0], desire_axis[1], 100)
y_fit_desire = np.polyval(coefficients_desire, x_fit_desire)

# Plot the linear fit line with new color
plt.plot(x_fit_desire, y_fit_desire, color='orange', linewidth=2, label=f'Fit: y={slope_desire:.2f}x+{intercept_desire:.2f}')

plt.axis(desire_axis)
plt.xlabel('Desire Model')
plt.ylabel('Desire People')
plt.title('Desire Comparison')
plt.legend()
plt.savefig(f'{savedir}/desire_plot.png', dpi=300)

print('Belief model shape:', belief_model.shape)
print('Belief people shape:', belief_people.shape)

# Plot belief
plt.figure(2)
plt.plot(x_belief, y_belief, 'k.', markersize=markersize)

# Compute linear fit for belief data
coefficients_belief = np.polyfit(x_belief, y_belief, 1)
slope_belief, intercept_belief = coefficients_belief
print(f"Belief linear fit: y = {slope_belief:.2f}x + {intercept_belief:.2f}")

# Create fit line data
x_fit_belief = np.linspace(belief_axis[0], belief_axis[1], 100)
y_fit_belief = np.polyval(coefficients_belief, x_fit_belief)

# Plot the linear fit line with new color
plt.plot(x_fit_belief, y_fit_belief, color='orange', linewidth=2, label=f'Fit: y={slope_belief:.2f}x+{intercept_belief:.2f}')

plt.axis(belief_axis)
plt.xlabel('Belief Model')
plt.ylabel('Belief People')
plt.title('Belief Comparison')
plt.legend()
plt.savefig(f'{savedir}/belief_plot.png', dpi=300)
