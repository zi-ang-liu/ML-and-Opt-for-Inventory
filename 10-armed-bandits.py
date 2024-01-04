import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set the figure size
plt.figure(figsize=(10, 6))

# Generate the data and plot the violin plot
data = np.random.randn(2000, 10) + np.random.randn(10)
plt.violinplot(dataset=data)

# Set the labels and title with increased font sizes
plt.xlabel("Bandit", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.title("Violin plot for 10-armed bandits", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=12)
plt.xticks(np.arange(0, 11))

# Save the figure
plt.savefig('bandits_violin_plot.pdf', format='pdf')