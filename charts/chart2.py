import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Rectangle

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.patch.set_facecolor('white')

# Chart 1: Vacant value range
ax1.set_facecolor('white')

# Data for the horizontal bars
categories = ['Upper bound', 'Expected value', 'Lower bound']
values = [590000, 531000, 480000]
colors = ['#c084fc', '#a855f7', '#9333ea']  # More vibrant purple shades

# Create horizontal bars with smaller rounded ends
bar_height = 0.4
y_positions = [2, 1, 0]

for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
    # Calculate bar width as proportion of max value
    bar_width = val / 600000 * 6
    
    # Create main rectangle (body of the bar) - reduced width to accommodate rounded end
    rect_width = bar_width - bar_height/2  # Proper rounded end
    rect = Rectangle((0, y_positions[i] - bar_height/2), rect_width, bar_height, 
                    facecolor=color, edgecolor='none')
    ax1.add_patch(rect)
    
    # Add proper rounded end (semicircle at the right end)
    circle = patches.Circle((rect_width, y_positions[i]), bar_height/2,  # Proper radius
                           facecolor=color, edgecolor='none')
    ax1.add_patch(circle)

# Customize the first chart
ax1.set_xlim(-0.2, 6.2)
ax1.set_ylim(-0.5, 2.5)
ax1.set_yticks(y_positions)
ax1.set_yticklabels(categories, fontsize=11, color='#4a5568')
ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax1.set_xticklabels(['€ -', '€ 100,000', '€ 200,000', '€ 300,000', '€ 400,000', '€ 500,000', '€ 600,000'], 
                   fontsize=9, color='#718096')

# Add value labels at the end of bars
value_labels = ['€ 590,000', '€ 531,000', '€ 480,000']
for i, (val, label) in enumerate(zip(values, value_labels)):
    bar_width = val / 600000 * 6
    ax1.text(6.5, y_positions[i], label, 
            va='center', ha='left', fontsize=11, color='#4a5568', fontweight='normal')

ax1.set_title('Vacant value range', fontsize=18, fontweight='bold', color='#1a202c', pad=25)
ax1.text(0, -1.0, 'The graph below shows the 95% confidence interval for the vacant value estimate, including the lower and\nupper bounds and the expected sale price.',
         fontsize=11, color='#718096', ha='left')

# Add vertical grid lines
for x in [1, 2, 3, 4, 5]:
    ax1.axvline(x, color='#e2e8f0', linestyle='-', linewidth=1, alpha=0.8)

# Remove spines
for spine in ax1.spines.values():
    spine.set_visible(False)
ax1.tick_params(axis='y', length=0)
ax1.tick_params(axis='x', length=0, colors='#718096')

# Chart 2: Price per square meter analysis
ax2.set_facecolor('white')

# Create realistic scatter data that matches the pattern
np.random.seed(123)

# Generate main scatter points
x_coords = []
y_coords = []

# Cluster around 45-50 m²
cluster1_x = np.random.normal(47, 2, 8)
cluster1_y = np.random.uniform(5200, 5400, 8)
x_coords.extend(cluster1_x)
y_coords.extend(cluster1_y)

# Cluster around 55-65 m²  
cluster2_x = np.random.uniform(55, 65, 12)
cluster2_y = np.random.uniform(5400, 6800, 12)
x_coords.extend(cluster2_x)
y_coords.extend(cluster2_y)

# Cluster around 75-85 m²
cluster3_x = np.random.uniform(75, 85, 10)
cluster3_y = np.random.uniform(5000, 6500, 10)
x_coords.extend(cluster3_x)
y_coords.extend(cluster3_y)

# Cluster around 95-110 m²
cluster4_x = np.random.uniform(95, 110, 15)
cluster4_y = np.random.uniform(4800, 6400, 15)
x_coords.extend(cluster4_x)
y_coords.extend(cluster4_y)

# Cluster around 130-145 m²
cluster5_x = np.random.uniform(130, 145, 8)
cluster5_y = np.random.uniform(4600, 6200, 8)
x_coords.extend(cluster5_x)
y_coords.extend(cluster5_y)

# Create scatter plot with hollow circles - NO TREND LINES
ax2.scatter(x_coords, y_coords, c='none', edgecolors='#9333ea', alpha=0.7, s=40, linewidth=1.2)

# Add simple legend positioned in the center-right area like the example
legend_lines = [
    plt.Line2D([0], [0], color='#4a5568', linewidth=1.5, label='Upper bound'),
    plt.Line2D([0], [0], color='#9333ea', linewidth=2, label='Expected value'),
    plt.Line2D([0], [0], color='#4a5568', linewidth=1.5, label='Lower bound')
]

# Position legend in center-right area
legend = ax2.legend(handles=legend_lines, loc='center right', bbox_to_anchor=(0.95, 0.65), 
                   frameon=True, fontsize=10, labelcolor='#4a5568')

# Style the legend frame
legend.get_frame().set_facecolor('#f8f9fa')
legend.get_frame().set_edgecolor('#e2e8f0')
legend.get_frame().set_linewidth(1)

# Customize the second chart
ax2.set_xlim(35, 150)
ax2.set_ylim(4250, 8250)

# Set ticks
ax2.set_xticks([40, 60, 80, 100, 120, 140])
ax2.set_xticklabels(['40 m²', '60 m²', '80 m²', '100 m²', '120 m²', '140 m²'], 
                   fontsize=10, color='#718096')
ax2.set_yticks([4250, 4750, 5250, 5750, 6250, 6750, 7250, 7750, 8250])
ax2.set_yticklabels(['€ 4,250', '€ 4,750', '€ 5,250', '€ 5,750', '€ 6,250', 
                    '€ 6,750', '€ 7,250', '€ 7,750', '€ 8,250'], 
                   fontsize=10, color='#718096')

ax2.set_title('Price per square meter analysis', fontsize=18, fontweight='bold', color='#1a202c', pad=25)
ax2.text(40, 3800, 'The graph below shows the price per square meter for properties of varying sizes in the area, helping to\ncontextualize the subject property\'s pricing.',
         fontsize=11, color='#718096', ha='left')

# Style the grid and spines
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='#e2e8f0')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color('#e2e8f0')
ax2.spines['left'].set_color('#e2e8f0')
ax2.tick_params(colors='#718096', length=0)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)

# Show the plot
plt.show()