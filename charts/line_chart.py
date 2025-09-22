import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def create_price_development_chart(dates, values, title="Price development Amsterdam", 
                                 subtitle="(Index, 05/24=100)", y_min=None, y_max=None):
    """
    Create a dynamic price development chart
    
    Parameters:
    dates (list): List of date strings in format 'MMM-YY' (e.g., ['May-24', 'Jun-24'])
    values (list): List of corresponding values
    title (str): Main title of the chart
    subtitle (str): Subtitle with index information
    y_min (float): Minimum y-axis value (auto if None)
    y_max (float): Maximum y-axis value (auto if None)
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot the line with purple color and circular markers
    ax.plot(range(len(dates)), values, color='#A855F7', linewidth=2.5, 
            marker='o', markersize=8, markerfacecolor='#A855F7', 
            markeredgecolor='#A855F7', alpha=0.9)
    
    # Set x-axis
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=0, fontsize=12, color='#666666')
    
    # Set y-axis range
    if y_min is None:
        y_min = min(values) - 1
    if y_max is None:
        y_max = max(values) + 1
    
    ax.set_ylim(y_min, y_max)
    
    # Create y-axis ticks (every 1 unit from y_min to y_max)
    y_ticks = np.arange(int(y_min), int(y_max) + 1, 1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(tick)) for tick in y_ticks], fontsize=12, color='#666666')
    
    # Add horizontal grid lines only (no vertical lines)
    ax.grid(True, axis='y', color='#C0C0C0', linestyle='-', linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    
    # Set title
    full_title = f"{title} {subtitle}"
    ax.set_title(full_title, fontsize=18, fontweight='bold', color='black', pad=20)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')
    
    # Set margins
    ax.margins(x=0.02)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Example data matching the screenshot
dates_example = ['May-24', 'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 
                'Nov-24', 'Dec-24', 'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25']

values_example = [100.0, 100.7, 101.1, 101.7, 102.2, 102.8, 103.2, 103.3, 103.2, 103.3, 103.7, 104.4, 105.2]

# Create the chart with default data
create_price_development_chart(dates_example, values_example, y_min=97, y_max=106)

# Example with different data
# dates_custom = ['Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24']
# values_custom = [95, 97, 98, 99, 100, 102]
# create_price_development_chart(dates_custom, values_custom, 
#                               title="Custom Price Index", subtitle="(Index, Jan/24=95)")