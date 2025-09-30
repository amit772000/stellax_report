import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import pandas as pd

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

# def create_line_chart_page_7(df, x_col='Bid_offered', y_col='Chance_of_winning_pct', 
#                       x_label=None, y_label=None, color='purple', marker='o', 
#                       figsize=(5, 3), save_path="assets/charts/line_chart.png",
#                       label_fontsize=8, tick_fontsize=4):
#     """
#     Creates a dynamic line chart similar to the attached image.
    
#     Parameters:
#     - df: pandas DataFrame containing the data
#     - x_col: column name for x-axis values (default 'Bid_offered')
#     - y_col: column name for y-axis values (default 'Chance_of_winning_pct')
#     - x_label: optional label for x-axis (defaults to formatted x_col)
#     - y_label: optional label for y-axis (defaults to formatted y_col with '%')
#     - color: line and marker color (default 'purple')
#     - marker: marker style (default 'o' for circles)
#     - figsize: tuple for figure size (default (5, 3) for a smaller chart)
#     - save_path: path to save the PNG (default "assets/charts/line_chart.png")
#     - label_fontsize: font size for axis labels (default 8 for smaller)
#     - tick_fontsize: font size for axis ticks (default 6 for smaller, e.g., for 0%, 20% on y-axis and €450,000 on x-axis)
    
#     Example usage with your data:
#     data = {
#         "Bid_offered": [450000, 475000, 500000, 525000, 550000, 575000, 600000, 625000],
#         "Chance_of_winning_pct": [5, 13, 28, 47, 66, 81, 91, 96]
#     }
#     bid_vs_winning_chance_df = pd.DataFrame(data)
#     create_line_chart(bid_vs_winning_chance_df)
#     """
#     if x_label is None:
#         x_label = x_col.replace('_', ' ').title()
#     if y_label is None:
#         y_label = y_col.replace('_', ' ').title() + ' (%1)'
    
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Plot the line with markers
#     ax.plot(df[x_col], df[y_col], color=color, marker=marker, linestyle='-', linewidth=2)
    
#     # Set y-axis as percentage
#     ax.set_ylabel(y_label, fontsize=label_fontsize)
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))
#     ax.set_ylim(0, 100)
    
#     # Set x-axis with euro formatting
#     ax.set_xlabel(x_label, fontsize=label_fontsize)
#     ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'€{x:,.0f}'))
    
#     # Set tick font sizes
#     ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
#     # Add grid lines
#     ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.5)
    
#     # Remove top and right spines
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     plt.tight_layout()
#     plt.show()
    
#     # # Save as transparent PNG
#     # plt.savefig(save_path, format="png", transparent=True, dpi=300)
#     # plt.close(fig)

# data = {
#     "Bid_offered": [450000, 475000, 500000, 525000, 550000, 575000, 600000, 625000],
#     "Chance_of_winning_pct": [5, 13, 28, 47, 66, 81, 91, 96]
# }

# bid_vs_winning_chance_df = pd.DataFrame(data)
# create_line_chart_page_7(bid_vs_winning_chance_df)