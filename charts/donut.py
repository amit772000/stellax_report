import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_rental_pie_chart(exploitatie_pct=14, nettohuur_pct=86):
    """
    Create a dynamic pie chart for rental costs
    
    Parameters:
    exploitatie_pct (float): Percentage for Exploitatiekosten
    nettohuur_pct (float): Percentage for Nettohuurinkomsten
    """
    
    # Data
    labels = ['Exploitatiekosten', 'Nettohuurinkomsten']
    sizes = [exploitatie_pct, nettohuur_pct]
    colors = ['#F0E6F7', '#E8C8E8']  # Light gray-pink and light pink
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create the pie chart with exploded slice
    explode = (0.1, 0)  # explode the first slice
    wedges, texts, autotexts = ax.pie(sizes, labels=None, colors=colors, autopct='%1.0f%%', 
                                      explode=explode, startangle=90, counterclock=False,
                                      textprops={'fontsize': 14, 'fontweight': 'bold', 'color': 'black'},
                                      wedgeprops={'edgecolor': 'none'})
    
    # Add custom edge color to the larger wedge
    wedges[1].set_edgecolor('#D8A8D8')  # Darker pink border for larger segment
    wedges[1].set_linewidth(2)
    
    # Calculate positions for percentage labels dynamically
    # For the exploded slice (first slice - exploitatie) - centered in the exploded slice
    # First slice goes from 90° to (90° - exploitatie_pct * 3.6°)
    start_angle1 = 90
    end_angle1 = 90 - exploitatie_pct * 3.6
    mid_angle1 = (start_angle1 + end_angle1) / 2
    angle1 = np.radians(mid_angle1)
    
    # Position it in the center of the exploded slice
    explode_distance = 0.1  # Same as explode value
    x1 = (0.5 + explode_distance) * np.cos(angle1)
    y1 = (0.5 + explode_distance) * np.sin(angle1)
    autotexts[0].set_position((x1, y1))
    
    # For the main slice (second slice - nettohuur) - centered in main circle
    # Second slice goes from (90° - exploitatie_pct * 3.6°) to (90° - 360°)
    start_angle2 = 90 - exploitatie_pct * 3.6
    end_angle2 = 90 - 360
    mid_angle2 = (start_angle2 + end_angle2) / 2
    angle2 = np.radians(mid_angle2)
    
    # Position it in the center of the main slice
    x2 = 0.4 * np.cos(angle2)
    y2 = 0.4 * np.sin(angle2)
    autotexts[1].set_position((x2, y2))
    
    # Create custom legend with squares
    legend_elements = [
        patches.Rectangle((0, 0), 1, 1, facecolor='#F0E6F7', edgecolor='#C8A8C8', linewidth=1),
        patches.Rectangle((0, 0), 1, 1, facecolor='#E8C8E8', edgecolor='#D8A8D8', linewidth=1)
    ]
    
    ax.legend(legend_elements, labels, loc='center left', bbox_to_anchor=(1, 0.5), 
              frameon=False, fontsize=10)
    
    # Set title
    ax.set_title('% van totaal gehanteerde huurstroom', fontsize=12, fontweight='normal', 
                 loc='left', pad=20)
    
    # Remove axes
    ax.set_aspect('equal')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Example usage with default values (14%, 86%)
create_rental_pie_chart()

# Example with different percentages
# create_rental_pie_chart(20, 80)  # Uncomment to test with different values
# create_rental_pie_chart(30, 70)  # Uncomment to test with different values