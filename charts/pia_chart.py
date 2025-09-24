import matplotlib.pyplot as plt

def create_pie_chart(sizes, labels, colors=None, explode=None, startangle=90):
    """
    Creates a dynamic pie chart similar to the attached image.
    
    Parameters:
    - sizes: list of numbers representing the slice sizes (will be normalized to percentages)
    - labels: list of strings for the legend labels
    - colors: optional list of colors for the slices (e.g., ['gray', 'purple'])
    - explode: optional list of floats to offset slices (e.g., (0.1, 0) to explode the first slice)
    - startangle: optional starting angle for the pie (default 90 for top-start)
    
    Example usage:
    sizes = [14, 86]
    labels = ['Exploitatiekosten', 'Nettohuurinkomsten']
    colors = ['gray', 'purple']
    explode = (0.1, 0)
    create_pie_chart(sizes, labels, colors, explode)
    """
    if colors is None:
        colors = plt.cm.Pastel1(range(len(sizes)))  # Default colors if not provided
    if explode is None:
        explode = (0,) * len(sizes)  # No explosion by default
    
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=None, colors=colors, autopct='%1.0f%%', shadow=False, startangle=startangle)
    ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
    ax.legend(labels, loc='center right', bbox_to_anchor=(1.2, 0.5))
    plt.tight_layout()
    plt.show()

sizes = [14, 86]  # The slice sizes (they'll be converted to percentages)
labels = ['Exploitatiekosten', 'Nettohuurinkomsten']  # Legend labels
colors = ['#F0E6F7', '#C8A8C8']  # Optional: Colors for each slice
explode = (0.1, 0)  # Optional: Offset the first slice slightly

create_pie_chart(sizes, labels, colors=colors, explode=explode)