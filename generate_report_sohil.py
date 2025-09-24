from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import stringWidth
from PyPDF2 import PdfWriter, PdfReader, Transformation
from reportlab.pdfgen.canvas import Canvas
import fitz  # PyMuPDF
import io
from PIL import Image, ImageDraw
import variables
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import numpy as np
import statsmodels.api as sm
import pandas as pd
from matplotlib.ticker import MultipleLocator

# Register fonts
pdfmetrics.registerFont(TTFont('Bold', 'assets/fonts/Inter-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Light', 'assets/fonts/Inter-Light.ttf'))
pdfmetrics.registerFont(TTFont('Regular', 'assets/fonts/Inter.ttf'))
pdfmetrics.registerFont(TTFont('Medium', 'assets/fonts/Inter-Medium.ttf'))


class MultiPageTemplateFiller:
    def __init__(self, template):
        self.reader = PdfReader(open(template, "rb"))
        self.writer = PdfWriter()
        self.num_pages = len(self.reader.pages)

    def add_overlay(self, page_num, draw_fn):
        """
        draw_fn(canvas, width, height) = function that draws all text/images for this page
        """
        page = self.reader.pages[page_num]
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)

        # Create overlay
        packet = io.BytesIO()
        c = Canvas(packet, pagesize=(width, height))
        draw_fn(c, width, height)
        c.save()

        # Merge overlay with template page
        packet.seek(0)
        overlay_pdf = PdfReader(packet)
        overlay_page = overlay_pdf.pages[0]

        op = Transformation().rotate(0).translate(tx=0, ty=0)
        overlay_page.add_transformation(op)

        page.merge_page(overlay_page)
        self.writer.add_page(page)

    def save(self, dest):
        with open(dest, "wb") as f:
            self.writer.write(f)

from reportlab.pdfbase.pdfmetrics import stringWidth

def draw_wrapped_text(c, text, x, y, font_name="Regular", font_size=10, max_width=400, leading=None, link=None):
    """
    Draw wrapped text starting at (x,y).
    If `link` is provided, the WHOLE text becomes clickable.
    If you want only the URL part clickable, detect it and pass that substring + coords.
    """
    if leading is None:
        leading = font_size + 2  # line spacing

    c.setFont(font_name, font_size)

    words = text.split()
    line = ""
    for word in words:
        test_line = (line + " " + word).strip()
        if stringWidth(test_line, font_name, font_size) <= max_width:
            line = test_line
        else:
            # draw line
            c.drawString(x, y, line)
            if link:
                text_width = stringWidth(line, font_name, font_size)
                c.linkURL(link, (x, y, x + text_width, y + font_size + 2), relative=0)
            y -= leading
            line = word
    if line:
        c.drawString(x, y, line)
        if link:
            text_width = stringWidth(line, font_name, font_size)
            c.linkURL(link, (x, y, x + text_width, y + font_size + 2), relative=0)
        y -= leading

    return y


def draw_right_aligned(c, text, right_x, y, font_name="Regular", font_size=10):
    c.setFont(font_name, font_size)
    text_width = stringWidth(text, font_name, font_size)
    c.drawString(right_x - text_width, y, text)

def make_rounded_image(input_path, output_path, radius=30, size=(280, 280)):
    """Create a square image with rounded corners"""
    im = Image.open(input_path).convert("RGBA")
    im = im.resize(size, Image.LANCZOS)

    # Create same size mask with rounded rectangle
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, size[0], size[1]], radius=radius, fill=255)

    # Apply mask
    im.putalpha(mask)
    im.save(output_path, format="PNG")

def market_rent_range_chart(save_path=None):
    fig, ax1 = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('white')

    # Chart 1: Vacant value range
    ax1.set_facecolor('white')

    # Data for the horizontal bars
    categories = ['Upper bound', 'Expected value', 'Lower bound']
    values = [variables.vacant_value_high, variables.vacant_value, variables.vacant_value_low]
    colors = ['#c794fb', '#c794fb', '#c794fb']  # More vibrant purple shades

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
    ax1.set_yticklabels(categories, fontsize=14, color='#4a5568')
    ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    tick_labels = ['€ -'] + [f'€ {val:,}' for val in range(100000, 600001, 100000)]
    ax1.set_xticklabels(tick_labels)       

    # Add value labels at the end of bars
    value_labels = [f"€{val:,} " for val in [variables.vacant_value_high, variables.vacant_value, variables.vacant_value_low]]
    for i, (val, label) in enumerate(zip(values, value_labels)):
        bar_width = val / 600000 * 6
        ax1.text(6.5, y_positions[i], label, 
                va='center', ha='left', fontsize=16, color='#4a5568', fontweight='normal')

    # Add vertical grid lines
    for x in [1, 2, 3, 4, 5]:
        ax1.axvline(x, color='#e2e8f0', linestyle='-', linewidth=1, alpha=0.8)

    # Remove spines
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.tick_params(axis='y', length=0, labelsize=16)
    ax1.tick_params(axis='x', length=0, labelsize=14, colors='#718096')
    plt.tight_layout()
    
    # Save transparent PNG
    if save_path:
        plt.savefig(save_path, format="png", transparent=True, dpi=150, bbox_inches="tight")
    else:
        chart_path = "assets/charts/vacant_value_chart.png"
        plt.savefig(chart_path, format="png", transparent=True)

    plt.close(fig)


def chart2():
    fig, ax2 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax2.set_facecolor('white')

    # ✅ Use real data from variables.py
    df = variables.vacant_values_per_sqm_comps_df

    # Scatter plot (hollow purple circles)
    ax2.scatter(
        df["Sqm"], df["Price_m2"],
        c='none', edgecolors='#9333ea',
        alpha=0.7, s=80, linewidth=2
    )

    # ✅ Convert values into €/m² for highlighting
    sqm_ref = variables.sqm
    low_per_sqm = variables.vacant_value_low / sqm_ref
    expected_per_sqm = variables.vacant_value / sqm_ref
    high_per_sqm = variables.vacant_value_high / sqm_ref

    # ✅ Plot highlight points
    ax2.scatter([sqm_ref], [low_per_sqm], color="#9333ea", s=120, zorder=5)
    ax2.scatter([sqm_ref], [expected_per_sqm], color="#9333ea", s=120, zorder=5)
    ax2.scatter([sqm_ref], [high_per_sqm], color="#9333ea", s=120, zorder=5)

    # ✅ Add arrows + labels
    ax2.annotate("Lower bound",
                 xy=(sqm_ref, low_per_sqm),
                 xytext=(sqm_ref + 10, low_per_sqm - 100),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=12, ha="left", va="center")

    ax2.annotate("Expected value",
                 xy=(sqm_ref, expected_per_sqm),
                 xytext=(sqm_ref + 10, expected_per_sqm),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=12, ha="left", va="center")

    ax2.annotate("Upper bound",
                 xy=(sqm_ref, high_per_sqm),
                 xytext=(sqm_ref + 10, high_per_sqm + 100),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=12, ha="left", va="center")

    # Axis limits and ticks
    ax2.set_xlim(35, 150)
    ax2.set_ylim(4250, 8250)

    ax2.set_xticks([40, 60, 80, 100, 120, 140])
    ax2.set_xticklabels(
        ['40 m²', '60 m²', '80 m²', '100 m²', '120 m²', '140 m²'],
        fontsize=12, color='#718096'
    )
    ax2.set_yticks([4250, 4750, 5250, 5750, 6250, 6750, 7250, 7750, 8250])
    ax2.set_yticklabels(
        ['€ 4,250', '€ 4,750', '€ 5,250', '€ 5,750', '€ 6,250',
         '€ 6,750', '€ 7,250', '€ 7,750', '€ 8,250'],
        fontsize=14, color='#718096'
    )

    # ✅ Add spacing between labels and axes
    ax2.xaxis.labelpad = 10
    ax2.yaxis.labelpad = 10

    # Style
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color('#a1a5ab')
    ax2.spines['left'].set_color("#a1a5ab")
    ax2.tick_params(colors='#718096', length=0)

    plt.tight_layout()

    # Save transparent PNG
    chart_path = "assets/charts/chart2.png"
    plt.savefig(chart_path, format="png", transparent=True)
    plt.close(fig)


def insert_image(pdf_path, img_path, rect, page_num=0):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    page.insert_image(rect, filename=img_path, keep_proportion=True)
    doc.save(pdf_path, incremental=True, encryption=0)
    doc.close()


def format_number(n):
    """Format numbers like 531000 -> 531K, 1250000 -> 1M, etc."""
    if n is None:
        return "0"
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n // 1_000_000_000}B"
    elif n >= 1_000_000:
        return f"{n // 1_000_000}M"
    elif n >= 10_000:
        return f"{n // 1_000}K"
    else:
        return f"{n:,}"


def draw_page5_table(c, start_x=70, start_y=468, row_height=1):
    """
    Version with debug lines to help you fine-tune positions.
    This will show exactly where each value should be placed.
    """
    df = variables.vacant_values_comps_df

    # Column positions - adjust these numbers to align perfectly
    columns = {
        'type': start_x,
        'asking_price': start_x + 115,      
        'bid_above': start_x + 233,         
        'adjusted_price': start_x + 348,    
        'square_meters': start_x,     
        'lot_size': start_x + 115,          
        'year': start_x + 233,              
        'date': start_x + 348,              
    }   

    # Reset to black for text
    c.setStrokeColorRGB(0, 0, 0)

    for i, row in df.iterrows():
        y = start_y - i * row_height

        # Property header
        c.setFont("Bold", 11)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(
            80, y - i * 76,
            f" {row['Address']} ({row['Distance_meters']} meters from target)"
        )

        # Table values
        c.setFont("Regular", 9)
        c.setFillColorRGB(60/255, 60/255, 60/255)
        
        y = start_y - i * 77

        # Row 1
        c.drawString(columns['type'], y - 30, f"{row['Type']}")
        c.drawString(columns['asking_price'], y - 30, f"€ {row['Asking_price']:,}")
        c.drawString(columns['bid_above'], y - 30, f"{row['Bid_above_asking_pct']}")
        c.drawString(columns['adjusted_price'], y - 30, f"€ {row['Adjusted_price']:,}")

        # # Row 2
        c.drawString(columns['square_meters'], y - 58, f"{row['Square_meters']}")
        c.drawString(columns['lot_size'], y - 58, f"{row['Lot_size']}")
        c.drawString(columns['year'], y - 58, f"{row['Year']}")
        c.drawString(columns['date'], y - 58, f"{row['Date']}")

def create_pie_chart(sizes, labels, colors=None, explode=None, startangle=90, pct_fontsize=14, label_fontsize=12, wedge_border=True):
    """
    Creates a dynamic pie chart similar to the attached image.
    
    Parameters:
    - sizes: list of numbers representing the slice sizes (will be normalized to percentages)
    - labels: list of strings for the legend labels
    - colors: optional list of colors for the slices (e.g., ['gray', 'purple'])
    - explode: optional list of floats to offset slices (e.g., (0.1, 0) to explode the first slice)
    - startangle: optional starting angle for the pie (default 90 for top-start)
    - pct_fontsize: optional font size for the percentage labels (default 14)
    - label_fontsize: optional font size for the legend labels (default 12)
    - wedge_border: optional boolean to add borders to pie wedges (default True)
    
    Example usage:
    sizes = [14, 86]
    labels = ['Exploitatiekosten', 'Nettohuurinkomsten']
    colors = ['gray', 'purple']
    explode = (0.1, 0)
    create_pie_chart(sizes, labels, colors, explode, pct_fontsize=20, label_fontsize=14, wedge_border=True)
    """
    if colors is None:
        colors = plt.cm.Pastel1(range(len(sizes)))  # Default colors if not provided
    if explode is None:
        explode = (0,) * len(sizes)  # No explosion by default
    
    wedgeprops = {'edgecolor': 'black', 'linewidth': 1} if wedge_border else None
    
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=None, colors=colors, autopct='%1.0f%%', 
           shadow=False, startangle=startangle, textprops={'fontsize': pct_fontsize},
           wedgeprops=wedgeprops)
    ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
    ax.legend(labels, loc='center right', bbox_to_anchor=(1.5, 0.5), fontsize=label_fontsize)
    plt.tight_layout()
    # Save transparent PNG
    chart_path = "assets/charts/chart_page_15.png"
    plt.savefig(chart_path, format="png", transparent=True)
    plt.close(fig)
    
def create_line_chart_page_7(df, x_col='Bid_offered', y_col='Chance_of_winning_pct', 
                      x_label=None, y_label=None, color='#A855F7', marker='o', 
                      figsize=(5, 3), save_path="assets/charts/line_chart_page_7.png",
                      label_fontsize=8, tick_fontsize=6):
    """
    Creates a dynamic line chart similar to the attached image.
    
    Parameters:
    - df: pandas DataFrame containing the data
    - x_col: column name for x-axis values (default 'Bid_offered')
    - y_col: column name for y-axis values (default 'Chance_of_winning_pct')
    - x_label: optional label for x-axis (defaults to formatted x_col)
    - y_label: optional label for y-axis (defaults to formatted y_col with '%')
    - color: line and marker color (default 'purple')
    - marker: marker style (default 'o' for circles)
    - figsize: tuple for figure size (default (5, 3) for a smaller chart)
    - save_path: path to save the PNG (default "assets/charts/line_chart_page_7.png")
    - label_fontsize: font size for axis labels (default 8 for smaller)
    - tick_fontsize: font size for axis ticks (default 6 for smaller, e.g., for 0%, 20% on y-axis and €450,000 on x-axis)
    
    Example usage with your data:
    data = {
        "Bid_offered": [450000, 475000, 500000, 525000, 550000, 575000, 600000, 625000],
        "Chance_of_winning_pct": [5, 13, 28, 47, 66, 81, 91, 96]
    }
    bid_vs_winning_chance_df = pd.DataFrame(data)
    create_line_chart(bid_vs_winning_chance_df)
    """
    if x_label is None:
        x_label = x_col.replace('_', ' ').title()
    if y_label is None:
        y_label = y_col.replace('_', ' ').title() + ' (%)'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the line with markers
    ax.plot(df[x_col], df[y_col], color=color, marker=marker, linestyle='-', linewidth=1)
    
    # Set y-axis as percentage
    # ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))
    ax.set_ylim(0, 100)
    
    # Set x-axis with euro formatting
    # ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'€{x:,.0f}'))
    
    # Set tick font sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Add grid lines
    ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.5)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save as transparent PNG
    plt.savefig(save_path, format="png", transparent=True, dpi=300)
    plt.close(fig)

def pie_chart_percentage_page_8(value, color, text_color="black", save_path=None):
    """
    Create a circular percentage chart (donut style).
    
    Parameters:
        value (float): Decimal value (e.g., 0.89 for 89%)
        color (str): Hex color for the filled part
        text_color (str): Color of the text inside
    """
    percentage = round(value * 100)

    # Data for chart
    sizes = [percentage, 100 - percentage]
    colors = [color, "#f8f6fc"]  # active color, background color

    # Create pie chart
    fig, ax = plt.subplots(figsize=(4,4))
    wedges, _ = ax.pie(
        sizes, 
        colors=colors, 
        startangle=90, 
        counterclock=False, 
        wedgeprops=dict(width=0.08)
    )
    

    # Add text in the middle
    plt.text(
        0.05, 0, f"{percentage}%", 
        ha="center", va="center", 
        fontsize=45, fontweight="bold", color=text_color
    )

    # Equal aspect ratio ensures circle
    ax.set(aspect="equal")
    # plt.show()
    # Save transparent PNG
    if save_path:
        plt.savefig(save_path, format="png", transparent=True, dpi=150, bbox_inches="tight")
    else:
        plt.savefig("assets/charts/pie_chart_page_8.png", format="png", transparent=True, dpi=150, bbox_inches="tight")
    # chart_path = "assets/charts/pie_chart_page_8.png"
    # plt.savefig(chart_path, format="png", transparent=True)
    plt.close(fig)

def create_line_chart_page_8(df, x_col="Date", y_col="Price_index", color="#c9a0ff", save_path=None):
    """
    Create a line chart with markers directly from a DataFrame.
    
    Parameters:
        df (pd.DataFrame): Data containing x and y columns
        x_col (str): Column name for x-axis (default: "Date")
        y_col (str): Column name for y-axis (default: "Price_index")
        color (str): Line and marker color
        save_path (str): If provided, saves the chart as an image (PNG)
    """
    plt.figure(figsize=(8, 4))
    plt.plot(df[x_col], df[y_col], color=color, marker="o", linewidth=2, markersize=6)

    # Grid lines
    plt.grid(True, linestyle="--", alpha=0.6)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")

    # Add some margins
    plt.margins(x=0.05, y=0.1)

    # Style
    ax = plt.gca()  # get current axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # ✅ Dynamic step size based on data range
    y_min, y_max = df[y_col].min(), df[y_col].max()
    y_range = y_max - y_min
    step = max(0.5, round(y_range / 10, 1))  # ~10 ticks, at least 0.5 apart
    ax.yaxis.set_major_locator(MultipleLocator(step))

    # # Add margins
    # plt.margins(x=0.05, y=0.1)

    # Change axis label colors
    ax.tick_params(axis="x", colors="#718096")   # X-axis labels in red
    ax.tick_params(axis="y", colors="#718096")  # Y-axis labels in blue

    # Grid: only horizontal lines
    ax.yaxis.grid(True, linestyle="-", alpha=0.6)
    ax.xaxis.grid(False)   # disable vertical lines

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", transparent=True, dpi=150, bbox_inches="tight")
    else:
        plt.savefig("assets/charts/line_chart_page_8.png", format="png", transparent=True, dpi=150, bbox_inches="tight")
    # chart_path = "assets/charts/line_chart_page_8.png"
    # plt.savefig(chart_path, dpi=150, bbox_inches="tight", format="png", transparent=True)

def create_market_rent_range_chart_page_9(values, categories=None, max_value=None, save_path=None):
    """
    Create a dynamic market rent range chart with rounded bars.
    
    Parameters:
        values (list of float): Numeric values (e.g., [high, expected, low])
        categories (list of str): Labels for each bar (default: ['Upper bound', 'Expected value', 'Lower bound'])
        max_value (float): Maximum scale for the x-axis (default: max(values))
        save_path (str): Path to save PNG. If None, saves to 'assets/charts/vacant_value_chart.png'
    """
    if categories is None:
        categories = ['Upper bound', 'Expected value', 'Lower bound']
    if max_value is None:
        max_value = max(values)

    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    # Dynamic scaling (map value → bar length out of 6)
    def scale(val):
        return val / max_value * 6

    # Bar appearance
    bar_height = 0.4
    # y_positions = list(range(len(values)-1, -1, -1))  # Top-down order
    y_positions = [2, 1, 0]
    colors = ['#c794fb'] * len(values)  # uniform purple (can be customized)
    colors = ['#c794fb', '#c794fb', '#c794fb']  # More vibrant purple shades

    # Draw bars
    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        bar_width = scale(val)
        rect_width = bar_width - bar_height/2
        rect = Rectangle((0, y_positions[i] - bar_height/2), rect_width, bar_height, 
                         facecolor=color, edgecolor='none')
        ax1.add_patch(rect)
        circle = patches.Circle((rect_width, y_positions[i]), bar_height/2, 
                                facecolor=color, edgecolor='none')
        ax1.add_patch(circle)

    # Axis formatting
    ax1.set_xlim(-0.2, 6.5)
    # ax1.set_ylim(-0.5, len(values)-0.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(categories, fontsize=14, color='#4a5568')

    # Dynamic x-ticks (steps of max_value / 6)
    # step = max_value // 6
    # ax1.set_xticks(range(7))
    # tick_labels = ['€ -'] + [f'€ {val:,.0f}' for val in range(int(step), int(max_value)+1, int(step))]
    # ax1.set_xticklabels(tick_labels)

    ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    tick_labels = ['€ -'] + [f'€ {val:,}' for val in range(500, 3500, 500)]
    ax1.set_xticklabels(tick_labels)   

    # Value labels at the end
    for i, val in enumerate(values):
        ax1.text(6.5, y_positions[i], f"€{val:,.0f}", 
                 va='center', ha='left', fontsize=16, color='#4a5568')

    # Grid lines
    for x in range(1, 6):
        ax1.axvline(x, color='#e2e8f0', linestyle='-', linewidth=1, alpha=0.8)

    # Remove spines & style
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.tick_params(axis='y', length=0)
    ax1.tick_params(axis='x', length=0, labelsize=14, colors='#718096')

    plt.tight_layout()

    # Save PNG
    if save_path:
        plt.savefig(save_path, format="png", transparent=True, dpi=150, bbox_inches="tight")
    else:
        plt.savefig("assets/charts/vacant_value_chart.png", format="png", transparent=True, dpi=150, bbox_inches="tight")

    plt.close(fig)

def plot_rent_chart_page_9(df, sqm_ref, low_per_sqm, expected_per_sqm, high_per_sqm):
    """
    Plots a scatter chart of rent per m² vs. square meters, with upper bound, expected value,
    and lower bound indicators at the specified sqm_ref using the provided per_sqm values,
    similar to the attached image.
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(df['Sqm'], df['Rent_m2'], c='none', edgecolors='#9333ea', alpha=0.7, s=80, linewidth=2)
    
    # ax.set_xlabel('m²')
    # ax.set_ylabel('€')
    ax.set_xlim(35, 150)
    ax.set_ylim(15, 50)

    ax.set_xticks([40, 60, 80, 100, 120, 140])
    ax.set_xticklabels(['40 m²', '60 m²', '80 m²', '100 m²', '120 m²', '140 m²'],fontsize=12, color='#718096')

    ax.set_yticks([15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_yticklabels(['€ 15', '€ 20', '€ 25', '€ 30', '€ 35', '€ 40', '€ 45', '€ 50'], fontsize=12, color='#718096')
    
    # Use provided values for bounds and expected
    lower = low_per_sqm
    expected = expected_per_sqm
    upper = high_per_sqm
    
    # Draw vertical bar
    ax.vlines(sqm_ref, lower, upper, color='black', linewidth=2, zorder=5)
    
    # Add filled circles for bounds and expected
    ax.plot(sqm_ref, upper, 'o', color='#9333ea', markersize=8, zorder=5)
    ax.plot(sqm_ref, expected, 'o', color='#9333ea', markersize=8, zorder=5)
    ax.plot(sqm_ref, lower, 'o', color='#9333ea', markersize=8, zorder=5)
    
    # Add annotations with arrows
    label_offset = 5  # Adjust as needed for positioning
    ax.annotate('Upper bound', xy=(sqm_ref, upper), xytext=(sqm_ref + label_offset, upper),
                arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1),
                va='center', ha='left', fontsize=13)
    ax.annotate('Expected value', xy=(sqm_ref, expected), xytext=(sqm_ref + label_offset, expected),
                arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1),
                va='center', ha='left', fontsize=13)
    ax.annotate('Lower bound', xy=(sqm_ref, lower), xytext=(sqm_ref + label_offset, lower),
                arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1),
                va='center', ha='left', fontsize=13)
    
    # ✅ Add spacing between labels and axes
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10

    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#a1a5ab')
    ax.spines['left'].set_color("#a1a5ab")
    ax.tick_params(colors='#718096', length=0)
    
    plt.tight_layout()
    # plt.show()

    # Save transparent PNG
    chart_path = "assets/charts/plot_rent_chart_page_9.png"
    plt.savefig(chart_path, format="png", transparent=True)
    plt.close(fig)


def format_percentage(value: float) -> str:
    # Multiply by 100 and format with 1 decimal place
    formatted = f"{value * 100:.1f}"
    # Replace dot with comma
    formatted = formatted.replace(".", ",")
    return f"{formatted}%"

def format_without_percent_number(val):
    return f"{val}".replace(".", ",")

def format_with_percent_number(val):
    return f"{val}".replace(".", ",") + "%"

def format_simple_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"

def to_percentage(value):
    """
    Convert a decimal number to a percentage string without unnecessary decimals.
    
    Parameters:
        value (float): Decimal value (e.g., 0.71)
    
    Returns:
        str: Percentage string (e.g., "71%")
    """
    percentage = value * 100
    # Convert to int if it's a whole number
    if percentage.is_integer():
        percentage = int(percentage)
    return f"{percentage}%"

def format_euro(value: float, mode: str = "auto") -> str:
    """
    Format numbers into Euro currency styles.
    
    mode:
      - "int"    → €504   (no decimals, no separator)
      - "thousands" → €27.000   (no decimals, with thousands separator)
      - "decimals"  → €28,48    (with two decimals, comma as decimal separator)
      - "auto"      → decides based on value type
    """
    
    if mode == "int":
        return f"€{int(value)}"
    
    elif mode == "thousands":
        formatted = f"{value:,.0f}".replace(",", ".")
        return f"€{formatted}"
    
    elif mode == "decimals":
        integer, decimal = f"{value:,.2f}".split(".")
        integer = integer.replace(",", ".")
        return f"€{integer},{decimal}"
    
    elif mode == "auto":
        if value == int(value):  # exact integer
            if value < 1000:
                return f"€{int(value)}"
            else:
                formatted = f"{value:,.0f}".replace(",", ".")
                return f"€{formatted}"
        else:
            integer, decimal = f"{value:,.2f}".split(".")
            integer = integer.replace(",", ".")
            return f"€{integer},{decimal}"
        
# ----------------------------------------------------------------------------------------
# Usage
# ----------------------------------------------------------------------------------------
filler = MultiPageTemplateFiller("Report_template.pdf")

# ---------- Page 07 ----------
def draw_page7(c, w, h):
    c.setFont("Bold", 30)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(87, 674, f"{to_percentage(variables.percent_sold_above_asking)}")
    c.drawString(325, 674, f"{to_percentage(variables.average_bidding)}")

    start_x=150
    start_y=115
    col_gap=49 

    df = variables.bid_vs_winning_chance_df

    # Fixed Y positions (two rows)
    y_price_paid = start_y
    y_asking_price = start_y - 29  # adjust spacing between rows    

    # Values row by row (horizontal)
    c.setFont("Regular", 10)
    c.setFillColorRGB(60/255, 60/255, 60/255)

    for i, row in df.iterrows():
        x = start_x + i * col_gap

        # Price paid row
        c.drawString(x, y_price_paid, f"€{format_number(row['Bid_offered'])}")

        # Asking price row
        c.drawString(x + 9, y_asking_price, f"{row['Chance_of_winning_pct']}%")   

    c.setStrokeColorRGB(1, 1, 1)  # white color
    c.setLineWidth(1)             # thickness
    c.line(start_x - 110, (y_price_paid + y_asking_price) / 2 + 5,
           start_x - 110 + 500, (y_price_paid + y_asking_price) / 2 + 5)

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

# filler.add_overlay(6, draw_page7)

# ---------- Page 08 ----------
def draw_page8(c, w, h):
    c.setFont("Bold", 15)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(300, 688, f"{variables.vacant_value_market_descitption}")

    c.setFont("Bold", 18)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(68, 300, f"Price development {variables.city}")
    c.setFont("Regular", 18)
    c.drawString(345, 300, f"(Index, {variables.vacant_value_index_reference_date})")

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

# filler.add_overlay(7, draw_page8)

# ---------- Page 09 ----------
def draw_page9(c, w, h):
    c.setFont("Bold", 30)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(87, 674, f"{format_euro(variables.market_rent)}")
    c.drawString(325, 674, f"{to_percentage(variables.market_rent_score)}")

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

# filler.add_overlay(8, draw_page9)

# ---------- Page 11 ----------
def draw_page11(c, w, h):
    c.setFont("Bold", 15)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(300, 680, f"{variables.market_rent_market_descitption}")

    c.setFont("Bold", 18)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(68, 300, f"Price development {variables.city}")
    c.setFont("Regular", 18)
    c.drawString(345, 300, f"(Index, {variables.market_rent_index_reference_date})")

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

filler.add_overlay(10, draw_page11)

# ---------- Page 13 ----------
def return_on_equity_table_page_13(c):
    data = variables.return_on_equity_df

    # Starting positions (adjust as needed to match your blank template)
    start_x = 140  # Left margin
    start_y = 315  # Top margin, subtract to go down

    # Draw headers
    c.setFont("Bold", 10)
    y = start_y - 13
    x = start_x + 60
    for col in data.columns[1:]:
        c.drawString(x, y, format_without_percent_number(col))
        x += 70

    # Draw LTV and data using for loop
    y -= 17
    for i in range(len(data)):
        row = data.iloc[i]
        # Draw LTV
        c.setFont("Bold", 10)
        c.drawString(start_x, y, row['LTV'])
        # Draw values
        c.setFont("Light", 10)
        x = start_x + 60
        for val in row[1:]:
            c.drawString(x, y, f"{format_with_percent_number(val)}")
            x += 70
        y -= 17

def monthly_cash_flow_table_page_13(c):
    data = variables.monthly_cash_flow_df

    # Starting positions (adjust as needed to match your blank template)
    start_x = 140  # Left margin
    start_y = 168  # Top margin, subtract to go down

    # Draw headers
    c.setFont("Bold", 10)
    y = start_y - 13
    x = start_x + 60
    for col in data.columns[1:]:
        c.drawString(x, y, format_without_percent_number(col))
        x += 70

    # Draw LTV and data using for loop
    y -= 17
    for i in range(len(data)):
        row = data.iloc[i]
        # Draw LTV
        c.setFont("Bold", 10)
        c.drawString(start_x, y, row['LTV'])
        # Draw values
        c.setFont("Light", 10)
        x = start_x + 60
        for val in row[1:]:
            c.drawString(x, y, f"€{format_number(val)}")
            x += 70
        y -= 17

def draw_page13(c, w, h):
    c.setFont("Bold", 30)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(90, 684, f"{format_percentage(variables.gross_yield)}")
    c.drawString(210, 684, f"{format_percentage(variables.net_yield)}")
    c.drawString(325, 684, f"{format_percentage(variables.return_on_equity)}")
    c.drawString(440, 684, f"€{format_number(variables.cashflow)}")    

    return_on_equity_table_page_13(c)
    monthly_cash_flow_table_page_13(c)

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

# filler.add_overlay(12, draw_page13)

# ---------- Page 14 ----------
def format_percentage_two_decimal(value: float) -> str:
    formatted = f"{value * 100:.2f}"      # two decimals
    formatted = formatted.replace(".", ",")  # dot → comma
    return f"{formatted}%"

def draw_page14(c, w, h):
    c.setFont("Bold", 30)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(97, 675, f"€{format_number(variables.rented_value)}")

    c.setFont("Bold", 10)
    c.setFillColorRGB(255, 255, 255)
    c.drawString(240, 512, f"{format_percentage_two_decimal(variables.bar_kk)}")
    c.drawString(240, 485, f"{format_percentage_two_decimal(variables.nar_kk)}")
    c.drawString(240, 458, f"{variables.capitalisation_factor}")
    c.drawString(440, 512, f"€{variables.vacant_value}")
    c.drawString(440, 485, f"{variables.vacant_value_ratio}%")
    c.drawString(440, 458, f"{format_simple_percentage(variables.rent_vacant_value_ratio)}")

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

# filler.add_overlay(13, draw_page14)

# ---------- Page 15 ----------
def draw_page15(c, w, h):
    c.setFont("Light", 10)
    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

    c.drawString(95, 528, f"{variables.sqm}")
    c.drawString(160, 528, f"{format_euro(variables.effective_rent_yearly)}")
    c.drawString(275, 528, f"{format_euro(variables.effective_rent_per_sqm)}")
    c.drawString(370, 528, f"{format_euro(variables.vacant_value)}")
    c.drawString(465, 528, f"{format_euro(variables.vacant_value_per_sqm)}")

    c.drawString(80, 447, f"{format_euro(variables.market_rent_yearly)}")
    c.drawString(163, 447, f"{format_euro(variables.contract_rent_yearly)}")
    c.drawString(265, 447, f"{format_euro(variables.wws_rent_yearly)}")
    c.drawString(373, 447, f"{format_euro(variables.effective_rent_yearly)}")
    c.drawString(454, 447, f"{variables.effective_rent_method}")

    c.drawString(240, 375, f"{format_euro(variables.municipality_taxes)}")
    c.drawString(240, 358, f"{format_euro(variables.management_costs, mode="int")}")
    c.drawString(240, 341, f"{format_euro(variables.maintenance_costs, mode="int")}")
    c.drawString(240, 324, f"{format_euro(variables.VVE_yearly, mode="int")}")
    c.drawString(240, 307, f"{format_euro(variables.erfpacht_amount, mode="int")}")
    c.drawString(240, 290, f"{format_euro(variables.other_running_costs, mode="int")}")
    c.setFont("Bold", 10)
    c.drawString(240, 275, f"{format_euro(variables.total_running_costs, mode="int")}")

    c.setFont("Light", 10)
    c.drawString(235, 196.5, f"{format_euro(variables.effective_rent_yearly)}")
    c.drawString(235, 176, f"{format_euro(variables.total_running_costs)}")
    c.drawString(235, 155, f"{format_euro(variables.net_rental_income)}")
    c.drawString(235, 136, f"{format_simple_percentage(variables.nar_von)}")
    c.setFont("Bold", 10)
    c.drawString(235, 116, f"{format_euro(variables.rented_value_von)}")

    c.setFont("Light", 10)
    c.drawString(470, 196, f"{format_euro(variables.rented_value_von)}")
    c.drawString(470, 177, f"{format_euro(variables.legal_and_delivery_costs)}")
    c.drawString(470, 155, f"{format_euro(variables.transfer_tax)}")
    c.drawString(470, 136, f"{format_euro(variables.other_costs_corrections)}")
    c.setFont("Bold", 10)
    c.drawString(470, 117, f"{format_euro(variables.rented_value)}")

# filler.add_overlay(14, draw_page15)


# ---------- Save ----------
filler.save("Template_filled.pdf")

# # Example: add image on first page
# rounded_img = "assets/pictures/property_google_photo_rounded.png"
# make_rounded_image("assets/pictures/property_google_photo.png", rounded_img, radius=15, size=(240, 240))

# insert_image("Template_filled.pdf", rounded_img, fitz.Rect(70, 170, 280, 360), page_num=2)

# cadastral_map_img = "assets/pictures/cadastral_map.png"
# insert_image("Template_filled.pdf", cadastral_map_img, fitz.Rect(70, 460, 530, 800), page_num=2)

# market_rent_range_chart()
# vacant_value_chart_img = "assets/charts/vacant_value_chart.png"
# insert_image("Template_filled.pdf", vacant_value_chart_img, fitz.Rect(70, 200, 530, 650), page_num=3)

# chart2()
# chart2_img = "assets/charts/chart2.png"
# insert_image("Template_filled.pdf", chart2_img, fitz.Rect(70, 450, 530, 900), page_num=3)

# rounded_img = "assets/pictures/vacant_values_comps_df_rounded.png"
# make_rounded_image("assets/pictures/vacant_values_comps_df.png", rounded_img, radius=15, size=(800, 400))

# insert_image("Template_filled.pdf", rounded_img, fitz.Rect(60, 120, 540, 350), page_num=4)

# page6_chart_img = "assets/charts/chart_page_6.png"
# insert_image("Template_filled.pdf", page6_chart_img, fitz.Rect(70, 210, 530, 900), page_num=5)



# create_line_chart_page_7(variables.bid_vs_winning_chance_df)
# page7_chart_img = "assets/charts/line_chart_page_7.png"
# insert_image("Template_filled.pdf", page7_chart_img, fitz.Rect(0, 450, 550, 710), page_num=0)
# # insert_image("Template_filled.pdf", page7_chart_img, fitz.Rect(70, 280, 530, 880), page_num=0)

# pie_chart_percentage_page_8(variables.vacant_value_demand_score, color="#c9a0ff")
# page8_chart_img = "assets/charts/pie_chart_page_8.png"
# insert_image("Template_filled.pdf", page8_chart_img, fitz.Rect(90, 150, 250, 300), page_num=1)

# create_line_chart_page_8(variables.vacant_value_index_df)
# page8_chart_img = "assets/charts/line_chart_page_8.png"
# insert_image("Template_filled.pdf", page8_chart_img, fitz.Rect(63, 480, 520, 860), page_num=1)

# market_rent_score = variables.market_rent_score
# market_rent_low = variables.market_rent_low
# market_rent_high = variables.market_rent_high

# # Calculate expected based on score
# market_rent_expected = market_rent_low + (market_rent_high - market_rent_low) * market_rent_score

# values = [market_rent_high, market_rent_expected, market_rent_low]
# categories = ['Upper bound', 'Expected value', 'Lower bound']

# # Call chart function
# create_market_rent_range_chart_page_9(values, categories, max_value=market_rent_high, save_path="assets/charts/vacant_value_chart_page_9.png")
# vacant_value_chart_img = "assets/charts/vacant_value_chart_page_9.png"
# insert_image("Template_filled.pdf", vacant_value_chart_img, fitz.Rect(70, 200, 530, 650), page_num=0)

# sqm_ref = variables.sqm
# low_per_sqm = variables.market_rent_low / sqm_ref
# expected_per_sqm = variables.market_rent / sqm_ref
# high_per_sqm = variables.market_rent_high / sqm_ref

# plot_rent_chart_page_9(variables.market_rent_per_sqm_comps_df, sqm_ref, low_per_sqm, expected_per_sqm, high_per_sqm)
# # plot_rent_chart_page_9()
# plot_rent_chart_page_9_img = "assets/charts/plot_rent_chart_page_9.png"
# insert_image("Template_filled.pdf", plot_rent_chart_page_9_img, fitz.Rect(70, 450, 530, 900), page_num=0)

page11_chart_img = "assets/charts/pie_chart_page_11.png"
pie_chart_percentage_page_8(variables.market_rent_demand_score, color="#D3D3D3", save_path=page11_chart_img)
insert_image("Template_filled.pdf", page11_chart_img, fitz.Rect(100, 155, 240, 295), page_num=0)

page11_line_chart_img = "assets/charts/line_chart_page_11.png"
create_line_chart_page_8(variables.market_rent_index_df, x_col="Date", y_col="Rent_index", save_path=page11_line_chart_img)
insert_image("Template_filled.pdf", page11_line_chart_img, fitz.Rect(63, 480, 520, 860), page_num=0)

# sizes = [variables.running_costs_to_effective_rent_percentage, variables.net_rental_income_to_effective_rent_percentage]  # The slice sizes (they'll be converted to percentages)
# labels = ['Exploitatiekosten', 'Nettohuurinkomsten']  # Legend labels
# colors = ['#F0E6F7', '#C8A8C8']  # Optional: Colors for each slice
# explode = (0.1, 0)  # Optional: Offset the first slice slightly

# create_pie_chart(sizes, labels, colors=colors, explode=explode)
# page15_chart_img = "assets/charts/chart_page_15.png"
# insert_image("Template_filled.pdf", page15_chart_img, fitz.Rect(280, 360, 580, 690), page_num=0)

print("✅ Multi-page PDF generated: Template_filled.pdf")
