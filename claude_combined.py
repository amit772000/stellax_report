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


# Utility Functions
def draw_wrapped_text(c, text, x, y, font_name="Regular", font_size=10, max_width=400, leading=None, link=None):
    """
    Draw wrapped text starting at (x,y).
    If `link` is provided, the WHOLE text becomes clickable.
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


def format_short(n):
    """Format deltas and rents like 53000 -> 53K, -127000 -> -127K, 0 -> 0, 7000 -> 7K, 6000 -> 6K"""
    if n is None:
        return "0"
    n = int(n)
    sign = "-" if n < 0 else ""
    a = abs(n)
    if a >= 1_000_000:
        return f"{sign}{a//1_000_000}M"
    elif a >= 1_000:
        return f"{sign}{a//1_000}K"
    else:
        return f"{sign}{a:,}"


# Chart Generation Functions
def market_rent_range_chart():
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
    chart_path = "assets/charts/vacant_value_chart.png"
    plt.savefig(chart_path, format="png", transparent=True)
    plt.close(fig)


def chart2():
    fig, ax2 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax2.set_facecolor('white')

    # Use real data from variables.py
    df = variables.vacant_values_per_sqm_comps_df

    # Scatter plot (hollow purple circles)
    ax2.scatter(
        df["Sqm"], df["Price_m2"],
        c='none', edgecolors='#9333ea',
        alpha=0.7, s=80, linewidth=2
    )

    # Convert values into €/m² for highlighting
    sqm_ref = variables.sqm
    low_per_sqm = variables.vacant_value_low / sqm_ref
    expected_per_sqm = variables.vacant_value / sqm_ref
    high_per_sqm = variables.vacant_value_high / sqm_ref

    # Plot highlight points
    ax2.scatter([sqm_ref], [low_per_sqm], color="#9333ea", s=120, zorder=5)
    ax2.scatter([sqm_ref], [expected_per_sqm], color="#9333ea", s=120, zorder=5)
    ax2.scatter([sqm_ref], [high_per_sqm], color="#9333ea", s=120, zorder=5)

    # Add arrows + labels
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

    # Add spacing between labels and axes
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


def create_dynamic_price_chart(data_df, vacant_value_final_price_paid, price_column='Price_paid'):
    """
    Create a dynamic single line price chart matching the provided image
    
    Parameters:
    data_df (DataFrame): DataFrame with price data
    price_column (str): Column name to plot (default: 'Price_paid')    
    """
    
    # Create figure and axis with exact proportions from image
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Generate x-axis positions (asking prices for x-axis)
    asking_prices = data_df['Asking_price'].values
    price_values = data_df[price_column].values
    
    # Plot single line with purple color and circular markers
    ax.plot(asking_prices, price_values, color='#A855F7', linewidth=2.5, 
            marker='o', markersize=6, markerfacecolor='#A855F7', 
            markeredgecolor='#A855F7', alpha=0.9)
    
    # Add value annotation for the peak (€535000)
    max_idx = np.argmax(price_values)
    max_value = vacant_value_final_price_paid
    max_asking = asking_prices[max_idx]
    ax.annotate(f'€{max_value:,.0f}', 
                xy=(max_asking, max_value), 
                xytext=(0, 20), textcoords='offset points',
                fontsize=14, color='black',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8e8e8', edgecolor='none', alpha=1.0),
                arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0', 
                              color='#e8e8e8', lw=1, alpha=1.0))
    
    def currency_formatter_euro(x, p):
        return f'€{x:,.0f}'
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(currency_formatter_euro))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter_euro))
    
    # Set axis ranges similar to the image
    x_min = min(asking_prices) - 20000
    x_max = max(asking_prices) + 20000
    y_min = 400000  # Fixed from image
    y_max = 560000  # Fixed from image
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Create ticks similar to image
    x_ticks = np.arange(400000, 600000, 25000)
    y_ticks = np.arange(400000, 580000, 20000)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # Style the ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#a1a5ab')
    ax.spines['left'].set_color("#a1a5ab")    
    ax.tick_params(axis='both', labelsize=14, colors='#718096')
    
    # Adjust layout
    plt.tight_layout()

    # Save transparent PNG
    chart_path = "assets/charts/chart_page_6.png"
    plt.savefig(chart_path, format="png", transparent=True)
    plt.close(fig)


# Table Drawing Functions
def draw_page5_table(c, start_x=70, start_y=468, row_height=1):
    """
    Draw table for vacant values comparables
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

        # Row 2
        c.drawString(columns['square_meters'], y - 58, f"{row['Square_meters']}")
        c.drawString(columns['lot_size'], y - 58, f"{row['Lot_size']}")
        c.drawString(columns['year'], y - 58, f"{row['Year']}")
        c.drawString(columns['date'], y - 58, f"{row['Date']}")


def draw_page10_table(c, start_x=70, start_y=468, row_height=1):
    """
    Draw table for market rent comparables
    """
    df = variables.market_rent_comps_df

    # Column positions - adjust these numbers to align perfectly
    columns = {
        'type': start_x,
        'asking_price': start_x + 115,      
        'bid_above': start_x + 230,         
        'adjusted_price': start_x + 343,    
        'square_meters': start_x,     
        'lot_size': start_x + 115,          
        'year': start_x + 230,              
        'date': start_x + 343,              
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

        # Row 2
        c.drawString(columns['square_meters'], y - 58, f"{row['Square_meters']}")
        c.drawString(columns['lot_size'], y - 58, f"{row['Lot_size']}")
        c.drawString(columns['year'], y - 58, f"{row['Year']}")
        c.drawString(columns['date'], y - 58, f"{row['Date']}")


# Energy Label Functions
def _get_impact_df():
    # support both a DataFrame or the raw dict in variables.py
    try:
        return variables.energy_label_impact_df
    except Exception:
        import pandas as pd
        return pd.DataFrame(variables.data)


def _box_color_for(n):
    if n > 0:
        return (0.78, 0.94, 0.84)   # green
    if n == 0:
        return (0.96, 0.94, 0.80)   # yellow
    return (0.96, 0.80, 0.82)       # red


def draw_energy_label_impact_table(c,
                                   top_y=566,
                                   row_h=None,
                                   col_vv=176,
                                   col_dvv=268,
                                   col_mv=396,
                                   col_dmv=488,
                                   box_w=100,  # wider colored boxes
                                   box_h=16,
                                   font_name="Regular",
                                   font_size=10):
    df = _get_impact_df()

    # Ensure stable Label ordering
    try:
        labels_order = variables.data.get("Label", None)
    except Exception:
        labels_order = None
    if labels_order and "Label" in df.columns:
        df = df.set_index("Label").loc[labels_order].reset_index()

    # Font metrics for vertical centering
    c.setFont(font_name, font_size)
    ascent = pdfmetrics.getAscent(font_name)
    descent = pdfmetrics.getDescent(font_name)
    ascent_pt = ascent * font_size / 1000.0
    descent_pt = descent * font_size / 1000.0
    if descent_pt > 0:
        descent_pt = -descent_pt
    glyph_half = (ascent_pt + descent_pt) / 2.0
    gray = (60/255, 60/255, 60/255)

    # Table 1: Use explicit Y positions for all columns
    row_ys = [553 - i*17 for i in range(10)]

    # Define column centers for centering text/boxes
    col_vv_center = col_vv
    col_dvv_center = col_dvv + box_w / 2
    col_mv_center = col_mv + 40   # adjust 40 for your template
    col_dmv_center = col_dmv + box_w / 2

    for i, row in df.iterrows():
        y_baseline = row_ys[i]
        rect_y = y_baseline - box_h / 2.0
        box_center = rect_y + box_h / 2.0
        text_baseline = box_center - glyph_half

        # Draw colored boxes centered under their columns
        dv = int(row["ΔVV"])
        dv_box_x = col_dvv_center - box_w / 2
        c.setFillColorRGB(*_box_color_for(dv))
        c.rect(dv_box_x, rect_y, box_w, box_h, stroke=0, fill=1)
        dm = int(row["ΔMV"])
        dm_box_x = col_dmv_center - box_w / 2
        c.setFillColorRGB(*_box_color_for(dm))
        c.rect(dm_box_x, rect_y, box_w, box_h, stroke=0, fill=1)

        # VV (centered)
        c.setFillColorRGB(*gray)
        c.drawCentredString(col_vv_center, text_baseline, f"€{format_number(row['VV'])}")

        # ΔVV (centered in box)
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(col_dvv_center, text_baseline, format_short(int(row["ΔVV"])))

        # MV (centered)
        c.setFillColorRGB(*gray)
        c.drawCentredString(col_mv_center, text_baseline, f"€{format_number(row['MV'])}")

        # ΔMV (centered in box)
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(col_dmv_center, text_baseline, format_short(int(row["ΔMV"])))

    # Table 2: Use explicit Y positions for all columns
    row_ys_lower = [348 - i*17 for i in range(10)]

    for i, row in df.iterrows():
        y_baseline = row_ys_lower[i]
        rect_y = y_baseline - box_h / 2.0
        box_center = rect_y + box_h / 2.0
        text_baseline = box_center - glyph_half

        # Draw colored boxes centered under their columns
        dw = int(row["ΔWWS"])
        dw_box_x = col_dvv_center - box_w / 2
        c.setFillColorRGB(*_box_color_for(dw))
        c.rect(dw_box_x, rect_y, box_w, box_h, stroke=0, fill=1)
        dr = int(row["ΔRent"])
        dr_box_x = col_dmv_center - box_w / 2
        c.setFillColorRGB(*_box_color_for(dr))
        c.rect(dr_box_x, rect_y, box_w, box_h, stroke=0, fill=1)

        # WWS (centered)
        c.setFillColorRGB(*gray)
        c.drawCentredString(col_vv_center, text_baseline, f"{int(row['WWS'])}")

        # ΔWWS (centered in box)
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(col_dvv_center, text_baseline, str(int(row["ΔWWS"])))

        # Rent (centered)
        c.setFillColorRGB(*gray)
        c.drawCentredString(col_mv_center, text_baseline, f"€{format_number(row['Rent'])}")

        # ΔRent (centered in box)
        dr_text = f"€{format_short(dr)}" if dr != 0 else "€0"
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(col_dmv_center, text_baseline, dr_text)


# Page Drawing Functions
def draw_page0(c, w, h):
    """Page 1 - Cover Page"""
    c.setFont("Medium", 14)
    c.setFillColorRGB(255, 255, 255)  
    c.drawString(87, 190, f"{variables.address}")  
    c.drawString(87, 135, f"{variables.report_date}")  
    c.drawString(87, 80, f"{variables.reference_date}")  


def draw_page1(c, w, h):
    """Page 2 - Summary Page"""
    c.setFillColorRGB(16/255, 16/255, 16/255)
    draw_wrapped_text(c, f"This report is for the property located at {variables.address}. "
                          f"The values in this report are estimated as of {variables.reference_date}. "
                          f"The report is available via the link.", 70, 715, "Regular", 10, max_width=450, link=variables.report_link)

    c.setFont("Bold", 30)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(82, 645, f"€{format_number(variables.vacant_value)}")
    c.drawString(325, 645, f"€{format_number(variables.rented_value)}")
    c.drawString(82, 510, f"€{format_number(variables.market_rent)}")
    c.drawString(325, 510, f"{variables.wws_points}")

    # Data and sources
    c.setFont("Regular", 10)
    c.setFillColorRGB(130/255, 130/255, 130/255)
    c.drawString(225, 257, f"{variables.property_type}")
    c.drawString(376, 257, f"{variables.property_type_source}")

    c.drawString(225, 240, f"{variables.sqm}")
    c.drawString(376, 240, f"{variables.sqm_source}")

    c.drawString(225, 223, f"{variables.year}")
    c.drawString(376, 223, f"{variables.year_source}")

    c.drawString(225