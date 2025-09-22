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
    

    # Format x-axis with asking prices
    # def currency_formatter_euro(x, p):
    #     if x >= 1000000:
    #         return f'€{x/1000000:.1f}M'
    #     elif x >= 1000:
    #         return f'€{x/1000:.0f}K'
    #     else:
    #         return f'€{x:.0f}'
        
    def currency_formatter_euro(x, p):
        return f'€{x:,.0f}'
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(currency_formatter_euro))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter_euro))

    # Format x-axis with asking prices
   
    
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
    
    # # Add light gray grid
    # ax.grid(True, color='#E0E0E0', linestyle='-', linewidth=0.5, alpha=0.7)
    # ax.set_axisbelow(True)        
    
    # Adjust layout
    plt.tight_layout()

    # Save transparent PNG
    chart_path = "assets/charts/chart_page_6.png"
    plt.savefig(chart_path, format="png", transparent=True)
    plt.close(fig)


# ----------------------------------------------------------------------------------------
# Usage
# ----------------------------------------------------------------------------------------
filler = MultiPageTemplateFiller("Report_template.pdf")

# ---------- Page 1 (example, fill with your vars) ----------
def draw_page0(c, w, h):
    c.setFont("Medium", 14)
    c.setFillColorRGB(255, 255, 255)  
    c.drawString(87, 190, f"{variables.address}")  
    c.drawString(87, 135, f"{variables.report_date}")  
    c.drawString(87, 80, f"{variables.reference_date}")  

filler.add_overlay(0, draw_page0)


# ---------- Page 2 ----------
def draw_page1(c, w, h):   
    c.setFillColorRGB(16/255, 16/255, 16/255)
    # c.drawString(70, 715, f"This report is for the property located at {variables.address}. "
    #                       f"The values in this report are estimated as of {variables.reference_date}. "
    #                       f"The report is available via the link: {variables.report_link}")
    draw_wrapped_text(c, f"This report is for the property located at {variables.address}. "
                          f"The values in this report are estimated as of {variables.reference_date}. "
                          f"The report is available via the link.", 70, 715, "Regular", 10, max_width=450, link=variables.report_link )

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

    c.drawString(225, 206, f"{variables.lot_size}")
    c.drawString(376, 206, f"{variables.lot_size_source}")

    c.drawString(225, 189, f"{variables.energy_label}")
    c.drawString(376, 189, f"{variables.energy_label_source}")

    c.drawString(225, 172, f"{variables.contract_rent}/ma")
    c.drawString(376, 172, f"{variables.contract_rent_source}")

    c.drawString(225, 155, f"{variables.wws_points}")
    c.drawString(376, 155, f"{variables.wws_points_source}")

    c.drawString(225, 138, f"{variables.wws_points_rent}/ma")
    c.drawString(376, 138, f"{variables.wws_rent_source}")

    c.drawString(225, 121, f"{variables.vve}/ma")
    c.drawString(376, 121, f"{variables.vve_source}")

    c.drawString(225, 104, f"{variables.erfpact_date} (bought off)")
    c.drawString(376, 104, f"{variables.erfpact_date_source}")

    c.drawString(225, 87, f"{variables.erfpacht_amount}/j")
    c.drawString(376, 87, f"{variables.erfpacht_amount_source}")

    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

filler.add_overlay(1, draw_page1)

# ---------- Page 3 ----------
def draw_page2(c, w, h):
    c.setFont("Bold", 19)
    c.setFillColorRGB(0, 0, 0)  
    c.drawString(70, 690, f"{variables.address_short}")     
        
    draw_wrapped_text(c, f"{variables.property_overview_1}", 300, 660, "Regular", 10, max_width=270)
    draw_wrapped_text(c, f"{variables.property_overview_2}", 300, 580, "Regular", 10, max_width=270)

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

filler.add_overlay(2, draw_page2)

# ---------- Page 4 ----------
def draw_page3(c, w, h):
    c.setFont("Bold", 30)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(85, 675, f"€{format_number(variables.vacant_value)}")
    c.drawString(325, 675, f"{variables.vacant_value_score}%")

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

filler.add_overlay(3, draw_page3)

# ---------- Page 5 ----------
def draw_page4(c, w, h):
    c.setFont("Regular", 10)
    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_page5_table(c)

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

filler.add_overlay(4, draw_page4)


# ---------- Page 6 ----------
def draw_page5(c, w, h):
    c.setFont("Bold", 30)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(85, 675, f"€{format_number(variables.vacant_value_optimal_asking)}")
    c.drawString(325, 675, f"€{format_number(variables.vacant_value_final_price_paid)}")      

    create_dynamic_price_chart(variables.asking_vs_price_paid_df, variables.vacant_value_final_price_paid)

    c.setFont("Bold", 10)
    c.setFillColorRGB(255, 255, 255)
    c.drawString(150, 167, f"The asking price sweet spot is between €{format_number(variables.vacant_value_optimal_asking_low)} and €{format_number(variables.vacant_value_optimal_asking_high)}.")  

    start_x=150
    start_y=120
    col_gap=50 

    df = variables.asking_vs_price_paid_df

    # Fixed Y positions (two rows)
    y_price_paid = start_y
    y_asking_price = start_y - 30  # adjust spacing between rows    

    # Values row by row (horizontal)
    c.setFont("Regular", 10)
    c.setFillColorRGB(60/255, 60/255, 60/255)

    for i, row in df.iterrows():
        x = start_x + i * col_gap

        # Price paid row
        c.drawString(x, y_price_paid, f"€{format_number(row['Price_paid'])}")

        # Asking price row
        c.drawString(x, y_asking_price, f"€{format_number(row['Asking_price'])}")   

    c.setStrokeColorRGB(1, 1, 1)  # white color
    c.setLineWidth(1)             # thickness
    c.line(start_x - 110, (y_price_paid + y_asking_price) / 2 + 5,
           start_x - 110 + 500, (y_price_paid + y_asking_price) / 2 + 5)

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

filler.add_overlay(5, draw_page5)


# ---------- Save ----------
filler.save("Template_filled.pdf")

# Example: add image on first page
rounded_img = "assets/pictures/property_google_photo_rounded.png"
make_rounded_image("assets/pictures/property_google_photo.png", rounded_img, radius=15, size=(240, 240))

insert_image("Template_filled.pdf", rounded_img, fitz.Rect(70, 170, 280, 360), page_num=2)

cadastral_map_img = "assets/pictures/cadastral_map.png"
insert_image("Template_filled.pdf", cadastral_map_img, fitz.Rect(70, 460, 530, 800), page_num=2)

market_rent_range_chart()
vacant_value_chart_img = "assets/charts/vacant_value_chart.png"
insert_image("Template_filled.pdf", vacant_value_chart_img, fitz.Rect(70, 200, 530, 650), page_num=3)

chart2()
chart2_img = "assets/charts/chart2.png"
insert_image("Template_filled.pdf", chart2_img, fitz.Rect(70, 450, 530, 900), page_num=3)

rounded_img = "assets/pictures/vacant_values_comps_df_rounded.png"
make_rounded_image("assets/pictures/vacant_values_comps_df.png", rounded_img, radius=15, size=(800, 400))

insert_image("Template_filled.pdf", rounded_img, fitz.Rect(60, 120, 540, 350), page_num=4)

page6_chart_img = "assets/charts/chart_page_6.png"
insert_image("Template_filled.pdf", page6_chart_img, fitz.Rect(70, 210, 530, 900), page_num=5)

print("✅ Multi-page PDF generated: Template_filled.pdf")
