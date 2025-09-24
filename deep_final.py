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


def draw_wrapped_text(c, text, x, y, font_name="Regular", font_size=10, max_width=400, leading=None, link=None):
    """
    Draw wrapped text starting at (x,y).
    If `link` is provided, the WHOLE text becomes clickable.
    """
    if leading is None:
        leading = font_size + 2

    c.setFont(font_name, font_size)
    words = text.split()
    line = ""
    for word in words:
        test_line = (line + " " + word).strip()
        if stringWidth(test_line, font_name, font_size) <= max_width:
            line = test_line
        else:
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

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, size[0], size[1]], radius=radius, fill=255)

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
    """Format deltas and rents like 53000 -> 53K, -127000 -> -127K, 0 -> 0"""
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


def to_percentage(value):
    """Convert decimal to percentage string"""
    percentage = value * 100
    if percentage.is_integer():
        percentage = int(percentage)
    return f"{percentage}%"


def format_percentage(value):
    """Format percentage with comma decimal separator"""
    formatted = f"{value * 100:.1f}".replace(".", ",")
    return f"{formatted}%"


def format_percentage_two_decimal(value):
    """Format percentage with two decimals and comma"""
    formatted = f"{value * 100:.2f}".replace(".", ",")
    return f"{formatted}%"


def format_simple_percentage(value):
    return f"{value * 100:.2f}%"


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
 

def format_without_percent_number(val):
    return f"{val}".replace(".", ",")


def format_with_percent_number(val):
    return f"{val}".replace(".", ",") + "%"


# Chart functions
def market_rent_range_chart():
    fig, ax1 = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    categories = ['Upper bound', 'Expected value', 'Lower bound']
    values = [variables.vacant_value_high, variables.vacant_value, variables.vacant_value_low]
    colors = ['#c794fb', '#c794fb', '#c794fb']

    bar_height = 0.4
    y_positions = [2, 1, 0]

    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        bar_width = val / 600000 * 6
        rect_width = bar_width - bar_height/2
        rect = Rectangle((0, y_positions[i] - bar_height/2), rect_width, bar_height, 
                        facecolor=color, edgecolor='none')
        ax1.add_patch(rect)
        
        circle = patches.Circle((rect_width, y_positions[i]), bar_height/2,
                            facecolor=color, edgecolor='none')
        ax1.add_patch(circle)

    ax1.set_xlim(-0.2, 6.2)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(categories, fontsize=14, color='#4a5568')
    ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    tick_labels = ['€ -'] + [f'€ {val:,}' for val in range(100000, 600001, 100000)]
    ax1.set_xticklabels(tick_labels)

    value_labels = [f"€{val:,} " for val in [variables.vacant_value_high, variables.vacant_value, variables.vacant_value_low]]
    for i, (val, label) in enumerate(zip(values, value_labels)):
        bar_width = val / 600000 * 6
        ax1.text(6.5, y_positions[i], label, 
                va='center', ha='left', fontsize=16, color='#4a5568', fontweight='normal')

    for x in [1, 2, 3, 4, 5]:
        ax1.axvline(x, color='#e2e8f0', linestyle='-', linewidth=1, alpha=0.8)

    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.tick_params(axis='y', length=0, labelsize=16)
    ax1.tick_params(axis='x', length=0, labelsize=14, colors='#718096')
    plt.tight_layout()
    
    chart_path = "assets/charts/vacant_value_chart.png"
    plt.savefig(chart_path, format="png", transparent=True)
    plt.close(fig)


def chart2():
    fig, ax2 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax2.set_facecolor('white')

    df = variables.vacant_values_per_sqm_comps_df

    ax2.scatter(
        df["Sqm"], df["Price_m2"],
        c='none', edgecolors='#9333ea',
        alpha=0.7, s=80, linewidth=2
    )

    sqm_ref = variables.sqm
    low_per_sqm = variables.vacant_value_low / sqm_ref
    expected_per_sqm = variables.vacant_value / sqm_ref
    high_per_sqm = variables.vacant_value_high / sqm_ref

    ax2.scatter([sqm_ref], [low_per_sqm], color="#9333ea", s=120, zorder=5)
    ax2.scatter([sqm_ref], [expected_per_sqm], color="#9333ea", s=120, zorder=5)
    ax2.scatter([sqm_ref], [high_per_sqm], color="#9333ea", s=120, zorder=5)

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

    ax2.xaxis.labelpad = 10
    ax2.yaxis.labelpad = 10

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color('#a1a5ab')
    ax2.spines['left'].set_color("#a1a5ab")
    ax2.tick_params(colors='#718096', length=0)

    plt.tight_layout()
    chart_path = "assets/charts/chart2.png"
    plt.savefig(chart_path, format="png", transparent=True)
    plt.close(fig)


def create_dynamic_price_chart(data_df, vacant_value_final_price_paid, price_column='Price_paid'):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    asking_prices = data_df['Asking_price'].values
    price_values = data_df[price_column].values
    
    ax.plot(asking_prices, price_values, color='#A855F7', linewidth=2.5, 
            marker='o', markersize=6, markerfacecolor='#A855F7', 
            markeredgecolor='#A855F7', alpha=0.9)
    
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
    
    x_min = min(asking_prices) - 20000
    x_max = max(asking_prices) + 20000
    y_min = 400000
    y_max = 560000
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    x_ticks = np.arange(400000, 600000, 25000)
    y_ticks = np.arange(400000, 580000, 20000)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#a1a5ab')
    ax.spines['left'].set_color("#a1a5ab")    
    ax.tick_params(axis='both', labelsize=14, colors='#718096')
    
    plt.tight_layout()
    chart_path = "assets/charts/chart_page_6.png"
    plt.savefig(chart_path, format="png", transparent=True)
    plt.close(fig)


def create_pie_chart(sizes, labels, colors=None, explode=None, startangle=90, pct_fontsize=14, label_fontsize=12, wedge_border=True):
    if colors is None:
        colors = plt.cm.Pastel1(range(len(sizes)))
    if explode is None:
        explode = (0,) * len(sizes)
    
    wedgeprops = {'edgecolor': 'black', 'linewidth': 1} if wedge_border else None
    
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=None, colors=colors, autopct='%1.0f%%', 
           shadow=False, startangle=startangle, textprops={'fontsize': pct_fontsize},
           wedgeprops=wedgeprops)
    ax.axis('equal')
    ax.legend(labels, loc='center right', bbox_to_anchor=(1.5, 0.5), fontsize=label_fontsize)
    plt.tight_layout()
    chart_path = "assets/charts/chart_page_15.png"
    plt.savefig(chart_path, format="png", transparent=True)
    plt.close(fig)


def create_line_chart_page_7(df, x_col='Bid_offered', y_col='Chance_of_winning_pct', 
                      color='#A855F7', marker='o', figsize=(5, 3), save_path="assets/charts/line_chart_page_7.png",
                      label_fontsize=8, tick_fontsize=6):
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[x_col], df[y_col], color=color, marker=marker, linestyle='-', linewidth=1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'€{x:,.0f}'))
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, format="png", transparent=True, dpi=300)
    plt.close(fig)


def pie_chart_percentage_page_8(value, color, text_color="black", save_path=None):
    percentage = round(value * 100)
    sizes = [percentage, 100 - percentage]
    colors = [color, "#f8f6fc"]

    fig, ax = plt.subplots(figsize=(4,4))
    wedges, _ = ax.pie(
        sizes, 
        colors=colors, 
        startangle=90, 
        counterclock=False, 
        wedgeprops=dict(width=0.08)
    )
    
    plt.text(0.05, 0, f"{percentage}%", 
             ha="center", va="center", 
             fontsize=45, fontweight="bold", color=text_color)
    ax.set(aspect="equal")
    
    if save_path:
        plt.savefig(save_path, format="png", transparent=True, dpi=150, bbox_inches="tight")
    else:
        plt.savefig("assets/charts/pie_chart_page_8.png", format="png", transparent=True, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_line_chart_page_8(df, x_col="Date", y_col="Price_index", color="#c9a0ff", save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(df[x_col], df[y_col], color=color, marker="o", linewidth=2, markersize=6)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45, ha="right")
    plt.margins(x=0.05, y=0.1)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    y_min, y_max = df[y_col].min(), df[y_col].max()
    y_range = y_max - y_min
    step = max(0.5, round(y_range / 10, 1))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.tick_params(axis="x", colors="#718096")
    ax.tick_params(axis="y", colors="#718096")
    ax.yaxis.grid(True, linestyle="-", alpha=0.6)
    ax.xaxis.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", transparent=True, dpi=150, bbox_inches="tight")
    else:
        plt.savefig("assets/charts/line_chart_page_8.png", format="png", transparent=True, dpi=150, bbox_inches="tight")
    plt.close()


def create_market_rent_range_chart_page_9(values, categories=None, max_value=None, save_path=None):
    if categories is None:
        categories = ['Upper bound', 'Expected value', 'Lower bound']
    if max_value is None:
        max_value = max(values)

    fig, ax1 = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    def scale(val):
        return val / max_value * 6

    bar_height = 0.4
    y_positions = [2, 1, 0]
    colors = ['#c794fb', '#c794fb', '#c794fb']

    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        bar_width = scale(val)
        rect_width = bar_width - bar_height/2
        rect = Rectangle((0, y_positions[i] - bar_height/2), rect_width, bar_height, 
                         facecolor=color, edgecolor='none')
        ax1.add_patch(rect)
        circle = patches.Circle((rect_width, y_positions[i]), bar_height/2, 
                                facecolor=color, edgecolor='none')
        ax1.add_patch(circle)

    ax1.set_xlim(-0.2, 6.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(categories, fontsize=14, color='#4a5568')
    ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    tick_labels = ['€ -'] + [f'€ {val:,}' for val in range(500, 3500, 500)]
    ax1.set_xticklabels(tick_labels)

    for i, val in enumerate(values):
        ax1.text(6.5, y_positions[i], f"€{val:,.0f}", 
                 va='center', ha='left', fontsize=16, color='#4a5568')

    for x in range(1, 6):
        ax1.axvline(x, color='#e2e8f0', linestyle='-', linewidth=1, alpha=0.8)

    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.tick_params(axis='y', length=0)
    ax1.tick_params(axis='x', length=0, labelsize=14, colors='#718096')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="png", transparent=True, dpi=150, bbox_inches="tight")
    else:
        plt.savefig("assets/charts/vacant_value_chart.png", format="png", transparent=True, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rent_chart_page_9(df, sqm_ref, low_per_sqm, expected_per_sqm, high_per_sqm):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(df['Sqm'], df['Rent_m2'], c='none', edgecolors='#9333ea', alpha=0.7, s=80, linewidth=2)
    ax.set_xlim(35, 150)
    ax.set_ylim(15, 50)

    ax.set_xticks([40, 60, 80, 100, 120, 140])
    ax.set_xticklabels(['40 m²', '60 m²', '80 m²', '100 m²', '120 m²', '140 m²'],fontsize=12, color='#718096')
    ax.set_yticks([15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_yticklabels(['€ 15', '€ 20', '€ 25', '€ 30', '€ 35', '€ 40', '€ 45', '€ 50'], fontsize=12, color='#718096')
    
    lower = low_per_sqm
    expected = expected_per_sqm
    upper = high_per_sqm
    
    ax.vlines(sqm_ref, lower, upper, color='black', linewidth=2, zorder=5)
    ax.plot(sqm_ref, upper, 'o', color='#9333ea', markersize=8, zorder=5)
    ax.plot(sqm_ref, expected, 'o', color='#9333ea', markersize=8, zorder=5)
    ax.plot(sqm_ref, lower, 'o', color='#9333ea', markersize=8, zorder=5)
    
    label_offset = 5
    ax.annotate('Upper bound', xy=(sqm_ref, upper), xytext=(sqm_ref + label_offset, upper),
                arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1),
                va='center', ha='left', fontsize=13)
    ax.annotate('Expected value', xy=(sqm_ref, expected), xytext=(sqm_ref + label_offset, expected),
                arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1),
                va='center', ha='left', fontsize=13)
    ax.annotate('Lower bound', xy=(sqm_ref, lower), xytext=(sqm_ref + label_offset, lower),
                arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1),
                va='center', ha='left', fontsize=13)
    
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#a1a5ab')
    ax.spines['left'].set_color("#a1a5ab")
    ax.tick_params(colors='#718096', length=0)
    plt.tight_layout()
    chart_path = "assets/charts/plot_rent_chart_page_9.png"
    plt.savefig(chart_path, format="png", transparent=True)
    plt.close(fig)


def _get_impact_df():
    try:
        return variables.energy_label_impact_df
    except Exception:
        import pandas as pd
        return pd.DataFrame(variables.data)


def _box_color_for(n):
    if n > 0:
        return (0.78, 0.94, 0.84)
    if n == 0:
        return (0.96, 0.94, 0.80)
    return (0.96, 0.80, 0.82)


def draw_energy_label_impact_table(c, top_y=566, row_h=None, col_vv=176, col_dvv=268, 
                                   col_mv=396, col_dmv=488, box_w=100, box_h=16, 
                                   font_name="Regular", font_size=10):
    df = _get_impact_df()

    try:
        labels_order = variables.data.get("Label", None)
    except Exception:
        labels_order = None
    if labels_order and "Label" in df.columns:
        df = df.set_index("Label").loc[labels_order].reset_index()

    c.setFont(font_name, font_size)
    ascent = pdfmetrics.getAscent(font_name)
    descent = pdfmetrics.getDescent(font_name)
    ascent_pt = ascent * font_size / 1000.0
    descent_pt = descent * font_size / 1000.0
    if descent_pt > 0:
        descent_pt = -descent_pt
    glyph_half = (ascent_pt + descent_pt) / 2.0
    gray = (60/255, 60/255, 60/255)

    row_ys = [553 - i*17 for i in range(10)]
    col_vv_center = col_vv
    col_dvv_center = col_dvv + box_w / 2
    col_mv_center = col_mv + 40
    col_dmv_center = col_dmv + box_w / 2

    for i, row in df.iterrows():
        y_baseline = row_ys[i]
        rect_y = y_baseline - box_h / 2.0
        box_center = rect_y + box_h / 2.0
        text_baseline = box_center - glyph_half

        dv = int(row["ΔVV"])
        dv_box_x = col_dvv_center - box_w / 2
        c.setFillColorRGB(*_box_color_for(dv))
        c.rect(dv_box_x, rect_y, box_w, box_h, stroke=0, fill=1)
        dm = int(row["ΔMV"])
        dm_box_x = col_dmv_center - box_w / 2
        c.setFillColorRGB(*_box_color_for(dm))
        c.rect(dm_box_x, rect_y, box_w, box_h, stroke=0, fill=1)

        c.setFillColorRGB(*gray)
        c.drawCentredString(col_vv_center, text_baseline, f"€{format_number(row['VV'])}")
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(col_dvv_center, text_baseline, format_short(int(row["ΔVV"])))
        c.setFillColorRGB(*gray)
        c.drawCentredString(col_mv_center, text_baseline, f"€{format_number(row['MV'])}")
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(col_dmv_center, text_baseline, format_short(int(row["ΔMV"])))

    row_ys_lower = [348 - i*17 for i in range(10)]

    for i, row in df.iterrows():
        y_baseline = row_ys_lower[i]
        rect_y = y_baseline - box_h / 2.0
        box_center = rect_y + box_h / 2.0
        text_baseline = box_center - glyph_half

        dw = int(row["ΔWWS"])
        dw_box_x = col_dvv_center - box_w / 2
        c.setFillColorRGB(*_box_color_for(dw))
        c.rect(dw_box_x, rect_y, box_w, box_h, stroke=0, fill=1)
        dr = int(row["ΔRent"])
        dr_box_x = col_dmv_center - box_w / 2
        c.setFillColorRGB(*_box_color_for(dr))
        c.rect(dr_box_x, rect_y, box_w, box_h, stroke=0, fill=1)

        c.setFillColorRGB(*gray)
        c.drawCentredString(col_vv_center, text_baseline, f"{int(row['WWS'])}")
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(col_dvv_center, text_baseline, str(int(row["ΔWWS"])))
        c.setFillColorRGB(*gray)
        c.drawCentredString(col_mv_center, text_baseline, f"€{format_number(row['Rent'])}")
        dr_text = f"€{format_short(dr)}" if dr != 0 else "€0"
        c.setFillColorRGB(0, 0, 0)
        c.drawCentredString(col_dmv_center, text_baseline, dr_text)


# Table drawing functions
def draw_page5_table(c, start_x=70, start_y=468, row_height=1):
    df = variables.vacant_values_comps_df
    columns = {
        'type': start_x,
        'asking_price': start_x + 115,      
        'bid_above': start_x + 230,         
        'adjusted_price': start_x + 343,    
        'square_meters': start_x,     
        'lot_size': start_x + 115,          
        'year': start_x + 230,              
    }   

    c.setStrokeColorRGB(0, 0, 0)

    for i, row in df.iterrows():
        y = start_y - i * row_height
        c.setFont("Bold", 11)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(80, y - i * 76, f" {row['Address']} ({row['Distance_meters']} meters from target)")

        c.setFont("Regular", 9)
        c.setFillColorRGB(60/255, 60/255, 60/255)
        y = start_y - i * 77

        c.drawString(columns['type'], y - 30, f"{row['Type']}")
        c.drawString(columns['asking_price'], y - 30, f"€ {row['Asking_price']:,}")
        c.drawString(columns['bid_above'], y - 30, f"{row['Bid_above_asking_pct']}")
        c.drawString(columns['adjusted_price'], y - 30, f"€ {row['Adjusted_price']:,}")
        c.drawString(columns['square_meters'], y - 58, f"{row['Square_meters']}")
        c.drawString(columns['lot_size'], y - 58, f"{row['Lot_size']}")
        c.drawString(columns['year'], y - 58, f"{row['Year']}")


def draw_page10_table(c, start_x=70, start_y=468, row_height=1):
    df = variables.market_rent_comps_df
    columns = {
        'type': start_x,
        'asking_price': start_x + 115,      
        'bid_above': start_x + 230,         
        'adjusted_price': start_x + 343,    
        'square_meters': start_x,     
        'lot_size': start_x + 115,          
        'year': start_x + 230,              
    }   

    c.setStrokeColorRGB(0, 0, 0)

    for i, row in df.iterrows():
        y = start_y - i * row_height
        c.setFont("Bold", 11)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(80, y - i * 76, f" {row['Address']} ({row['Distance_meters']} meters from target)")

        c.setFont("Regular", 9)
        c.setFillColorRGB(60/255, 60/255, 60/255)
        y = start_y - i * 77

        c.drawString(columns['type'], y - 30, f"{row['Type']}")
        c.drawString(columns['asking_price'], y - 30, f"€ {row['Asking_price']:,}")
        c.drawString(columns['bid_above'], y - 30, f"{row['Bid_above_asking_pct']}")
        c.drawString(columns['adjusted_price'], y - 30, f"€ {row['Adjusted_price']:,}")
        c.drawString(columns['square_meters'], y - 58, f"{row['Square_meters']}")
        c.drawString(columns['lot_size'], y - 58, f"{row['Lot_size']}")
        c.drawString(columns['year'], y - 58, f"{row['Year']}")


def return_on_equity_table_page_13(c):
    data = variables.return_on_equity_df
    start_x = 140
    start_y = 315

    c.setFont("Bold", 10)
    y = start_y - 13
    x = start_x + 60
    for col in data.columns[1:]:
        c.drawString(x, y, format_without_percent_number(col))
        x += 70

    y -= 17
    for i in range(len(data)):
        row = data.iloc[i]
        c.setFont("Bold", 10)
        c.drawString(start_x, y, row['LTV'])
        c.setFont("Light", 10)
        x = start_x + 60
        for val in row[1:]:
            c.drawString(x, y, f"{format_with_percent_number(val)}")
            x += 70
        y -= 17


def monthly_cash_flow_table_page_13(c):
    data = variables.monthly_cash_flow_df
    start_x = 140
    start_y = 168

    c.setFont("Bold", 10)
    y = start_y - 13
    x = start_x + 60
    for col in data.columns[1:]:
        c.drawString(x, y, format_without_percent_number(col))
        x += 70

    y -= 17
    for i in range(len(data)):
        row = data.iloc[i]
        c.setFont("Bold", 10)
        c.drawString(start_x, y, row['LTV'])
        c.setFont("Light", 10)
        x = start_x + 60
        for val in row[1:]:
            c.drawString(x, y, f"€{format_number(val)}")
            x += 70
        y -= 17


# Page drawing functions
def draw_page0(c, w, h):
    """Page 0: Cover page"""
    c.setFont("Medium", 14)
    c.setFillColorRGB(255, 255, 255)  
    c.drawString(87, 190, f"{variables.address}")  
    c.drawString(87, 135, f"{variables.report_date}")  
    c.drawString(87, 80, f"{variables.reference_date}")  


def draw_page1(c, w, h):   
    """Page 1: Summary page"""
    c.setFillColorRGB(16/255, 16/255, 16/255)
    draw_wrapped_text(c, f"This report is for the property located at {variables.address}. "
                          f"The values in this report are estimated as of {variables.reference_date}. "
                          f"The report is available via the link.", 70, 715, "Regular", 10, 
                          max_width=450, link=variables.report_link)

    c.setFont("Bold", 30)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(82, 645, f"€{format_number(variables.vacant_value)}")
    c.drawString(325, 645, f"€{format_number(variables.rented_value)}")
    c.drawString(82, 510, f"€{format_number(variables.market_rent)}")
    c.drawString(325, 510, f"{variables.wws_points}")

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


def draw_page2(c, w, h):
    """Page 2: Property overview"""
    c.setFont("Bold", 19)
    c.setFillColorRGB(0, 0, 0)  
    c.drawString(70, 690, f"{variables.address_short}")     
        
    draw_wrapped_text(c, f"{variables.property_overview_1}", 300, 660, "Regular", 10, max_width=270)
    draw_wrapped_text(c, f"{variables.property_overview_2}", 300, 580, "Regular", 10, max_width=270)

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)


def draw_page3(c, w, h):
    """Page 3: Vacant value and score"""
    c.setFont("Bold", 30)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(85, 675, f"€{format_number(variables.vacant_value)}")
    c.drawString(325, 675, f"{variables.vacant_value_score}%")

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)


def draw_page4(c, w, h):
    """Page 4: Vacant values comps table"""
    c.setFont("Regular", 10)
    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_page5_table(c,73)

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)


def draw_page5(c, w, h):
    """Page 5: Optimal asking price"""
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

    y_price_paid = start_y
    y_asking_price = start_y - 30

    c.setFont("Regular", 10)
    c.setFillColorRGB(60/255, 60/255, 60/255)

    for i, row in df.iterrows():
        x = start_x + i * col_gap
        c.drawString(x, y_price_paid, f"€{format_number(row['Price_paid'])}")
        c.drawString(x, y_asking_price, f"€{format_number(row['Asking_price'])}")   

    c.setStrokeColorRGB(1, 1, 1)
    c.setLineWidth(1)
    c.line(start_x - 110, (y_price_paid + y_asking_price) / 2 + 5,
           start_x - 110 + 500, (y_price_paid + y_asking_price) / 2 + 5)

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)


def draw_page6(c, w, h):
    """Page 6: Bidding statistics"""
    c.setFont("Bold", 30)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(87, 674, f"{to_percentage(variables.percent_sold_above_asking)}")
    c.drawString(325, 674, f"{to_percentage(variables.average_bidding)}")

    start_x=150
    start_y=115
    col_gap=49 
    df = variables.bid_vs_winning_chance_df

    y_price_paid = start_y
    y_asking_price = start_y - 29

    c.setFont("Regular", 10)
    c.setFillColorRGB(60/255, 60/255, 60/255)

    for i, row in df.iterrows():
        x = start_x + i * col_gap
        c.drawString(x, y_price_paid, f"€{format_number(row['Bid_offered'])}")
        c.drawString(x + 9, y_asking_price, f"{row['Chance_of_winning_pct']}%")   

    c.setStrokeColorRGB(1, 1, 1)
    c.setLineWidth(1)
    c.line(start_x - 110, (y_price_paid + y_asking_price) / 2 + 5,
           start_x - 110 + 500, (y_price_paid + y_asking_price) / 2 + 5)

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)


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

def draw_page9(c, w, h):
    c.setFont("Bold", 30)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(87, 674, f"{format_euro(variables.market_rent)}")
    c.drawString(325, 674, f"{to_percentage(variables.market_rent_score)}")

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)


def draw_page10(c, w, h):
    c.setFont("Regular", 10)
    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_page10_table(c,73,460)

    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)




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

def draw_page12(c, w, h):
    """Page 12: Common template"""
    c.setFont("Regular", 10)
    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)


def draw_wws_points_page(c, w, h):

    # --- Top values ---
    c.setFont("Bold", 27)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(86, 670, str(variables.wws_points))
    c.drawString(324, 670, f"€{variables.wws_points_rent:,}")

  
    # Bar settings
    bar_x0 = 70
    bar_x1 = 530
    bar_y = 475

    
    min_points = 30
    max_points = 500
    threshold = 186
    threshold_x = bar_x0 + (bar_x1 - bar_x0) * (threshold - min_points) / (max_points - min_points)
    c.setFillColorRGB(0.8, 0.8, 1)
    

    # Draw wws_points bubble
    wws_points = variables.wws_points
    wws_x = bar_x0 + (bar_x1 - bar_x0) * (wws_points - min_points) / (max_points - min_points)
    c.setFillColorRGB(0.2, 0.6, 1)
    c.circle(wws_x, bar_y, 7, stroke=0, fill=1)
    c.setFont("Regular", 10)
    c.setFillColorRGB(0, 0, 0)
    c.drawCentredString(wws_x, bar_y - 22, "You are here")


    sector_text = variables.sector_text
    # c.setFont("Bold", 12)
    # c.setFillColorRGB(1, 1, 1)
    # c.drawCentredString(300, 415, f"It seems like your propertyis in the {sector_text} sector!")

    y = 415
    x_start = 265

    # Set font and color for the first part
    c.setFont("Bold", 11)
    c.setFillColorRGB(0.53, 0.81, 0.98)
    text1 = "It seems like your property is in the"
    text1_width = stringWidth(text1, "Regular", 11)+5
    c.drawString(x_start - text1_width / 2, y, text1)

    # Set font and color for sector_text
    c.setFont("Bold", 11)
    c.setFillColorRGB(0.2, 0.6, 1)  # blue
    text2 = str(sector_text)
    text2_width = stringWidth(text2, "Bold", 11)
    c.drawString(x_start - text1_width / 2 + text1_width, y, f"{text2}")

    # Set font and color for the last part
    c.setFont("Bold", 11)
    c.setFillColorRGB(1, 1, 1)
    text3 = "sector!"
    c.drawString(x_start - text1_width / 2 + text1_width + text2_width, y, text3)

    # --- WWS points breakdown table ---
    left_values = list(variables.wws_points_breakdown_dict.values())[:7]
    right_values = list(variables.wws_points_breakdown_dict.values())[7:]
    table_y = 277
    left_x = 260
    right_x = 370
    row_h = 17
    c.setFont("Regular", 11)
    c.setFillColorRGB(0, 0, 0)
    for i in range(max(len(left_values), len(right_values))):
        if i == 5:
            y = table_y - i * row_h - (row_h / 2) + 3
        elif i > 5:
            # Shift all rows after the double row down by one row_h
            y = table_y - i * row_h - 10
        else:
            y = table_y - i * row_h
        
        # print(y)
        # Left column
        if i < len(left_values):
            # c.drawString(left_x, y, left_labels[i])
            c.drawString(left_x, y, str(left_values[i]))
        # Right column
        if i < len(right_values):
            # c.drawString(right_x, y, right_labels[i])
            c.drawString(right_x + 120, y, str(right_values[i]))
   

def draw_page13(c, w, h):
    """Page 13: Return on equity"""
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


def draw_page15(c, w, h):
    c.setFont("Regular", 10)
    c.drawString(200, 609, f"{variables.address}")
 
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
    c.drawString(240, 358, f"{format_euro(variables.management_costs, mode='int')}")
    c.drawString(240, 341, f"{format_euro(variables.maintenance_costs, mode='int')}")
    c.drawString(240, 324, f"{format_euro(variables.VVE_yearly, mode='int')}")
    c.drawString(240, 307, f"{format_euro(variables.erfpacht_amount, mode='int')}")
    c.drawString(240, 290, f"{format_euro(variables.other_running_costs, mode='int')}")
    c.setFont("Bold", 10)
    c.drawString(240, 275, f"{format_euro(variables.total_running_costs, mode='int')}")
 
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
 
    c.setFont("Light", 10)
    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

def draw_ep_online_excerpt_page(c, w, h):
    """Page 16: Energy label details"""
    c.setFont("Regular", 9.5)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(420, 665, str(variables.energy_label_register_dat))
    c.drawString(420, 640, str(variables.energy_label_expiration_date))
    c.drawString(420, 615, str(variables.energy_label_score))
    c.drawString(420, 590, str(variables.energy_label_certificate_holder))


def draw_energy_label_page(c, w, h):
    """Page 17: Energy label impact table"""
    draw_energy_label_impact_table(
        c,
        top_y=553,
        row_h=16,
        col_vv=168,
        col_dvv=214,
        col_mv=335,
        col_dmv=422,
        box_w=115,
        box_h=15,
        font_name="Regular",
        font_size=10
    )
    c.setFont("Regular", 10)
    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)


def draw_page18(c, w, h):
    """Page 18: Common template"""
    c.setFont("Regular", 10)
    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)


def draw_page19(c, w, h):
    """Page 19: Common template"""
    c.setFont("Regular", 10)
    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

def common_page_template(c, w, h):
    # Use robust defaults; change only top_y and row_h if you need micro-nudges.
   

    # footer / address right aligned
    c.setFont("Regular", 10)
    c.setFillColorRGB(130/255, 130/255, 130/255)
    draw_right_aligned(c, variables.address, 540, 23, "Regular", 10)

# Main execution
if __name__ == "__main__":
    filler = MultiPageTemplateFiller("Report_template.pdf")

    # Add overlays for all pages (0-19)
    filler.add_overlay(0, draw_page0)
    filler.add_overlay(1, draw_page1)
    filler.add_overlay(2, draw_page2)
    filler.add_overlay(3, draw_page3)
    filler.add_overlay(4, draw_page4)
    filler.add_overlay(5, draw_page5)
    filler.add_overlay(6, draw_page7)
    filler.add_overlay(7, draw_page8)
    filler.add_overlay(8, draw_page9)
    filler.add_overlay(9, draw_page10)
    filler.add_overlay(10, draw_page11)
    filler.add_overlay(11, draw_wws_points_page)
    filler.add_overlay(12, draw_page13)
    filler.add_overlay(13, draw_page14)
    filler.add_overlay(14, draw_page15)
    filler.add_overlay(15, draw_ep_online_excerpt_page)
    filler.add_overlay(16, draw_energy_label_page)
    filler.add_overlay(17, common_page_template)
    filler.add_overlay(18, common_page_template)
    filler.add_overlay(19, common_page_template)

    # Save the PDF
    filler.save("Template_filled.pdf")

    # Generate and insert images for all pages
    # Page 2 images
    rounded_img = "assets/pictures/property_google_photo_rounded.png"
    make_rounded_image("assets/pictures/property_google_photo.png", rounded_img, radius=15, size=(240, 240))
    insert_image("Template_filled.pdf", rounded_img, fitz.Rect(70, 170, 280, 360), page_num=2)

    cadastral_map_img = "assets/pictures/cadastral_map.png"
    insert_image("Template_filled.pdf", cadastral_map_img, fitz.Rect(70, 460, 530, 800), page_num=2)

    # Page 3 charts
    market_rent_range_chart()
    vacant_value_chart_img = "assets/charts/vacant_value_chart.png"
    insert_image("Template_filled.pdf", vacant_value_chart_img, fitz.Rect(70, 200, 530, 650), page_num=3)

    chart2()
    chart2_img = "assets/charts/chart2.png"
    insert_image("Template_filled.pdf", chart2_img, fitz.Rect(70, 450, 530, 900), page_num=3)

    # Page 4 image
    rounded_img = "assets/pictures/vacant_values_comps_df_rounded.png"
    make_rounded_image("assets/pictures/vacant_values_comps_df.png", rounded_img, radius=15, size=(800, 400))
    insert_image("Template_filled.pdf", rounded_img, fitz.Rect(60, 120, 540, 350), page_num=4)

    # Page 5 chart
    page6_chart_img = "assets/charts/chart_page_6.png"
    insert_image("Template_filled.pdf", page6_chart_img, fitz.Rect(70, 210, 530, 900), page_num=5)

    # Page 6 chart
    create_line_chart_page_7(variables.bid_vs_winning_chance_df)
    page7_chart_img = "assets/charts/line_chart_page_7.png"
    insert_image("Template_filled.pdf", page7_chart_img, fitz.Rect(0, 450, 550, 710), page_num=6)

    # Page 7 charts
    pie_chart_percentage_page_8(variables.vacant_value_demand_score, color="#c9a0ff")
    page8_chart_img = "assets/charts/pie_chart_page_8.png"
    insert_image("Template_filled.pdf", page8_chart_img, fitz.Rect(90, 150, 250, 300), page_num=7)

    create_line_chart_page_8(variables.vacant_value_index_df)
    page8_line_chart_img = "assets/charts/line_chart_page_8.png"
    insert_image("Template_filled.pdf", page8_line_chart_img, fitz.Rect(63, 480, 520, 860), page_num=7)

    # Page 8 charts
    market_rent_score = variables.market_rent_score
    market_rent_low = variables.market_rent_low
    market_rent_high = variables.market_rent_high
    market_rent_expected = market_rent_low + (market_rent_high - market_rent_low) * market_rent_score
    values = [market_rent_high, market_rent_expected, market_rent_low]
    categories = ['Upper bound', 'Expected value', 'Lower bound']
    create_market_rent_range_chart_page_9(values, categories, max_value=market_rent_high, save_path="assets/charts/vacant_value_chart_page_9.png")
    vacant_value_chart_img = "assets/charts/vacant_value_chart_page_9.png"
    insert_image("Template_filled.pdf", vacant_value_chart_img, fitz.Rect(70, 200, 530, 650), page_num=8)

    sqm_ref = variables.sqm
    low_per_sqm = variables.market_rent_low / sqm_ref
    expected_per_sqm = variables.market_rent / sqm_ref
    high_per_sqm = variables.market_rent_high / sqm_ref
    plot_rent_chart_page_9(variables.market_rent_per_sqm_comps_df, sqm_ref, low_per_sqm, expected_per_sqm, high_per_sqm)
    plot_rent_chart_page_9_img = "assets/charts/plot_rent_chart_page_9.png"
    insert_image("Template_filled.pdf", plot_rent_chart_page_9_img, fitz.Rect(70, 450, 530, 900), page_num=8)

    # Page 9 image
    rounded_img_10 = "assets/pictures/vacant_values_comps_df.png"
    make_rounded_image("assets/pictures/vacant_values_comps_df.png", rounded_img_10, radius=15, size=(800, 400))
    insert_image("Template_filled.pdf", rounded_img_10, fitz.Rect(60, 120, 540, 350), page_num=9)

    # Page 10 charts
    page11_chart_img = "assets/charts/pie_chart_page_11.png"
    pie_chart_percentage_page_8(variables.market_rent_demand_score, color="#D3D3D3", save_path=page11_chart_img)
    insert_image("Template_filled.pdf", page11_chart_img, fitz.Rect(100, 155, 240, 295), page_num=10)

    page11_line_chart_img = "assets/charts/line_chart_page_11.png"
    create_line_chart_page_8(variables.market_rent_index_df, x_col="Date", y_col="Rent_index", save_path=page11_line_chart_img)
    insert_image("Template_filled.pdf", page11_line_chart_img, fitz.Rect(63, 480, 520, 860), page_num=10)

    # Page 15 chart
    sizes = [variables.running_costs_to_effective_rent_percentage, variables.net_rental_income_to_effective_rent_percentage]
    labels = ['Exploitatiekosten', 'Nettohuurinkomsten']
    colors = ['#F0E6F7', '#C8A8C8']
    explode = (0.1, 0)
    create_pie_chart(sizes, labels, colors=colors, explode=explode)
    page15_chart_img = "assets/charts/chart_page_15.png"
    insert_image("Template_filled.pdf", page15_chart_img, fitz.Rect(280, 360, 580, 690), page_num=14)

    # Page 16 image
    insert_image("Template_filled.pdf", "assets/pictures/energy_label_C.png", fitz.Rect(30, 165, 290, 280), page_num=15)

    print("✅ Multi-page PDF generated: Template_filled.pdf")