import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import os
import numpy as np
import cairosvg


def create_demand_score_chart(path="sample_chart.png", percentage=95, title="Demand Score"):
    """
    Create a dynamic circular progress chart
    
    Parameters:
    percentage (float): Percentage value to display (0-100)
    title (str): Title text to display above the chart
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Set background color to light pink/purple
    fig.patch.set_facecolor('#F5E6F5')
    ax.set_facecolor('#F5E6F5')
    
    # Add border around the entire figure
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=2, 
                        edgecolor='#D8A8D8', facecolor='none', transform=fig.transFigure)
    fig.patches.append(rect)
    
    # Circle parameters
    center = (0.5, 0.4)  # Slightly lower to accommodate title
    radius = 0.25
    line_width = 15  # Thickness of the progress ring
    
    # Create full background circle (white/light gray outline)
    full_circle = plt.Circle(center, radius, fill=False, 
                            color='#D0D0D0', linewidth=line_width)
    ax.add_patch(full_circle)
    
    # Progress arc
    # Calculate angle for the progress (starts from top, goes clockwise)
    start_angle = 90  # Start from top
    progress_angle = (percentage / 100) * 360  # Total progress in degrees
    
    # Create progress arc using matplotlib's arc
    if percentage > 0:
        # Create the progress arc
        theta = np.linspace(0, progress_angle * np.pi / 180, int(progress_angle * 2))
        
        # Calculate arc points
        x_progress = []
        y_progress = []
        
        for angle in theta:
            # Start from top (90 degrees) and go clockwise
            actual_angle = np.pi/2 - angle
            x_progress.append(center[0] + radius * np.cos(actual_angle))
            y_progress.append(center[1] + radius * np.sin(actual_angle))
        
        # Draw the progress as a thick line
        if len(x_progress) > 1:
            ax.plot(x_progress, y_progress, color='#A0A0A0', 
                   linewidth=line_width, solid_capstyle='round')
    
    # Add percentage text in the center
    ax.text(center[0], center[1], f'{percentage}%', 
            fontsize=32, fontweight='bold', ha='center', va='center',
            color='black', family='sans-serif')
    
    # Add title at the top
    ax.text(0.5, 0.85, title, 
            fontsize=18, fontweight='bold', ha='center', va='center',
            color='black', family='sans-serif', transform=ax.transAxes)
    
    # Remove axes and set equal aspect
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Remove margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.tight_layout()
    # plt.savefig(path, dpi=200, bbox_inches="tight", transparent=True)
    # plt.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.0, transparent=True)
    plt.savefig('chart.svg', format='svg', bbox_inches='tight', pad_inches=0.0, transparent=True)
    plt.close()
    
    # plt.show()


# 1. Generate a sample image (matplotlib)
def generate_sample_chart(path="sample_chart.png"):
    plt.figure(figsize=(3, 2))
    plt.plot([1, 2, 3], [4, 5, 6], color='blue', linewidth=3)
    plt.title("Sample Chart")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close()

# 2. Place the image on a blank PDF
def place_image_on_pdf(img_path, pdf_path="output.pdf", rect=fitz.Rect(50, 100, 350, 380)):
    # Create a blank PDF
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size
    # Insert the image
    page.insert_image(rect, filename=img_path, keep_proportion=True)
    doc.save(pdf_path)
    doc.close()



import cairosvg

def place_svg_on_pdf(svg_path, pdf_path="output.pdf", rect=fitz.Rect(50, 100, 350, 380)):
    # Convert SVG → high-res PNG
    png_path = svg_path.replace(".svg", ".png")
    cairosvg.svg2png(url=svg_path, write_to=png_path, dpi=300)

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    page.insert_image(rect, filename=png_path, keep_proportion=True)
    doc.save(pdf_path)
    doc.close()


# 3. Open the PDF for preview (Windows)
def open_pdf(pdf_path):
    os.startfile(pdf_path)

# --- Usage ---
if __name__ == "__main__":
    # img_file = "sample_chart.png"
    img_file = "chart.svg"
    pdf_file = "output.pdf"
    # generate_sample_chart(img_file)
    create_demand_score_chart(img_file, percentage=75, title="Demand Score")
    # place_image_on_pdf(img_file, pdf_file)
    place_svg_on_pdf(img_file, pdf_file)
    open_pdf(pdf_file)