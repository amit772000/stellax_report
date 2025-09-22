from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import stringWidth
from PyPDF2 import PdfWriter, PdfReader, Transformation
from reportlab.pdfgen.canvas import Canvas
import fitz  # PyMuPDF
import io
import variables
import os

# Register fonts
pdfmetrics.registerFont(TTFont('Bold', 'assets/fonts/Inter-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Light', 'assets/fonts/Inter-Light.ttf'))
pdfmetrics.registerFont(TTFont('Regular', 'assets/fonts/Inter.ttf'))
pdfmetrics.registerFont(TTFont('Medium', 'assets/fonts/Inter-Medium.ttf'))


class GenerateFromTemplate:

        
    def __init__(self, template, page_number=0):
        self.template_pdf = PdfReader(open(template, "rb"))
        self.template_page = self.template_pdf.pages[page_number]

        self.packet = io.BytesIO()
        self.c = Canvas(
            self.packet,
            pagesize=(
                self.template_page.mediabox.width,
                self.template_page.mediabox.height
            )
        )

    def addText(self, text, pos, font_name, font_size, r, g, b, max_width=None, link=None):
        x, y = pos
        self.c.setFont(font_name, font_size)
        self.c.setFillColorRGB(r/255, g/255, b/255)

        if max_width is None:
            # Draw single-line text
            self.c.drawString(x, y, text)

            if link:
                text_width = stringWidth(text, font_name, font_size)
                text_height = font_size + 2
                self.c.linkURL(link, (x, y, x + text_width, y + text_height), relative=0)
        else:
            # Wrapped version
            words = text.split(" ")
            line = ""
            for word in words:
                test_line = (line + " " + word).strip()
                if stringWidth(test_line, font_name, font_size) <= max_width:
                    line = test_line
                else:
                    self.c.drawString(x, y, line)
                    y -= font_size + 4
                    line = word
            if line:
                self.c.drawString(x, y, line)

                if link:
                    text_width = stringWidth(line, font_name, font_size)
                    text_height = font_size + 2
                    self.c.linkURL(link, (x, y, x + text_width, y + text_height), relative=0)
   
    def text_width(self, text, font_name, font_size):
        """Get width of text in points (useful for right-aligning values)"""
        return self.c.stringWidth(text, font_name, font_size)

    def merge(self):
        """Finalize ReportLab canvas and merge with template page"""
        self.c.save()
        self.packet.seek(0)
        result_pdf = PdfReader(self.packet)
        result = result_pdf.pages[0]

        op = Transformation().rotate(0).translate(tx=0, ty=0)
        result.add_transformation(op)

        self.template_page.merge_page(result)

        self.output = PdfWriter()
        self.output.add_page(self.template_page)

    def generate(self, dest):
        """Save merged PDF"""
        with open(dest, "wb") as f:
            self.output.write(f)


def insert_image(pdf_path, img_path, rect):
    """Insert image into an existing PDF using PyMuPDF"""
    doc = fitz.open(pdf_path)
    page = doc[0]
    page.insert_image(rect, filename=img_path)
    doc.save(pdf_path, incremental=True, encryption=0)
    doc.close()

def format_number(n):
    """
    Convert numbers into K, M, B format without decimals.
    4-digit numbers are not converted.
    Examples:
        2339 -> '2,339'
        531000 -> '531K'
        1250000 -> '1M'
        2000000000 -> '2B'
    """
    if n is None:
        return "0"

    n = int(n)

    if n >= 10_000_000_000:  # 10B+
        return f"{n // 1_000_000_000}B"
    elif n >= 1_000_000_000:  # 1B–9,999,999,999
        return f"{n // 1_000_000_000}B"
    elif n >= 1_000_000:  # 1M–999,999,999
        return f"{n // 1_000_000}M"
    elif n >= 10_000:  # 10,000–999,999
        return f"{n // 1_000}K"
    else:  # 0–9,999
        return f"{n:,}"


# -------------------------------
# Usage
# -------------------------------
gen = GenerateFromTemplate("Report_template-2.pdf", 0)

# Add variable text
gen.addText(f"This report is for the property located at {variables.address}. The values in this report are estimated as of 01.01.2025. The report is available via the link.", (70, 715), "Medium", 10, 16, 16, 16, max_width=450, link=variables.report_link)
gen.addText(f"€{format_number(variables.vacant_value)}", (82, 645), "Bold", 30, 0, 0, 0)
gen.addText(f"€{format_number(variables.rented_value)}", (325, 645), "Bold", 30, 0, 0, 0)
gen.addText(f"€{format_number(variables.market_rent)}", (82, 510), "Bold", 30, 0, 0, 0)
gen.addText(f"{variables.wws_points}", (325, 510), "Bold", 30, 0, 0, 0)

# Data and sources
gen.addText(f"{variables.property_type}", (225, 257), "Regular", 10, 130, 130, 130)
gen.addText(f"{variables.property_type_source}", (376, 257), "Regular", 10, 130, 130, 130)

gen.addText(f"{variables.sqm}", (225, 240), "Regular", 10, 130, 130, 130)
gen.addText(f"{variables.sqm_source}", (376, 240), "Regular", 10, 130, 130, 130)

gen.addText(f"{variables.year}", (225, 223), "Regular", 10, 130, 130, 130)
gen.addText(f"{variables.year_source}", (376, 223), "Regular", 10, 130, 130, 130)

gen.addText(f"{variables.lot_size}", (225, 206), "Regular", 10, 130, 130, 130)
gen.addText(f"{variables.lot_size_source}", (376, 206), "Regular", 10, 130, 130, 130)

gen.addText(f"{variables.energy_label}", (225, 189), "Regular", 10, 130, 130, 130)
gen.addText(f"{variables.energy_label_source}", (376, 189), "Regular", 10, 130, 130, 130)

gen.addText(f"{variables.contract_rent}/ma", (225, 172), "Regular", 10, 130, 130, 130)
gen.addText(f"{variables.contract_rent_source}", (376, 172), "Regular", 10, 130, 130, 130)

gen.addText(f"{variables.wws_points}", (225, 155), "Regular", 10, 130, 130, 130)
gen.addText(f"{variables.wws_points_source}", (376, 155), "Regular", 10, 130, 130, 130)

gen.addText(f"{variables.wws_points_rent}/ma", (225, 138), "Regular", 10, 130, 130, 130)
gen.addText(f"{variables.wws_rent_source}", (376, 138), "Regular", 10, 130, 130, 130)

gen.addText(f"{variables.vve}/ma", (225, 121), "Regular", 10, 130, 130, 130)
gen.addText(f"{variables.vve_source}", (376, 121), "Regular", 10, 130, 130, 130)

gen.addText(f"{variables.erfpact_date} (bought off)", (225, 104), "Regular", 10, 130, 130, 130)
gen.addText(f"{variables.erfpact_date_source}", (376, 104), "Regular", 10, 130, 130, 130)

gen.addText(f"{variables.erfpacht_amount}/j", (225, 87), "Regular", 10, 130, 130, 130)
gen.addText(f"{variables.erfpacht_amount_source}", (376, 87), "Regular", 10, 130, 130, 130)

# Merge text overlay with template
gen.merge()
gen.generate("Template_filled.pdf")

# # Insert image (example: cadastral map)
# x0, y0 = 113, 145
# x1, y1 = x0 + 120, y0 + 120
# insert_image("Template_filled.pdf", "assets/pictures/cadastral_map.png", fitz.Rect(x0, y0, x1, y1))

print("✅ Report generated: Template_filled.pdf")
