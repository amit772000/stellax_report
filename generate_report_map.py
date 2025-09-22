import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
from PyPDF2 import PdfReader, PdfWriter
import variables

# Register fonts
pdfmetrics.registerFont(TTFont('Bold', 'assets/fonts/Inter-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Light', 'assets/fonts/Inter-Light.ttf'))
pdfmetrics.registerFont(TTFont('Regular', 'assets/fonts/Inter.ttf'))
pdfmetrics.registerFont(TTFont('Medium', 'assets/fonts/Inter-Medium.ttf'))

# Define replacement values (from variables.py)
VALUES = {
    "[address]": variables.address,
    "[reference_date]": variables.reference_date,
    "[report_link]": variables.report_link,
    "[vacant_value]": f"€ {variables.vacant_value:,}",
    "[rented_value]": f"€ {variables.rented_value:,}",
    "[market_rent]": f"€ {variables.market_rent:,}/mo",
    "[wws_points]": f"{variables.wws_points} pts"
}

def replace_placeholders(input_pdf, output_pdf):
    doc = fitz.open(input_pdf)

    for page in doc:
        blocks = page.get_text("blocks")  # extract text blocks with coords
        print(blocks)
        for b in blocks:
            text = b[4].strip()
            if text in VALUES:
                x0, y0, x1, y1 = b[:4]  # block rectangle
                replacement = VALUES[text]
                print(text)
                # 1. Cover placeholder with white rectangle
                page.draw_rect([x0, y0, x1, y1], color=(1,1,1), fill=(1,1,1))

                # 2. Insert replacement text
                page.insert_text(
                    (x0, y0),                # bottom-left anchor
                    replacement,
                    fontname="helv",         # or match your registered font
                    fontsize=10,
                    color=(0, 0, 0)
                )

    doc.save(output_pdf)
    doc.close()
    print(f"✅ Generated {output_pdf}")


if __name__ == "__main__":
    replace_placeholders("Report_variables_mapping-2.pdf", "Report_filled.pdf")
