
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pymupdf
from PyPDF2 import PdfWriter, PdfReader, Transformation
import io
from reportlab.pdfgen.canvas import Canvas
import variables

pdfmetrics.registerFont(TTFont('Bold', 'assets/fonts/Inter-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Light', 'assets/fonts/Inter-Light.ttf'))
pdfmetrics.registerFont(TTFont('Regular', 'assets/fonts/Inter.ttf'))
pdfmetrics.registerFont(TTFont('Medium', 'assets/fonts/Inter-Medium.ttf'))


class GenerateFromTemplate:
    def __init__(self,template, page_number):
        self.template_pdf = PdfReader(open(template, "rb"))
        self.template_page= self.template_pdf.pages[page_number]

        self.packet = io.BytesIO()
        self.c = Canvas(self.packet,pagesize=(self.template_page.mediabox.width,self.template_page.mediabox.height))
        
    
    def addText(self,text,point, font, font_size, R, G, B):
        self.c.setFillColorRGB(R/256,G/256,B/256) #choose your font colour
        self.c.setFont(font, font_size)
        self.c.drawString(point[0],point[1],text)
        
    def merge(self):
        self.c.save()
        self.packet.seek(0)
        result_pdf = PdfReader(self.packet)
        result = result_pdf.pages[0]

        self.output = PdfWriter()

        op = Transformation().rotate(0).translate(tx=0, ty=0)
        result.add_transformation(op)
        self.template_page.merge_page(result)
        self.output.add_page(self.template_page)
    
    def generate(self,dest):
        outputStream = open(dest,"wb")
        self.output.write(outputStream)
        outputStream.close()
        
        
    def text_width(self, text, font_name, font_size):
        text_length=self.c.stringWidth(text , font_name, font_size)
        return text_length



gen = GenerateFromTemplate("Report_template.pdf", 0)

#add text to pdf
gen.addText(variables.address,(70,315), "Medium", 16, 16, 16, 16)
gen.merge()
gen.generate('Template_filled.pdf')

#insert image
doc = pymupdf.open('Template_filled.pdf')
page = doc[0]
x0=113
y0=145
x1=x0+42
y1=y0+42
page.insert_image(pymupdf.Rect(x0,y0,x1,y1),filename="assets/pictures/cadastral_map.png")   
doc.save(doc.name, incremental=True, encryption=0)
doc.close()