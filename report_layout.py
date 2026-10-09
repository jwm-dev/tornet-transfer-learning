"""Small checked layout helper for a four-page classroom study report."""
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

WIDTH, HEIGHT = 612, 792
LEFT, RIGHT, BOTTOM = 44, 568, 46
CONTENT = RIGHT - LEFT
NAVY = colors.HexColor("#19344d")
TEAL = colors.HexColor("#247e83")
INK = colors.HexColor("#23313d")
MUTED = colors.HexColor("#536675")
PALE = colors.HexColor("#edf3f7")


class Report:
    def __init__(self, output: Path):
        output.parent.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.c = canvas.Canvas(str(output), pagesize=(WIDTH, HEIGHT))
        self.c.setTitle("Adapting a Pretrained Radar CNN for Tornado Detection")
        self.c.setAuthor("Team 2 - Jeffrey Gregory and Jake Van Slyke - technical study")
        self.page_number = 0
        self.styles = {
            "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.4,
                                   leading=12.8, textColor=INK, spaceAfter=7),
            "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.1,
                                    leading=10.7, textColor=MUTED, spaceAfter=5),
            "caption": ParagraphStyle("caption", fontName="Helvetica", fontSize=8.1,
                                      leading=10.6, textColor=MUTED, spaceAfter=6),
            "table": ParagraphStyle("table", fontName="Helvetica", fontSize=8.1,
                                    leading=10.5, textColor=INK),
            "table_head": ParagraphStyle("table_head", fontName="Helvetica-Bold",
                                         fontSize=8.1, leading=10.5, textColor=colors.white),
        }

    def new_page(self, section, title):
        if self.page_number:
            self.c.showPage()
        self.page_number += 1
        self.c.setFillColor(NAVY)
        self.c.rect(0, HEIGHT - 19, WIDTH, 19, stroke=0, fill=1)
        self.c.setFillColor(MUTED)
        self.c.setFont("Helvetica", 8)
        self.c.drawString(LEFT, HEIGHT - 40, "TORNET / CONTROLLED TRANSFER LEARNING STUDY")
        self.c.drawRightString(RIGHT, HEIGHT - 40, "Jeffrey Gregory & Jake Van Slyke")
        self.c.setFillColor(TEAL)
        self.c.setFont("Helvetica-Bold", 9)
        self.c.drawString(LEFT, HEIGHT - 68, section.upper())
        self.c.setFillColor(NAVY)
        self.c.setFont("Helvetica-Bold", 20)
        self.c.drawString(LEFT, HEIGHT - 95, title)
        self.c.setStrokeColor(colors.HexColor("#d4dfe7"))
        self.c.line(LEFT, 37, RIGHT, 37)
        self.c.setFillColor(MUTED)
        self.c.setFont("Helvetica", 7.5)
        self.c.drawString(LEFT, 25, "Deep Learning Unit 4 | Team review and shared understanding required")
        self.c.drawRightString(RIGHT, 25, f"{self.page_number} / 4")
        return HEIGHT - 116

    def paragraph(self, text, y, *, style="body", x=LEFT, width=CONTENT):
        item = Paragraph(text, self.styles[style])
        _, height = item.wrap(width, HEIGHT)
        if y - height < BOTTOM:
            raise ValueError(f"Page {self.page_number} paragraph exceeds margin: {text[:100]}")
        item.drawOn(self.c, x, y - height)
        return y - height - self.styles[style].spaceAfter

    def heading(self, text, y):
        self.c.setFillColor(NAVY)
        self.c.setFont("Helvetica-Bold", 11)
        self.c.drawString(LEFT, y - 9, text)
        return y - 24

    def table(self, rows, widths, y, *, highlight=None):
        cells = [[Paragraph(str(value), self.styles["table_head" if index == 0 else "table"])
                  for value in row] for index, row in enumerate(rows)]
        table = Table(cells, colWidths=widths, hAlign="LEFT")
        commands = [("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE])]
        if highlight is not None:
            commands.append(("BACKGROUND", (0, highlight), (-1, highlight), colors.HexColor("#ddefeb")))
        table.setStyle(TableStyle(commands))
        _, height = table.wrap(CONTENT, HEIGHT)
        if y - height < BOTTOM:
            raise ValueError(f"Page {self.page_number} table exceeds margin")
        table.drawOn(self.c, LEFT, y - height)
        return y - height - 10

    def image(self, path, y, *, width=CONTENT, max_height=None, x=LEFT):
        with Image.open(path) as source:
            w, h = source.size
        height = width * h / w
        if max_height and height > max_height:
            width *= max_height / height
            height = max_height
        if y - height < BOTTOM:
            raise ValueError(f"Page {self.page_number} image exceeds margin")
        self.c.drawImage(str(path), x, y - height, width=width, height=height)
        return y - height - 8

    def save(self):
        if self.page_number != 4:
            raise ValueError("Report must contain four pages")
        self.c.save()
        if len(PdfReader(self.output).pages) != 4:
            raise ValueError("Saved report must contain four pages")
