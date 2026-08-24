"""Generate mock PDF contracts for quick testing and evaluation."""
from pathlib import Path
from pypdf import PdfWriter
import io

def create_sample_pdf(output_path: Path, text: str):
    """Creates a basic PDF file with text content."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    
    y = 750
    for line in text.split("\n"):
        if y < 50:
            can.showPage()
            y = 750
        can.drawString(50, y, line[:90])
        y -= 15
    can.save()

    packet.seek(0)
    with open(output_path, "wb") as f:
        f.write(packet.read())

if __name__ == "__main__":
    pass
