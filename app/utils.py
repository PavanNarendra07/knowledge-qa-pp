import pandas as pd
from pypdf import PdfReader
import docx

def read_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
 