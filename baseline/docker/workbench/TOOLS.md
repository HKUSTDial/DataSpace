# Data Workbench Runtime

This task-local runtime is offline and already provisioned. Package installation
is unavailable. Use the following generic utilities instead of attempting to
download dependencies.

- CSV, JSON, and Markdown: Python 3, pandas, NumPy, PyArrow, jq, and ripgrep.
- SQLite: sqlite3 and Python's sqlite3 module.
- PDF: pdftotext, pdfinfo, pdftoppm, PyMuPDF (`fitz`), pypdf, and pdfplumber.
- Images and OCR: Pillow, OpenCV (`cv2`), pytesseract, and the tesseract CLI
  with English and Simplified Chinese language data.
- Video: ffmpeg, ffprobe, and OpenCV. Extract selected frames before visual
  inspection. Offline speech recognition is not installed.

The benchmark workspace is read-only. Write scripts, rendered pages, extracted
frames, intermediate files, and the final `prediction.csv` only to the provided
output directory.
