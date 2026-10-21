# Builds dist\img2text\img2text.exe (one-folder, fast startup).
# The excludes keep unrelated libraries in the global Python out of the bundle.
$exclude = "torch tensorflow transformers pandas matplotlib sklearn scipy IPython tkinter gi zmq PyQt5 PyQt6 PySide2 PySide6".Split(" ") | ForEach-Object { "--exclude-module", $_ }
py -3.12 -m flet_cli.cli pack app.py -D -y --name img2text --icon assets/icon.ico --add-data "assets;assets" `
    --product-name img2text --file-description img2text --product-version 1.0.0 --file-version 1.0.0.0 `
    --copyright "Copyright (c) 2026 Dalmanski" -- --collect-data rapidocr_onnxruntime @exclude
