<p align="center">
  <img src="assets/icon.svg" width="96" alt="img2text icon">
</p>

<h1 align="center">img2text</h1>

<p align="center">Extract text, tables and layout from screenshots and images, offline.</p>

---

img2text is a desktop app built with [Flet](https://flet.dev) and [RapidOCR](https://github.com/RapidAI/RapidOCR) (ONNX Runtime). Pick an image and it returns plain text that keeps the original layout as closely as possible.

## Features

- **Offline OCR**: everything runs locally, nothing is uploaded.
- **Clipboard input**: copy or snip an image (Win+Shift+S, "Copy image" in a browser, or an image file in Explorer) and it converts automatically. Turn this off with the **Auto-paste** switch, then use **Paste** or Ctrl+V instead.
- **Layout aware**: keeps columns, spacing and horizontal rules.
- **Table detection**: aligned columns become a monospaced text table with a header separator.
- **Extras**: checkboxes (`[✓]` / `[ ]`), chips/tags (`[label]`), photos (`[photo]`) and peso amounts (`₱`).
- **Dark screenshots**: dark-mode images are inverted and upscaled automatically.
- **GPU support**: uses CUDA when `onnxruntime-gpu` and a CUDA device are available, otherwise CPU.
- Copy to clipboard or save as `.txt`, with light and dark themes.

Supported formats: PNG, JPG/JPEG, WEBP and BMP.

## Run from source

Requires Python 3.12.

```powershell
pip install -r requirements.txt
python app.py
```

## Build the exe

```powershell
pip install pyinstaller
.\build.ps1
```

[build.ps1](build.ps1) runs `flet pack` with the icon, assets, version info, the OCR model files, and a list of unused libraries to leave out of the exe.

The app is built as a folder in `dist\img2text\`. Run `img2text.exe` inside it and keep the `_internal` folder next to the exe. A folder build starts faster than a single exe because it doesn't unpack itself on every launch. To share it, zip the whole `dist\img2text` folder.

## Project layout

```
img2text/
├── app.py           # OCR pipeline + Flet UI
├── assets/
│   ├── icon.svg     # source icon
│   ├── icon.ico     # Windows exe and window icon
│   └── icon.png
├── build.ps1        # builds the exe
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE.md
```

## License

[MIT](LICENSE.md) © 2026 Dalmanski
