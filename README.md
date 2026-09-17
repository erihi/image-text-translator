# i2t

Windows desktop tool that takes an image (via drag & drop), runs OCR on it,
and translates the recognized text. Built as two native C++ / Win32
executables chained together: `i2t.exe | s2s.exe`.

## Pipeline

1. **UI** - a Win32 drag-and-drop window (`i2t/DragDrop.cpp/h`) receives an
   image file.
2. **OCR** (`i2t.sln` / `i2t/`) - text detection + recognition via
   PaddleOCR's PP-OCRv4 models, run through the Paddle Inference C++ SDK
   (`ocr_det.*`, `ocr_rec.*`, `preprocess_op.*`, `postprocess_op.*`,
   `clipper.*`, `utility.*`). Prints the recognized text to stdout.
3. **Translation** (`s2s.sln` / `s2s/`) - `s2s.exe` reads the OCR text from
   stdin (`FileName.cpp`) and translates it with an NLLB model via
   CTranslate2 + a SentencePiece tokenizer. A simple button UI lets you
   pick the target language (Korean/English/Chinese) and shows the
   translated result.

Run the two together as `i2t.exe | s2s.exe` (see `run.bat` in a built/
packaged distribution).

## Building

This repository contains only the source code that is original to this
project or explicitly permitted to be redistributed (see `NOTICE`). The
following are **not** included and must be installed locally before the
project will build - point each Visual Studio project's include/library
paths at wherever you install them:

| Dependency | License | Notes |
|---|---|---|
| OpenCV (4.12.x) | Apache-2.0 | image I/O / processing (`i2t`) |
| Paddle Inference SDK | Apache-2.0 | OCR model runtime (`i2t`) |
| CTranslate2 | MIT | translation model runtime (`s2s`) |
| SentencePiece | Apache-2.0 | tokenizer (`s2s`) |
| Intel oneDNN / MKL-ML | Apache-2.0 | CPU inference backend (pulled in by Paddle Inference) |
| NVIDIA cuDNN | NVIDIA SLA | only if building with GPU support |

You will also need:
- The PP-OCRv4 detection/recognition inference models (place them under
  `i2t/inference/<model_name>/`, matching the paths the project expects) -
  download these from PaddleOCR's own model zoo.
- An NLLB checkpoint converted for CTranslate2 (place it under
  `s2s/nllb_ct2/`, alongside `s2s.exe`).

Model weight files are intentionally excluded from this repo (see
`.gitignore`) - see the license note below before sourcing an NLLB
checkpoint.

Open `i2t.sln` and `s2s.sln` in Visual Studio 2022 (v143 toolset), set the
include/lib paths for the dependencies above, and build the `x64` `Release`
or `Debug` configuration of each.

## License

This project's own code is licensed under the Apache License 2.0 (see
`LICENSE`). It incorporates modified source from
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0) and
unmodified source from Angus Johnson's
[Clipper](https://sourceforge.net/projects/polyclipping/) library (Boost
Software License 1.0). Full attribution and file-level detail is in
`NOTICE`.

**If you plan to use a machine-translation model with this project**,
check that model's license separately. Meta's official NLLB checkpoints,
for example, are released under CC-BY-NC 4.0 and cannot be used
commercially.
