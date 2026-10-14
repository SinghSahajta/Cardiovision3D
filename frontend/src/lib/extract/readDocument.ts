/**
 * Get text out of a report, entirely inside the browser.
 *
 * Nothing is uploaded. A lab report is a medical record, and the most defensible
 * thing a tool can do with one is never transmit it - so the PDF is parsed and
 * any OCR is run here, in the page.
 *
 * Two paths, and the first is far better when it is available:
 *
 *   digital PDF -> pdf.js text layer -> the publisher's own characters, exact
 *   scan/photo  -> rasterise -> tesseract.js -> a best guess
 *
 * Most lab reports are generated as digital PDFs, so most of the time no OCR
 * runs at all and there is no recognition error to worry about. The distinction
 * is reported back as `method` so the review table can say which one happened:
 * "read from the PDF's text layer" and "recognised from an image" deserve
 * different levels of trust from whoever is checking the numbers.
 */

import * as pdfjs from 'pdfjs-dist';
import type { PDFDocumentProxy } from 'pdfjs-dist';

// Vite resolves this to a hashed asset URL and bundles the worker, so the PDF
// path works with no network at all.
import PdfWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

pdfjs.GlobalWorkerOptions.workerSrc = PdfWorker;

export type ReadMethod = 'pdf-text' | 'ocr';

export interface ReadResult {
  text: string;
  method: ReadMethod;
  pages: number;
  /** 0-1, only meaningful for OCR; the text layer is not a guess. */
  confidence?: number;
}

export interface ReadProgress {
  stage: 'reading' | 'rendering' | 'recognising' | 'done';
  page?: number;
  pages?: number;
  /** 0-1 within the current stage. */
  ratio?: number;
}

/** Below this many characters a "text layer" is really just page furniture. */
const MIN_TEXT_LAYER_CHARS = 120;

/** Rasterise at this scale before OCR; below ~2x, small print is unreadable. */
const OCR_SCALE = 2;

/**
 * Where tesseract looks for its WASM core and language data.
 *
 * `tools/fetch_ocr_assets.py` puts them in public/ocr/, which keeps OCR offline
 * too. If they are absent tesseract.js falls back to its CDN, so the UI checks
 * `hasLocalOcrAssets()` and says which is about to happen rather than silently
 * reaching for the network on a page that claims to be local-only.
 */
const LOCAL_OCR_PATH = `${import.meta.env.BASE_URL}ocr/`;

export async function hasLocalOcrAssets(): Promise<boolean> {
  try {
    const res = await fetch(`${LOCAL_OCR_PATH}eng.traineddata.gz`, { method: 'HEAD' });
    return res.ok;
  } catch {
    return false;
  }
}

async function readPdfTextLayer(
  doc: PDFDocumentProxy,
  onProgress?: (p: ReadProgress) => void,
): Promise<string> {
  const parts: string[] = [];
  for (let n = 1; n <= doc.numPages; n++) {
    onProgress?.({ stage: 'reading', page: n, pages: doc.numPages, ratio: n / doc.numPages });
    const page = await doc.getPage(n);
    const content = await page.getTextContent();

    // Rebuild lines from item positions. Concatenating items in order loses the
    // row structure a lab report's whole meaning depends on - "Sodium" and
    // "139" are separate items and have to stay on one line, or the parser
    // cannot tell which analyte the number belongs to.
    const rows = new Map<number, { x: number; s: string }[]>();
    for (const item of content.items as Array<{ str: string; transform: number[] }>) {
      if (!('str' in item) || !item.str.trim()) continue;
      const y = Math.round(item.transform[5]);
      // Tolerate sub-pixel baseline jitter by bucketing to 2 units.
      const key = Math.round(y / 2) * 2;
      if (!rows.has(key)) rows.set(key, []);
      rows.get(key)!.push({ x: item.transform[4], s: item.str });
    }
    const lines = [...rows.entries()]
      .sort((a, b) => b[0] - a[0]) // PDF y grows upward, so descending is reading order
      .map(([, items]) =>
        items.sort((a, b) => a.x - b.x).map((i) => i.s).join(' ').replace(/\s+/g, ' ').trim(),
      )
      .filter(Boolean);
    parts.push(lines.join('\n'));
  }
  return parts.join('\n\n');
}

async function renderPageToCanvas(
  doc: PDFDocumentProxy,
  pageNumber: number,
): Promise<HTMLCanvasElement> {
  const page = await doc.getPage(pageNumber);
  const viewport = page.getViewport({ scale: OCR_SCALE });
  const canvas = document.createElement('canvas');
  canvas.width = Math.ceil(viewport.width);
  canvas.height = Math.ceil(viewport.height);
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Could not get a 2D canvas context for rendering.');
  await page.render({ canvasContext: context, viewport }).promise;
  return canvas;
}

async function recognise(
  images: (HTMLCanvasElement | File)[],
  onProgress?: (p: ReadProgress) => void,
): Promise<{ text: string; confidence: number }> {
  const { createWorker } = await import('tesseract.js');
  const local = await hasLocalOcrAssets();
  const worker = await createWorker('eng', 1, {
    ...(local
      ? { langPath: LOCAL_OCR_PATH, corePath: LOCAL_OCR_PATH, cacheMethod: 'none' as const }
      : {}),
    logger: (m: { status: string; progress: number }) => {
      if (m.status === 'recognizing text') {
        onProgress?.({ stage: 'recognising', ratio: m.progress });
      }
    },
  });

  try {
    const texts: string[] = [];
    let confidenceSum = 0;
    for (let i = 0; i < images.length; i++) {
      onProgress?.({ stage: 'recognising', page: i + 1, pages: images.length });
      const { data } = await worker.recognize(images[i]);
      texts.push(data.text);
      confidenceSum += data.confidence;
    }
    return {
      text: texts.join('\n\n'),
      confidence: images.length ? confidenceSum / images.length / 100 : 0,
    };
  } finally {
    await worker.terminate();
  }
}

/** Read one report. Decides for itself whether OCR is needed. */
export async function readDocument(
  file: File,
  onProgress?: (p: ReadProgress) => void,
): Promise<ReadResult> {
  const isPdf =
    file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');

  if (!isPdf) {
    const { text, confidence } = await recognise([file], onProgress);
    onProgress?.({ stage: 'done' });
    return { text, method: 'ocr', pages: 1, confidence };
  }

  const buffer = await file.arrayBuffer();
  const doc = await pdfjs.getDocument({ data: new Uint8Array(buffer) }).promise;

  try {
    const layer = await readPdfTextLayer(doc, onProgress);
    if (layer.replace(/\s/g, '').length >= MIN_TEXT_LAYER_CHARS) {
      onProgress?.({ stage: 'done' });
      return { text: layer, method: 'pdf-text', pages: doc.numPages };
    }

    // A scanned PDF: pages are images with no text behind them.
    const canvases: HTMLCanvasElement[] = [];
    for (let n = 1; n <= doc.numPages; n++) {
      onProgress?.({ stage: 'rendering', page: n, pages: doc.numPages, ratio: n / doc.numPages });
      canvases.push(await renderPageToCanvas(doc, n));
    }
    const { text, confidence } = await recognise(canvases, onProgress);
    onProgress?.({ stage: 'done' });
    return { text, method: 'ocr', pages: doc.numPages, confidence };
  } finally {
    await doc.destroy();
  }
}

export const ACCEPTED_TYPES = '.pdf,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff';
