import { useCallback, useEffect, useRef, useState } from 'react';

import { extractFields, type Candidate } from '../../lib/extract/parse';
import {
  ACCEPTED_TYPES,
  hasLocalOcrAssets,
  readDocument,
  type ReadMethod,
  type ReadProgress,
} from '../../lib/extract/readDocument';

export interface ExtractionBatch {
  candidates: Candidate[];
  method: ReadMethod;
  confidence?: number;
  fileName: string;
  pages: number;
}

interface Props {
  onExtracted: (batch: ExtractionBatch) => void;
}

const STAGE_LABEL: Record<ReadProgress['stage'], string> = {
  reading: 'Reading the PDF',
  rendering: 'Rendering pages for OCR',
  recognising: 'Recognising text',
  done: 'Done',
};

/**
 * Drop a lab, echo or ECG report here.
 *
 * Everything happens in this tab: the PDF is parsed and any OCR is run in the
 * page, and the file is never sent anywhere. That is worth stating on the
 * control itself rather than in a privacy policy, because it is the reason a
 * clinician might be willing to use it at all.
 *
 * It is also worth stating what this cannot do. A report supplies 31 of the
 * model's 55 inputs; the rest are symptoms, history and examination findings
 * that only a clinician records. And no OCR engine reads handwriting reliably -
 * promising otherwise would just produce confident nonsense.
 */
export function UploadPanel({ onExtracted }: Props) {
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<ReadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [localOcr, setLocalOcr] = useState<boolean | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    hasLocalOcrAssets().then(setLocalOcr).catch(() => setLocalOcr(false));
  }, []);

  const handle = useCallback(
    async (files: FileList | null) => {
      if (!files?.length) return;
      setBusy(true);
      setError(null);
      try {
        for (const file of Array.from(files)) {
          const read = await readDocument(file, setProgress);
          const { candidates } = extractFields(read.text);
          onExtracted({
            candidates,
            method: read.method,
            confidence: read.confidence,
            fileName: file.name,
            pages: read.pages,
          });
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
        setProgress(null);
        if (input.current) input.current.value = '';
      }
    },
    [onExtracted],
  );

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void handle(e.dataTransfer.files);
        }}
        onClick={() => input.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') input.current?.click();
        }}
        className="flex cursor-pointer flex-col items-center justify-center rounded-[var(--radius)] px-5 py-7 text-center transition-colors"
        style={{
          border: `1.5px dashed ${dragging ? 'var(--accent)' : 'var(--hairline-strong)'}`,
          background: dragging ? 'var(--accent-soft)' : 'var(--surface-sunken)',
        }}
      >
        <input
          ref={input}
          type="file"
          accept={ACCEPTED_TYPES}
          multiple
          className="hidden"
          onChange={(e) => void handle(e.target.files)}
        />

        {busy ? (
          <>
            <div className="text-sm font-medium">
              {progress ? STAGE_LABEL[progress.stage] : 'Working'}
              {progress?.page && progress.pages
                ? ` — page ${progress.page} of ${progress.pages}`
                : ''}
            </div>
            <div
              className="mt-3 h-1 w-48 overflow-hidden rounded-full"
              style={{ background: 'var(--grid)' }}
            >
              <div
                className="h-full rounded-full transition-[width] duration-200"
                style={{
                  width: `${Math.round((progress?.ratio ?? 0) * 100)}%`,
                  background: 'var(--accent)',
                }}
              />
            </div>
          </>
        ) : (
          <>
            <div className="text-sm font-medium" style={{ color: 'var(--ink)' }}>
              Drop a lab, echo or ECG report
            </div>
            <div className="mt-1 text-xs" style={{ color: 'var(--ink-secondary)' }}>
              PDF or image &middot; or click to choose
            </div>
          </>
        )}
      </div>

      {error && (
        <p className="mt-2 text-xs" style={{ color: 'var(--risk-high)' }}>
          {error}
        </p>
      )}

      <ul
        className="mt-3 space-y-1 text-[11.5px] leading-relaxed"
        style={{ color: 'var(--ink-muted)' }}
      >
        <li>
          <strong style={{ color: 'var(--ink-secondary)' }}>Nothing is uploaded.</strong>{' '}
          The file is read in this tab and never leaves your machine.
        </li>
        <li>
          A report can fill <strong style={{ color: 'var(--ink-secondary)' }}>31 of
          the 55 inputs</strong> &mdash; about two thirds of what the model weighs.
          The rest are symptoms and history, asked below.
        </li>
        <li>
          Digital PDFs are read exactly from their text layer. Scans and photos go
          through OCR, which makes mistakes &mdash; every value is shown for you to
          check before it is used. <strong style={{ color: 'var(--ink-secondary)' }}>
          Handwritten prescriptions will not read reliably.</strong>
        </li>
        {localOcr === false && (
          <li>
            OCR assets are not installed locally, so scanning an image will fetch
            them once from a CDN. Run{' '}
            <code style={{ color: 'var(--ink-secondary)' }}>
              python tools/fetch_ocr_assets.py
            </code>{' '}
            to keep it fully offline. PDFs with a text layer never need them.
          </li>
        )}
      </ul>
    </div>
  );
}
