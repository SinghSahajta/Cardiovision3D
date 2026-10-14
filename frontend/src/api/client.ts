/** Typed wrappers around the FastAPI backend. */

import type {
  AnatomyManifest,
  CohortIndex,
  CohortPatient,
  FeatureSchema,
  HealthResponse,
  MetricsResponse,
  PredictionResponse,
  PresetPatient,
} from './types';

export const API_BASE = (
  import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'
).replace(/\/$/, '');

export class ApiError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { signal });
  } catch (cause) {
    throw new ApiError(
      `Cannot reach the API at ${API_BASE}. Start it with: ` +
        `python -m uvicorn backend.app.main:app --port 8000`,
    );
  }
  if (!res.ok) {
    throw new ApiError(await describe(res, path), res.status);
  }
  return res.json() as Promise<T>;
}

async function describe(res: Response, path: string): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: string };
    if (body.detail) return body.detail;
  } catch {
    /* non-JSON error body */
  }
  return `${path} failed with HTTP ${res.status}`;
}

export const getHealth = (signal?: AbortSignal) =>
  get<HealthResponse>('/health', signal);

export const getMetrics = (signal?: AbortSignal) =>
  get<MetricsResponse>('/metrics', signal);

export const getPresets = (signal?: AbortSignal) =>
  get<PresetPatient[]>('/presets', signal);

/** All 303 training-cohort patients, with their held-out probabilities. */
export const getCohort = (signal?: AbortSignal) =>
  get<CohortIndex>('/cohort', signal);

/** One cohort patient by id ("ROW006") or 1-based index ("6"). */
export const getCohortPatient = (id: string, signal?: AbortSignal) =>
  get<CohortPatient>(`/cohort/${encodeURIComponent(id)}`, signal);

export const getSchema = (signal?: AbortSignal) =>
  get<FeatureSchema>('/schema', signal);

/** Vessel centrelines built from BodyParts3D meshes; served as a static asset. */
export async function getAnatomy(signal?: AbortSignal): Promise<AnatomyManifest> {
  const res = await fetch(`${import.meta.env.BASE_URL}models/vessels.json`, { signal });
  if (!res.ok) {
    throw new ApiError(
      'models/vessels.json missing. Run: python tools/build_anatomy.py',
      res.status,
    );
  }
  return res.json() as Promise<AnatomyManifest>;
}

export async function predict(
  features: Record<string, string | number | null>,
  topK = 12,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features, top_k: topK }),
      signal,
    });
  } catch {
    throw new ApiError(`Cannot reach the API at ${API_BASE}.`);
  }
  if (!res.ok) throw new ApiError(await describe(res, '/predict'), res.status);
  return res.json() as Promise<PredictionResponse>;
}
