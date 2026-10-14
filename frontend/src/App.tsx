import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  ApiError,
  getAnatomy,
  getCohort,
  getCohortPatient,
  getMetrics,
  getPresets,
  getSchema,
  predict,
} from './api/client';
import type {
  AnatomyManifest,
  CohortIndex,
  CohortPatient,
  FeatureSchema,
  MetricsResponse,
  PredictionResponse,
  PresetPatient,
  VesselKey,
} from './api/types';
import { DisclaimerBanner } from './components/common/DisclaimerBanner';
import { CadSummary } from './components/dashboard/CadSummary';
import { ModelCard } from './components/dashboard/ModelCard';
import { PatientPanel } from './components/dashboard/PatientPanel';
import { PhysiologyPanel } from './components/dashboard/PhysiologyPanel';
import { ShapBars } from './components/dashboard/ShapBars';
import { ShapWaterfall } from './components/dashboard/ShapWaterfall';
import { VesselCards } from './components/dashboard/VesselCards';
import { CohortBrowser } from './components/intake/CohortBrowser';
import { ExtractionReview } from './components/intake/ExtractionReview';
import { RequiredQuestions } from './components/intake/RequiredQuestions';
import { UploadPanel, type ExtractionBatch } from './components/intake/UploadPanel';
import { Sidebar, type NavSection } from './components/shell/Sidebar';
import { StatStrip } from './components/shell/StatStrip';
import { STAT_ICONS } from './components/shell/statIcons';
import { ThemeSwitch } from './components/shell/ThemeSwitch';
import { useScrollSpy } from './components/shell/useScrollSpy';
import { HeartViewer } from './components/viewer/HeartViewer';
import { coverageOf } from './lib/completeness';
import { VESSEL_FULL_NAME, VESSEL_TERRITORY, pct } from './lib/risk';

type Selection = VesselKey | 'CAD';
type Features = Record<string, string | number | null>;
type Tab = 'upload' | 'cohort' | 'manual';

export default function App() {
  const [presets, setPresets] = useState<PresetPatient[]>([]);
  const [schema, setSchema] = useState<FeatureSchema | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [anatomy, setAnatomy] = useState<AnatomyManifest | null>(null);
  const [anatomyError, setAnatomyError] = useState<string | null>(null);
  const [cohort, setCohort] = useState<CohortIndex | null>(null);

  const [activePresetId, setActivePresetId] = useState<string | null>(null);
  const [cohortPatient, setCohortPatient] = useState<CohortPatient | null>(null);
  const [features, setFeatures] = useState<Features>({});
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [selected, setSelected] = useState<Selection>('CAD');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [tab, setTab] = useState<Tab>('cohort');
  const [batch, setBatch] = useState<ExtractionBatch | null>(null);

  const inflight = useRef<AbortController | null>(null);
  const debounce = useRef<number | null>(null);

  // --- initial load ---------------------------------------------------------
  useEffect(() => {
    const ac = new AbortController();
    (async () => {
      try {
        const [p, s, m, c] = await Promise.all([
          getPresets(ac.signal),
          getSchema(ac.signal),
          getMetrics(ac.signal).catch(() => null),
          getCohort(ac.signal).catch(() => null),
        ]);
        setPresets(p);
        setSchema(s);
        setMetrics(m);
        setCohort(c);
        if (p.length) {
          setActivePresetId(p[0].patient_id);
          setFeatures(p[0].features);
        }
      } catch (e) {
        if (!ac.signal.aborted) {
          setError(e instanceof ApiError ? e.message : String(e));
        }
      }
      try {
        setAnatomy(await getAnatomy(ac.signal));
      } catch (e) {
        if (!ac.signal.aborted) {
          setAnatomyError(e instanceof ApiError ? e.message : String(e));
        }
      }
    })();
    return () => ac.abort();
  }, []);

  const coverage = useMemo(() => coverageOf(features), [features]);

  // --- prediction, debounced so dragging a number field is not a request storm
  const run = useCallback((f: Features) => {
    if (!Object.keys(f).length) return;
    inflight.current?.abort();
    const ac = new AbortController();
    inflight.current = ac;
    setBusy(true);
    predict(f, 12, ac.signal)
      .then((res) => {
        setPrediction(res);
        setError(null);
      })
      .catch((e) => {
        if (!ac.signal.aborted) setError(e instanceof ApiError ? e.message : String(e));
      })
      .finally(() => {
        if (!ac.signal.aborted) setBusy(false);
      });
  }, []);

  useEffect(() => {
    if (!Object.keys(features).length) return;
    // The blocking field carries 15.2% of the model's weight on its own. Running
    // without it would mean rendering a confident percentage built on the
    // cohort's most common answer standing in for this patient's.
    if (coverage.blocked) {
      // Cleared here rather than derived at render on purpose. Deriving
      // (`coverage.blocked ? null : prediction`) would leave the previous
      // patient's percentage in state, so answering the blocking question
      // would flash a stale risk figure for the length of the debounce before
      // the new one lands. A wrong number on screen for 220ms is not
      // acceptable in a clinical readout, so the cascading render is the
      // cheaper trade.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setPrediction(null);
      return;
    }
    if (debounce.current) window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => run(features), 220);
    return () => {
      if (debounce.current) window.clearTimeout(debounce.current);
    };
  }, [features, coverage.blocked, run]);

  const activePreset = useMemo(
    () => presets.find((p) => p.patient_id === activePresetId) ?? null,
    [presets, activePresetId],
  );

  const handlePreset = (id: string) => {
    const found = presets.find((p) => p.patient_id === id);
    if (!found) return;
    setActivePresetId(id);
    setCohortPatient(null);
    setFeatures(found.features);
  };

  const loadCohortPatient = useCallback(async (id: string) => {
    setBusy(true);
    try {
      const patient = await getCohortPatient(id);
      setCohortPatient(patient);
      setActivePresetId(null);
      setFeatures(patient.features);
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  /** Editing anything makes this a custom patient, not a cohort row any more. */
  const editFeature = useCallback((feature: string, value: string | number | null) => {
    setFeatures((prev) => ({ ...prev, [feature]: value }));
    setCohortPatient(null);
    setActivePresetId(null);
  }, []);

  const node = prediction
    ? selected === 'CAD'
      ? prediction.cad
      : prediction.vessels[selected]
    : null;

  const groundTruth = cohortPatient?.ground_truth ?? activePreset?.ground_truth ?? null;

  // --- shell ----------------------------------------------------------------

  const explained = Boolean(prediction && node);

  const sections = useMemo<NavSection[]>(
    () => [
      { id: 'overview', label: 'Overview', icon: 'heart', available: true },
      { id: 'explanation', label: 'Why this', icon: 'explain', available: explained },
      { id: 'intake', label: 'Patient data', icon: 'intake', available: true },
      { id: 'model', label: 'Model card', icon: 'model', available: true },
    ],
    [explained],
  );

  // Only sections that exist in the DOM can be observed.
  const spyIds = useMemo(
    () => sections.filter((s) => s.available).map((s) => s.id),
    [sections],
  );
  const activeSection = useScrollSpy(spyIds);

  const identity = cohortPatient
    ? {
        name: `Cohort patient ${cohortPatient.index}`,
        detail: `of ${cohort?.n_patients ?? 303} · in-sample`,
        initial: String(cohortPatient.index),
      }
    : activePreset
      ? { name: activePreset.label, detail: 'Demo case', initial: 'D' }
      : { name: 'Custom patient', detail: 'Entered by hand', initial: '—' };

  // The API returns "0.9136 ± 0.0537" - the mean and SD across CV folds. The
  // strip has room for the figure, not the interval, so the interval moves to
  // the tooltip rather than being dropped.
  const cadAucRaw = metrics?.per_target.CAD.threshold_independent.roc_auc ?? null;
  const cadAuc = cadAucRaw ? Number(cadAucRaw.split(' ')[0]).toFixed(3) : null;

  return (
    // The backdrop is a SIBLING of the shell, not a child of it.
    //
    // As a child it painted over the page: a `position: fixed` element paints in
    // the positioned-descendants layer, above static content like the
    // disclaimer. The cards survived only because .canvas carries a
    // backdrop-filter, which quietly creates its own stacking context - so the
    // notice vanished while everything inside the canvas stayed put, which is a
    // confusing way for a required safety notice to disappear.
    //
    // Two explicit layers, no negative z-index, nothing depending on which
    // elements happen to establish a context.
    <>
      <div className="app-backdrop" aria-hidden="true" />

      <div className="relative z-[1] flex min-h-full">
      <Sidebar
        sections={sections}
        active={activeSection}
        patientName={identity.name}
        patientDetail={identity.detail}
        patientInitial={identity.initial}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        {/* Below lg the rail is replaced by a compact bar, so the page stays a
            single column on a phone. */}
        <header
          className="sticky top-0 z-30 flex items-center justify-between gap-3 px-4 py-2.5 lg:hidden"
          style={{
            background: 'color-mix(in srgb, var(--page) 90%, transparent)',
            backdropFilter: 'blur(8px)',
          }}
        >
          <div className="flex min-w-0 items-center gap-2">
            <span
              className="grid h-7 w-7 shrink-0 place-items-center rounded-lg"
              style={{ background: 'var(--accent)' }}
              aria-hidden="true"
            >
              <svg width="15" height="15" viewBox="0 0 16 16" fill="var(--on-accent)">
                <path d="M8 14.2S1.9 10.4 1.9 6.3A3.6 3.6 0 0 1 8 4.5a3.6 3.6 0 0 1 6.1 1.8c0 4.1-6.1 7.9-6.1 7.9Z" />
              </svg>
            </span>
            <span className="min-w-0">
              <span className="block truncate text-[13px] font-semibold tracking-tight">
                Coronary Risk Mapping
              </span>
              <span
                className="block truncate text-[11px]"
                style={{ color: 'var(--ink-secondary)' }}
              >
                {identity.name}
              </span>
            </span>
          </div>
          <ThemeSwitch />
        </header>

        <main className="px-3 pt-2 pb-6 sm:px-4 sm:pb-8 lg:pt-4 lg:pr-5 lg:pl-0">
          <div className="mx-auto max-w-[1420px]">
            <DisclaimerBanner />

            {error && (
              <div
                className="mb-2.5 rounded-[var(--radius)] px-3.5 py-2 text-[12px]"
                style={{
                  background: 'color-mix(in srgb, var(--risk-high) 12%, var(--surface))',
                  color: 'var(--ink)',
                }}
              >
                {error}
              </div>
            )}

            <div className="canvas p-3.5 sm:p-5 lg:p-6">
            <StatStrip
              stats={[
                {
                  tint: 'tint-a',
                  label: cohortPatient ? 'Research cohort' : activePreset ? 'Demo case' : 'Patient',
                  value: cohortPatient ? String(cohortPatient.index) : identity.name,
                  suffix: cohortPatient ? `of ${cohort?.n_patients ?? 303}` : undefined,
                  compact: !cohortPatient,
                  icon: STAT_ICONS.people,
                  title: cohortPatient
                    ? 'The served model was trained on this row, so its live output is in-sample.'
                    : identity.name,
                },
                {
                  tint: 'tint-b',
                  label: 'Input coverage',
                  value: pct(coverage.weight),
                  suffix: 'of model weight',
                  icon: STAT_ICONS.gauge,
                  title:
                    'Share of the model’s total weight that is answered — not the share of fields filled in.',
                },
                {
                  tint: 'tint-c',
                  label: 'Held-out CAD ROC-AUC',
                  value: cadAuc ?? '—',
                  suffix: 'out-of-fold',
                  icon: STAT_ICONS.chip,
                  title: cadAucRaw
                    ? `${cadAucRaw} across nested cross-validation folds. Never the in-sample figure.`
                    : undefined,
                },
              ]}
            />

            {/* --- the result, with the anatomy beside it --------------------- */}
            <section
              id="overview"
              className="mt-3.5 grid scroll-mt-20 gap-3.5 sm:mt-5 sm:gap-5 lg:scroll-mt-5 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]"
            >
              {/* The heart renders onto a transparent canvas: --surface is the
                  ground it sits against and the tissue colours are fixed hex
                  that cannot adapt. .viewer-pin holds that token steady for
                  this subtree. No file under components/viewer/ is modified. */}
              <div className="viewer-pin min-w-0">
                <HeartViewer
                  anatomy={anatomy}
                  prediction={prediction}
                  features={features}
                  selected={selected}
                  onSelect={setSelected}
                  anatomyError={anatomyError}
                />
              </div>

              <div className="flex min-w-0 flex-col gap-3.5 sm:gap-4">
                {coverage.blocked ? (
                  <div className="card flex flex-1 flex-col justify-center p-6">
                    <p className="eyebrow">No prediction yet</p>
                    <h2 className="mt-2 text-xl font-semibold tracking-tight">
                      One answer is still missing
                    </h2>
                    <p
                      className="mt-2 max-w-md text-[13px] leading-relaxed"
                      style={{ color: 'var(--ink-secondary)' }}
                    >
                      Typical anginal chest pain carries <strong>15.2%</strong> of this
                      model&rsquo;s weight &mdash; more than twice any other input, and more
                      than the entire lipid panel combined. Answer it below and the
                      prediction appears.
                    </p>
                    <p className="mt-3 text-[11.5px]" style={{ color: 'var(--ink-muted)' }}>
                      {pct(coverage.weight)} of the model&rsquo;s weight is covered so far.
                    </p>
                  </div>
                ) : prediction ? (
                  <>
                    <CadSummary
                      cad={prediction.cad}
                      heldOut={cohortPatient?.held_out.CAD ?? null}
                      groundTruth={groundTruth?.CAD ?? null}
                      coverage={coverage.weight}
                    />
                    <VesselCards
                      prediction={prediction}
                      selected={selected}
                      onSelect={setSelected}
                      groundTruth={groundTruth}
                      heldOut={cohortPatient?.held_out ?? null}
                    />
                  </>
                ) : (
                  <div className="card flex flex-1 items-center justify-center p-6">
                    <p className="text-sm" style={{ color: 'var(--ink-secondary)' }}>
                      {error ? 'Cannot load predictions.' : 'Loading model output…'}
                    </p>
                  </div>
                )}
              </div>
            </section>

            {/* --- one wide explanation panel --------------------------------- */}
            {prediction && node && (
              <section
                id="explanation"
                className="card mt-3.5 scroll-mt-20 p-4 sm:mt-5 sm:p-6 lg:scroll-mt-5"
              >
                <header className="mb-5 flex flex-wrap items-baseline justify-between gap-3">
                  <div>
                    <h2 className="text-lg font-semibold tracking-tight">
                      Why this prediction
                    </h2>
                    <p
                      className="mt-0.5 text-[12px]"
                      style={{ color: 'var(--ink-secondary)' }}
                    >
                      {selected === 'CAD' ? (
                        'Overall coronary artery disease'
                      ) : (
                        <>
                          <strong style={{ color: 'var(--ink)' }}>
                            {VESSEL_FULL_NAME[selected]}
                          </strong>{' '}
                          &mdash; {VESSEL_TERRITORY[selected]}
                        </>
                      )}
                    </p>
                  </div>
                  <div
                    className="flex gap-1 rounded-[var(--radius-pill)] p-1"
                    style={{ background: 'var(--surface-sunken)' }}
                  >
                    {(['CAD', 'LAD', 'LCX', 'RCA'] as Selection[]).map((k) => (
                      <button
                        key={k}
                        onClick={() => setSelected(k)}
                        aria-pressed={selected === k}
                        className="pill px-3 py-1 text-[11.5px] font-medium"
                      >
                        {k}
                      </button>
                    ))}
                  </div>
                </header>

                <div className="grid gap-x-8 gap-y-7 xl:grid-cols-2">
                  <ShapBars
                    attributions={node.top_features}
                    baseValue={node.shap_base_value}
                  />
                  <div>
                    <h3 className="eyebrow mb-2">From cohort baseline to this patient</h3>
                    <ShapWaterfall
                      attributions={node.top_features}
                      baseValue={node.shap_base_value}
                      shapSum={node.shap_sum}
                    />
                  </div>
                </div>

                <div
                  className="mt-7 border-t pt-6"
                  style={{ borderColor: 'var(--hairline)' }}
                >
                  <h3 className="eyebrow mb-2">
                    Physiological measurements &amp; contribution
                  </h3>
                  <PhysiologyPanel attributions={node.top_features} />
                </div>
              </section>
            )}

            {/* --- intake ------------------------------------------------------ */}
            <section
              id="intake"
              className="mt-3.5 grid scroll-mt-20 gap-3.5 sm:mt-5 sm:gap-5 lg:scroll-mt-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]"
            >
              <div className="card min-w-0 p-4 sm:p-5">
                <div
                  className="mb-4 inline-flex gap-1 rounded-[var(--radius-pill)] p-1"
                  style={{ background: 'var(--surface-sunken)' }}
                >
                  {(
                    [
                      ['cohort', 'Research cohort'],
                      ['upload', 'Upload a report'],
                      ['manual', 'All inputs'],
                    ] as [Tab, string][]
                  ).map(([key, label]) => {
                    const on = tab === key;
                    return (
                      <button
                        key={key}
                        onClick={() => setTab(key)}
                        aria-pressed={on}
                        className="pill px-3 py-1.5 text-[12px] font-medium"
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>

                {tab === 'cohort' &&
                  (cohort ? (
                    <CohortBrowser
                      index={cohort}
                      activeId={cohortPatient?.patient_id ?? null}
                      onSelect={loadCohortPatient}
                      busy={busy}
                    />
                  ) : (
                    <p className="text-[12px]" style={{ color: 'var(--ink-secondary)' }}>
                      Cohort not exported yet. Run{' '}
                      <code>python tools/build_cohort.py</code>.
                    </p>
                  ))}

                {tab === 'upload' &&
                  (batch ? (
                    <ExtractionReview
                      batch={batch}
                      onApply={(values) => {
                        // Replace, do not merge. A report describes one patient, and
                        // merging it into whoever was loaded before would silently
                        // combine this document's labs with that patient's symptoms
                        // and history - a chimera the prediction would be computed
                        // on without anything saying so. Starting clean also makes
                        // the missing 33% visible, which is the honest state after
                        // an upload.
                        setFeatures(values);
                        setCohortPatient(null);
                        setActivePresetId(null);
                        setBatch(null);
                        setTab('manual');
                      }}
                      onDismiss={() => setBatch(null)}
                    />
                  ) : (
                    <UploadPanel onExtracted={setBatch} />
                  ))}

                {tab === 'manual' && (
                  <PatientPanel
                    schema={schema}
                    presets={presets}
                    activePresetId={activePresetId}
                    features={features}
                    onPreset={handlePreset}
                    onFeatureChange={editFeature}
                    onReset={() => activePreset && setFeatures(activePreset.features)}
                    busy={busy}
                  />
                )}
              </div>

              <div className="card min-w-0 p-4 sm:p-5">
                <RequiredQuestions features={features} onChange={editFeature} />
              </div>
            </section>

            <section id="model" className="mt-3.5 scroll-mt-20 sm:mt-5 lg:scroll-mt-5">
              <ModelCard metrics={metrics} />
            </section>

            <p
              className="mt-5 max-w-3xl text-[11px] leading-relaxed"
              style={{ color: 'var(--ink-muted)' }}
            >
              Anatomical meshes: BodyParts3D, &copy; The Database Center for Life Science,
              licensed under CC Attribution 4.0 International. Every structure shown is that
              project&rsquo;s own mesh geometry. It is a generic anatomical model &mdash; not
              this patient&rsquo;s coronary tree &mdash; and it is left-dominant, so its
              posterior descending artery is grouped with the LCX.
              </p>
            </div>
          </div>
        </main>
      </div>
      </div>
    </>
  );
}
