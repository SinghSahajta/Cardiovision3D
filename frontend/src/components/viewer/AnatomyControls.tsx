interface Finding {
  feature: string;
  label: string;
  value: string;
}

interface Props {
  xray: boolean;
  onXray: (v: boolean) => void;
  layers: { valves: boolean; conduction: boolean };
  onLayer: (key: 'valves' | 'conduction', v: boolean) => void;
  showLabels: boolean;
  onLabels: (v: boolean) => void;
  /** Features of this patient that point at a structure in the model. */
  findings: Finding[];
  activeFinding: string | null;
  onFinding: (feature: string | null) => void;
}

const panel: React.CSSProperties = {
  background: 'color-mix(in srgb, var(--surface) 92%, transparent)',
  borderColor: 'var(--hairline)',
  color: 'var(--ink-secondary)',
};

function Toggle({
  on,
  onClick,
  children,
  title,
}: {
  on: boolean;
  onClick: () => void;
  children: React.ReactNode;
  title?: string;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      aria-pressed={on}
      className="rounded px-1.5 py-0.5 text-[10.5px] font-medium transition-colors"
      style={{
        background: on ? 'var(--ink)' : 'transparent',
        color: on ? 'var(--surface)' : 'var(--ink-secondary)',
        border: '1px solid var(--hairline)',
      }}
    >
      {children}
    </button>
  );
}

/**
 * The viewer's own controls: what is drawn, how solid it is, and which clinical
 * finding is being pointed at.
 *
 * "Findings" are the features of this patient that map onto a structure the
 * model actually contains - a bundle branch block onto its bundle, for instance.
 * They are offered rather than applied, because a highlight that appears on its
 * own would look like the model asserting a location it never predicted.
 */
export function AnatomyControls({
  xray,
  onXray,
  layers,
  onLayer,
  showLabels,
  onLabels,
  findings,
  activeFinding,
  onFinding,
}: Props) {
  return (
    <div
      className="pointer-events-auto w-[11.5rem] rounded-lg border p-2 text-[11px]"
      style={panel}
    >
      <div className="mb-1 flex items-center gap-1">
        <Toggle on={!xray} onClick={() => onXray(false)} title="Opaque tissue">
          Solid
        </Toggle>
        <Toggle
          on={xray}
          onClick={() => onXray(true)}
          title="Fade muscle so every vessel shows through"
        >
          X-ray
        </Toggle>
        <Toggle on={showLabels} onClick={() => onLabels(!showLabels)}>
          Labels
        </Toggle>
      </div>

      <div className="flex items-center gap-1">
        <Toggle on={layers.valves} onClick={() => onLayer('valves', !layers.valves)}>
          Valves
        </Toggle>
        <Toggle
          on={layers.conduction}
          onClick={() => onLayer('conduction', !layers.conduction)}
        >
          Conduction
        </Toggle>
      </div>

      {findings.length > 0 && (
        <div className="mt-2 border-t pt-1.5" style={{ borderColor: 'var(--hairline)' }}>
          <div className="mb-1 text-[9.5px] tracking-wide uppercase">
            Show the structure for
          </div>
          <div className="flex flex-wrap gap-1">
            {findings.map((f) => {
              const on = activeFinding === f.feature;
              return (
                <button
                  key={f.feature}
                  onClick={() => onFinding(on ? null : f.feature)}
                  aria-pressed={on}
                  className="rounded px-1.5 py-0.5 text-[10px] font-medium"
                  style={{
                    background: on ? '#3987e5' : 'transparent',
                    color: on ? '#ffffff' : 'var(--ink-secondary)',
                    border: `1px solid ${on ? '#3987e5' : 'var(--hairline)'}`,
                  }}
                >
                  {f.label}
                  <span className="ml-1 opacity-70">{f.value}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
