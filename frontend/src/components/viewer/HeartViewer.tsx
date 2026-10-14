import { OrbitControls } from '@react-three/drei';
import { Canvas, useThree } from '@react-three/fiber';
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';

import type { AnatomyManifest, PredictionResponse, VesselKey } from '../../api/types';
import { VESSELS } from '../../api/types';
import { featureLink, LINKABLE_FEATURES } from '../../lib/anatomy';
import { colorOf, pct } from '../../lib/risk';
import { AnatomyControls } from './AnatomyControls';
import { HeartShell } from './HeartShell';
import { Legend } from './Legend';
import { useAnatomyGeometry } from './useAnatomyGeometry';
import { useMorphInfluence } from './useMorphInfluence';
import { VesselLabels } from './VesselLabels';
import type { LabelAnchor, ProjectedLabel } from './VesselLabels';
import { VesselMesh } from './VesselMesh';

interface Props {
  anatomy: AnatomyManifest | null;
  prediction: PredictionResponse | null;
  /** The exact payload sent to /predict, so findings match the prediction. */
  features: Record<string, string | number | null>;
  selected: VesselKey | 'CAD';
  onSelect: (key: VesselKey | 'CAD') => void;
  anatomyError?: string | null;
}

const SHELL_URL = `${import.meta.env.BASE_URL}models/anatomy.glb`;

const SHORT_LABEL: Record<string, string> = {
  'EF-TTE': 'Low EF',
  'Region RWMA': 'Wall motion',
  LVH: 'LVH',
  BBB: 'Bundle block',
  VHD: 'Valve disease',
  'Systolic Murmur': 'Syst. murmur',
  'Diastolic Murmur': 'Diast. murmur',
};

/**
 * Asks for a few frames whenever the scene's content changes.
 *
 * `frameloop="demand"` only renders when something calls `invalidate()`, and the
 * cost of a missed call is a blank canvas with the model loaded, positioned and
 * lit behind it. R3F does invalidate on its own commits, so this is insurance
 * rather than a fix for a known fault - and worth noting that the blank canvas
 * seen while building this was neither: a backgrounded tab gets no
 * requestAnimationFrame at all, so nothing renders however many are requested.
 *
 * A short burst rather than one frame covers the case where the first lands
 * before the GPU has the buffers. It sits outside <Suspense> on purpose, so it
 * is mounted and listening while the geometry is still loading.
 */
function RenderOnChange({ signal }: { signal: string }) {
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    let frame = 0;
    let id = 0;
    const tick = () => {
      invalidate();
      if (++frame < 6) id = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(id);
  }, [signal, invalidate]);
  return null;
}

/**
 * The left coronary stem: proximal to the LAD/LCX bifurcation, so it belongs to
 * neither territory and carries no prediction. Drawn in neutral tissue colour.
 *
 * It still goes through the morph hook. The build gives every artery a morph
 * target, and a mesh that has one but never calls updateMorphTargets() makes the
 * renderer throw mid-frame, which kills every draw queued behind it.
 */
function LeftMain({ geometry }: { geometry: THREE.BufferGeometry }) {
  const mesh = useRef<THREE.Mesh>(null);
  useMorphInfluence(mesh, 0, geometry);
  return (
    <mesh ref={mesh} geometry={geometry}>
      <meshStandardMaterial
        color="#9a6a6a"
        roughness={0.5}
        metalness={0.04}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/** Loads the GLB and draws it. Split out so <Suspense> has something to catch. */
function Scene({
  manifest,
  prediction,
  selected,
  onSelect,
  layers,
  xray,
  highlighted,
}: {
  manifest: AnatomyManifest;
  prediction: PredictionResponse | null;
  selected: VesselKey | 'CAD';
  onSelect: (key: VesselKey | 'CAD') => void;
  layers: { valves: boolean; conduction: boolean };
  xray: boolean;
  highlighted: string[];
}) {
  const geometries = useAnatomyGeometry(SHELL_URL);

  return (
    <>
      <HeartShell
        manifest={manifest}
        geometries={geometries}
        layers={layers}
        xray={xray}
        highlighted={highlighted}
      />

      {geometries.artery_left_main && (
        <LeftMain geometry={geometries.artery_left_main} />
      )}

      {VESSELS.map((key: VesselKey) => {
        const geometry = geometries[manifest.vessel_nodes[key]];
        if (!geometry) return null;
        const pred = prediction?.vessels[key];
        // With no prediction yet the arteries are still drawn, just uncoloured.
        // Dropping them would leave a heart with no coronary tree on it, which
        // reads as a rendering fault rather than as "nothing predicted yet".
        return (
          <VesselMesh
            key={key}
            vesselKey={key}
            geometry={geometry}
            risk={pred ? pred.risk_score : null}
            selected={selected === key}
            dimmed={highlighted.length > 0}
            onSelect={onSelect}
          />
        );
      })}
    </>
  );
}

/**
 * The 3D panel.
 *
 * Everything drawn here is BodyParts3D mesh geometry, decimated but not
 * reconstructed - see tools/build_anatomy.py for what that replaced.
 *
 * Performance: `frameloop="demand"` means three.js renders only when something
 * actually changes - a camera move, a hover, a new prediction - instead of
 * running a 60fps loop forever. On a machine without a discrete GPU that is the
 * difference between an idle fan and a hot one, and it is why there is no beat
 * animation or auto-rotation here.
 */
export function HeartViewer({
  anatomy,
  prediction,
  features,
  selected,
  onSelect,
  anatomyError,
}: Props) {
  const [showLabels, setShowLabels] = useState(true);
  const [labels, setLabels] = useState<ProjectedLabel[]>([]);
  const [xray, setXray] = useState(false);
  const [layers, setLayers] = useState({ valves: false, conduction: false });
  const [activeFinding, setActiveFinding] = useState<string | null>(null);

  // Which of this patient's features point at a structure the model contains.
  const findings = useMemo(
    () =>
      LINKABLE_FEATURES.flatMap((feature) => {
        const link = featureLink(feature, features[feature] ?? null);
        if (!link) return [];
        return [{
          feature,
          label: SHORT_LABEL[feature] ?? feature,
          value: String(features[feature]),
          link,
        }];
      }),
    [features],
  );

  const active = findings.find((f) => f.feature === activeFinding) ?? null;
  const highlighted = active ? active.link.nodes : [];

  const handleFinding = useCallback(
    (feature: string | null) => {
      setActiveFinding(feature);
      if (!feature) return;
      const found = findings.find((f) => f.feature === feature);
      if (found && found.link.layer !== 'core') {
        // Switching on the finding has to switch on the layer that holds it,
        // otherwise the highlight lands on something that is not drawn - and
        // the valves and the conduction system sit INSIDE the heart, so with
        // opaque muscle the highlight is drawn and still invisible. Asking to
        // see one is therefore also asking to see through the wall in front of
        // it. Solid remains one click away.
        setLayers((l) => ({ ...l, [found.link.layer]: true }));
        setXray(true);
      }
    },
    [findings],
  );

  const anchors = useMemo<LabelAnchor[]>(() => {
    if (!anatomy) return [];
    return VESSELS.flatMap((key) => {
      const node = anatomy.nodes[anatomy.vessel_nodes[key]];
      return node ? [{ key, point: node.anchor }] : [];
    });
  }, [anatomy]);

  const handleLabels = useCallback((next: ProjectedLabel[]) => setLabels(next), []);

  return (
    <div className="card flex flex-col overflow-hidden">
      {/* The 3D stage. The key used to float over the bottom of this box and
          covered the lower third of the heart; it now sits in its own strip
          below, so nothing is drawn on top of the anatomy.

          Explicit height, NOT min-height: the Canvas is absolutely positioned
          against this box, and a percentage height resolved against a
          min-height-only parent is zero - which silently leaves the canvas at
          its 300x150 default. */}
      <div className="relative" style={{ height: 'clamp(420px, 56vh, 660px)' }}>
      <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex items-start justify-between gap-2 p-3">
        <div
          className="pointer-events-auto rounded-lg border px-2.5 py-1.5 text-[11px]"
          style={{
            background: 'color-mix(in srgb, var(--surface) 92%, transparent)',
            borderColor: 'var(--hairline)',
            color: 'var(--ink-secondary)',
          }}
        >
          <div className="font-semibold" style={{ color: 'var(--ink)' }}>
            Coronary risk map
          </div>
          <div>Drag to rotate &middot; scroll to zoom &middot; click a vessel</div>
        </div>
        <AnatomyControls
          xray={xray}
          onXray={setXray}
          layers={layers}
          onLayer={(key, v) => setLayers((l) => ({ ...l, [key]: v }))}
          showLabels={showLabels}
          onLabels={setShowLabels}
          findings={findings.map(({ feature, label, value }) => ({
            feature,
            label,
            value,
          }))}
          activeFinding={activeFinding}
          onFinding={handleFinding}
        />
      </div>

      {anatomyError && (
        <div className="absolute inset-0 z-20 flex items-center justify-center p-6">
          <div
            className="max-w-md rounded-lg border p-4 text-sm"
            style={{ borderColor: 'var(--hairline)', background: 'var(--surface-raised)' }}
          >
            <p className="mb-2 font-semibold">3D anatomy not built yet</p>
            <p style={{ color: 'var(--ink-secondary)' }}>{anatomyError}</p>
            <pre
              className="mt-3 overflow-x-auto rounded p-2 text-xs"
              style={{ background: 'var(--page)', color: 'var(--ink-secondary)' }}
            >
              python tools/build_anatomy.py
            </pre>
          </div>
        </div>
      )}

      <div className="absolute inset-0">
        <Canvas
          frameloop="demand"
          dpr={[1, 1.5]}
          // preserveDrawingBuffer lets the canvas be read back after rendering,
          // which is what makes the view screenshot-able for the demo recording
          // and lets automated checks assert the vessels actually painted.
          gl={{ antialias: true, powerPreference: 'default', preserveDrawingBuffer: true }}
          camera={{ position: [0.12, 0.0, 2.55], fov: 42, near: 0.01, far: 50 }}
        >
          {/* Fixed three-point lighting. No environment map or postprocessing:
              both are expensive and neither adds clinical information. */}
          <ambientLight intensity={0.7} />
          <directionalLight position={[3, 4, 5]} intensity={1.25} />
          <directionalLight position={[-4, -1, -3]} intensity={0.45} />

          <Suspense fallback={null}>
            {anatomy && !anatomyError && (
              <Scene
                manifest={anatomy}
                prediction={prediction}
                selected={selected}
                onSelect={onSelect}
                layers={layers}
                xray={xray}
                highlighted={highlighted}
              />
            )}
          </Suspense>

          <RenderOnChange
            signal={[
              anatomy ? 'a' : '-',
              prediction ? 'p' : '-',
              xray ? 'x' : '-',
              layers.valves ? 'v' : '-',
              layers.conduction ? 'c' : '-',
              highlighted.join(','),
              selected,
            ].join('|')}
          />

          <VesselLabels anchors={anchors} onChange={handleLabels} />

          {/* No damping. Inertial damping needs a frame every tick to decay, so
              with frameloop="demand" it invalidates continuously and the render
              loop never settles - which pins a core on exactly the machines this
              view is supposed to stay responsive on. */}
          <OrbitControls
            makeDefault
            enablePan
            enableZoom
            enableDamping={false}
            minDistance={0.9}
            maxDistance={6}
            // Centred. This used to pivot to [0, -0.12, 0] to push the model
            // into the upper two thirds, clear of the legend that overlapped
            // the bottom of the frame. The legend now has its own strip below
            // the stage, so the heart gets the whole box back.
          />
        </Canvas>
      </div>

      {showLabels &&
        prediction &&
        labels.map(({ key, x, y, visible, facing }) => {
          const pred = prediction.vessels[key];
          if (!pred || !visible) return null;
          const color = colorOf(pred.risk_score);
          return (
            <button
              key={key}
              onClick={() => onSelect(key)}
              className="tnum absolute z-10 -translate-x-1/2 -translate-y-1/2 rounded border px-1 py-px text-[10px] leading-tight font-semibold whitespace-nowrap"
              style={{
                left: x,
                top: y,
                background: 'color-mix(in srgb, var(--surface) 90%, transparent)',
                borderColor: selected === key ? color : 'var(--hairline)',
                color: 'var(--ink)',
                // Solid muscle really does hide the vessel behind it, so a label
                // for a vessel that has rotated round the back is faded rather
                // than removed: it still says where to rotate to, without
                // claiming you can see what it points at.
                opacity: facing || xray ? 1 : 0.4,
              }}
              title={anatomy?.nodes[anatomy.vessel_nodes[key]]?.anatomical_name}
            >
              <span
                className="mr-1 inline-block h-1.5 w-1.5 rounded-full align-middle"
                style={{ background: color }}
                aria-hidden
              />
              {key} {pct(pred.risk_score)}
              {pred.is_stenotic && <span className="ml-0.5 font-bold">!</span>}
            </button>
          );
        })}
      </div>

      {/* The key, below the stage rather than over it. */}
      <div className="p-3" style={{ borderTop: '1px solid var(--hairline)' }}>
        {active && (
          <p
            className="mb-2 max-w-xl rounded-lg border px-2.5 py-1.5 text-[11px] leading-snug"
            style={{
              background: 'color-mix(in srgb, var(--surface) 94%, transparent)',
              borderColor: '#3987e5',
              color: 'var(--ink-secondary)',
            }}
          >
            {active.link.note}
          </p>
        )}
        <div className="flex flex-wrap items-end justify-between gap-2">
          <Legend />
          {/* The disclaimer sits inside the viewer frame as well as in the page
              banner, so a screenshot or demo recording of the 3D view alone still
              carries it. */}
          <p
            className="max-w-[13rem] text-right text-[10px] leading-snug"
            style={{ color: 'var(--ink-muted)' }}
          >
            Decision support only &mdash; not a diagnostic image. Anatomy is a
            generic BodyParts3D model, not this patient&rsquo;s own.
          </p>
        </div>
      </div>
    </div>
  );
}
