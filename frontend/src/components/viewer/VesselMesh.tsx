import { useRef, useState } from 'react';
import * as THREE from 'three';

import type { VesselKey } from '../../api/types';
import { TISSUE, stenosisInfluence } from '../../lib/anatomy';
import { colorOf } from '../../lib/risk';
import { useMorphInfluence } from './useMorphInfluence';

interface Props {
  vesselKey: VesselKey;
  geometry: THREE.BufferGeometry;
  /** Calibrated probability from the API, or null before one exists. */
  risk: number | null;
  selected: boolean;
  dimmed: boolean;
  onSelect: (key: VesselKey) => void;
}

/**
 * One coronary artery: the BodyParts3D mesh itself, trunk and named branches
 * merged, with nothing reconstructed.
 *
 * Risk is encoded twice on purpose. Colour comes from the validated status
 * palette; calibre comes from morph target 0, a uniform contraction toward the
 * vessel's own medial axis baked at build time. A reader who cannot separate the
 * three status hues still sees a stenotic vessel as a thinner one, and the
 * numeric label carries the actual probability either way.
 *
 * The narrowing is uniform along the whole vessel. The dataset labels a vessel
 * stenotic or not and never says where the lesion sits, so a localised pinch
 * would be inventing a finding.
 */
export function VesselMesh({
  vesselKey,
  geometry,
  risk,
  selected,
  dimmed,
  onSelect,
}: Props) {
  const [hovered, setHovered] = useState(false);
  const mesh = useRef<THREE.Mesh>(null);

  useMorphInfluence(mesh, risk === null ? 0 : stenosisInfluence(risk), geometry);

  // Neutral tissue colour until there is something to say. Reaching for a
  // status hue here would state a risk tier the model has not produced.
  const color = risk === null ? '#9a6a6a' : colorOf(risk);

  return (
    <mesh
      ref={mesh}
      geometry={geometry}
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(true);
        document.body.style.cursor = 'pointer';
      }}
      onPointerOut={() => {
        setHovered(false);
        document.body.style.cursor = '';
      }}
      onClick={(e) => {
        e.stopPropagation();
        onSelect(vesselKey);
      }}
    >
      <meshStandardMaterial
        color={color}
        roughness={TISSUE.artery.roughness}
        metalness={0.04}
        emissive={color}
        emissiveIntensity={selected ? 0.5 : hovered ? 0.3 : 0.1}
        transparent={dimmed}
        opacity={dimmed ? 0.35 : 1}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}
