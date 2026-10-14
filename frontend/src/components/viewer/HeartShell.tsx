import * as THREE from 'three';

import type { AnatomyManifest } from '../../api/types';
import { HIGHLIGHT, TISSUE } from '../../lib/anatomy';

interface Props {
  manifest: AnatomyManifest;
  geometries: Record<string, THREE.BufferGeometry>;
  /** Layers the user has switched on, besides "core". */
  layers: { valves: boolean; conduction: boolean };
  xray: boolean;
  /** Node names currently called out by a feature link. */
  highlighted: string[];
}

/**
 * Every non-arterial structure: ventricular myocardium, atria, great vessels,
 * and the valve and conduction layers when they are switched on.
 *
 * Solid by default. The previous version drew the whole heart at opacity
 * 0.12-0.22 with depthWrite off, which is why it read as a cloud rather than
 * tissue - nothing in the scene had a surface. Here opacity is only ever
 * reduced by the explicit x-ray control, and even then only for muscle and
 * atrium; valves, bundles and arteries stay opaque so x-ray reveals them
 * instead of fading everything together.
 */
export function HeartShell({
  manifest,
  geometries,
  layers,
  xray,
  highlighted,
}: Props) {
  return (
    <group>
      {Object.entries(manifest.nodes).map(([name, node]) => {
        if (node.group === 'artery') return null; // drawn by VesselMesh
        if (node.layer === 'valves' && !layers.valves) return null;
        if (node.layer === 'conduction' && !layers.conduction) return null;

        const geometry = geometries[name];
        if (!geometry) return null;

        const style = TISSUE[node.group];
        const isHighlighted = highlighted.includes(name);
        // Muscle and atrium are the bulk that hides everything else, so they are
        // the only things x-ray thins. A structure a feature is pointing at
        // stays solid regardless - that is the whole point of calling it out.
        const ghosted =
          xray &&
          !isHighlighted &&
          (node.group === 'myocardium' ||
            node.group === 'atrium' ||
            node.group === 'great_vessel');

        return (
          <mesh key={name} geometry={geometry} renderOrder={ghosted ? -1 : 0}>
            <meshStandardMaterial
              color={isHighlighted ? HIGHLIGHT : style.color}
              roughness={style.roughness}
              metalness={0}
              emissive={isHighlighted ? HIGHLIGHT : '#000000'}
              emissiveIntensity={isHighlighted ? 0.32 : 0}
              transparent={ghosted}
              opacity={ghosted ? style.ghost : 1}
              depthWrite={!ghosted}
              side={THREE.DoubleSide}
            />
          </mesh>
        );
      })}
    </group>
  );
}
