import { useThree } from '@react-three/fiber';
import { useCallback, useEffect, useLayoutEffect, useRef } from 'react';
import * as THREE from 'three';

import type { VesselKey } from '../../api/types';

export interface LabelAnchor {
  key: VesselKey;
  point: [number, number, number];
}

export interface ProjectedLabel {
  key: VesselKey;
  x: number;
  y: number;
  visible: boolean;
  /** False when the anchor is on the far side of the model from the camera. */
  facing: boolean;
}

/**
 * Projects vessel anchor points to screen coordinates and reports them upward,
 * so the labels can be rendered as ordinary DOM outside the Canvas.
 *
 * This exists instead of drei's <Html>, which updates its transform inside a
 * useFrame callback. Under `frameloop="demand"` that callback does not tick, so
 * Html labels either never position themselves or do so only by accident when
 * something else keeps the loop alive.
 *
 * Projecting explicitly and recomputing on the events that actually move things
 * (camera change, resize, new anchors) is both correct in demand mode and
 * cheaper than a permanent render loop.
 */
export function VesselLabels({
  anchors,
  onChange,
}: {
  anchors: LabelAnchor[];
  onChange: (labels: ProjectedLabel[]) => void;
}) {
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  const canvas = useThree((s) => s.gl.domElement);
  const controls = useThree((s) => s.controls) as
    | { addEventListener?: (t: string, fn: () => void) => void;
        removeEventListener?: (t: string, fn: () => void) => void }
    | null;

  const vec = useRef(new THREE.Vector3());
  const origin = useRef(new THREE.Vector3());

  const project = useCallback(() => {
    // Measure the canvas rather than trusting R3F's reported `size`. That value
    // comes from a ResizeObserver, which does not fire while the tab is
    // backgrounded - the canvas then reports its 300x150 default and every
    // label lands at a fraction of its true position, drifting off the model as
    // soon as the camera moves. getBoundingClientRect is always current.
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || size.width;
    const height = rect.height || size.height;

    // Depth of the model's centre, used to tell front-facing anchors from ones
    // that have rotated round the back. Cheaper and steadier than raycasting
    // every label against the myocardium on every mouse move.
    const centreDepth = origin.current.set(0, 0, 0).distanceTo(camera.position);

    onChange(
      anchors.map(({ key, point }) => {
        vec.current.set(point[0], point[1], point[2]);
        const depth = vec.current.distanceTo(camera.position);
        vec.current.project(camera);
        return {
          key,
          x: (vec.current.x * 0.5 + 0.5) * width,
          y: (-vec.current.y * 0.5 + 0.5) * height,
          // z outside [-1, 1] means the point is behind the camera or past the
          // far plane; such a label would otherwise appear mirrored on screen.
          visible: vec.current.z > -1 && vec.current.z < 1,
          facing: depth <= centreDepth,
        };
      }),
    );
  }, [anchors, camera, canvas, size.width, size.height, onChange]);

  // Layout effect so the first paint already has positions, rather than
  // flashing labels in the top-left corner for a frame.
  useLayoutEffect(project, [project]);

  useEffect(() => {
    if (!controls?.addEventListener) return;
    controls.addEventListener('change', project);
    return () => controls.removeEventListener?.('change', project);
  }, [controls, project]);

  useEffect(() => {
    window.addEventListener('resize', project);
    return () => window.removeEventListener('resize', project);
  }, [project]);

  return null;
}
