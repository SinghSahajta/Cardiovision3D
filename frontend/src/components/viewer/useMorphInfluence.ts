import { useThree } from '@react-three/fiber';
import { useLayoutEffect } from 'react';
import type { RefObject } from 'react';
import * as THREE from 'three';

/**
 * Bind a mesh's stenosis morph target to a 0..1 influence.
 *
 * Two things make this necessary rather than just assigning the array:
 *
 * 1. three.js builds `morphTargetInfluences` inside the Mesh constructor, from
 *    the geometry it was given. React Three Fiber constructs the Mesh first and
 *    attaches the geometry afterwards, so that constructor-time call finds
 *    nothing and the array is never created. Every mesh whose geometry carries
 *    `morphAttributes.position` therefore has to call `updateMorphTargets()`
 *    itself - and if it does not, the renderer dereferences the missing array
 *    and throws *during* the draw call, which aborts the whole frame. That is
 *    not a silent failure of one mesh: everything queued behind it disappears
 *    too, which is exactly how the arteries went missing while the myocardium
 *    in front of them kept drawing.
 * 2. `frameloop="demand"` means a change made outside an interaction renders
 *    nothing until something asks for a frame, so a new prediction would leave
 *    the vessel at its previous calibre.
 */
export function useMorphInfluence(
  ref: RefObject<THREE.Mesh | null>,
  influence: number,
  geometry: THREE.BufferGeometry,
) {
  const invalidate = useThree((s) => s.invalidate);

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    if (geometry.morphAttributes?.position) {
      mesh.updateMorphTargets();
      if (mesh.morphTargetInfluences?.length) {
        mesh.morphTargetInfluences[0] = influence;
      }
    }
    invalidate();
  }, [ref, geometry, influence, invalidate]);
}
