import { useGLTF } from '@react-three/drei';
import { useThree } from '@react-three/fiber';
import { useEffect, useMemo } from 'react';
import * as THREE from 'three';

/**
 * Pull every named node out of anatomy.glb as a flat `name -> geometry` map.
 *
 * The GLB is one file with ~20 nodes rather than one file per structure, so the
 * whole scene is a single request and a single parse. Node names are the
 * contract between `tools/build_anatomy.py` and this viewer: `myo_lv_free`,
 * `artery_LAD`, `valve_mitral`, and so on.
 */
export function useAnatomyGeometry(url: string): Record<string, THREE.BufferGeometry> {
  const { scene } = useGLTF(url);
  const invalidate = useThree((s) => s.invalidate);

  const geometries = useMemo(() => {
    const found: Record<string, THREE.BufferGeometry> = {};
    scene.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (mesh.isMesh && mesh.geometry && node.name) {
        found[node.name] = mesh.geometry;
      }
    });
    return found;
  }, [scene]);

  // Ask for a frame once the meshes exist.
  //
  // Defensive, not a fix for a known fault: `frameloop="demand"` renders only
  // when something requests it, and a Suspense resolution swapping geometry into
  // the scene is not obviously such a request. In practice R3F does invalidate
  // on that commit - a blank canvas here turned out to be the browser tab being
  // backgrounded, where requestAnimationFrame never fires at all and nothing
  // renders regardless. Two frames, because the first can land before the GPU
  // upload on some drivers.
  useEffect(() => {
    invalidate();
    const id = requestAnimationFrame(() => invalidate());
    return () => cancelAnimationFrame(id);
  }, [geometries, invalidate]);

  return geometries;
}
