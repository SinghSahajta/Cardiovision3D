import { useEffect, useState } from 'react';

/**
 * Which section is currently being read.
 *
 * The rail is navigation over one scrolling document rather than a router, so
 * the active item has to be derived from scroll position. IntersectionObserver
 * does that without a scroll handler firing on every frame.
 *
 * `rootMargin` pulls the detection band up to the top third of the viewport.
 * Without it a tall section like the explanation panel stays "intersecting"
 * long after you have scrolled past its heading, and two items light up at
 * once. Observing a narrow band means exactly one section qualifies, and the
 * one that qualifies is the one under the top of the screen - which is what a
 * reader considers the section they are in.
 *
 * `ids` is re-observed whenever it changes, because the explanation section
 * unmounts when no prediction is available.
 */
export function useScrollSpy(ids: string[]): string | null {
  const [active, setActive] = useState<string | null>(ids[0] ?? null);

  useEffect(() => {
    const nodes = ids
      .map((id) => document.getElementById(id))
      .filter((n): n is HTMLElement => n !== null);
    if (!nodes.length) return;

    const visible = new Set<string>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) visible.add(entry.target.id);
          else visible.delete(entry.target.id);
        }
        // Keep document order rather than the order they happened to fire in,
        // so scrolling up lands on the same item scrolling down did.
        const first = ids.find((id) => visible.has(id));
        if (first) setActive(first);
      },
      { rootMargin: '0px 0px -66% 0px', threshold: 0 },
    );

    nodes.forEach((n) => observer.observe(n));
    return () => observer.disconnect();
  }, [ids]);

  return active;
}
