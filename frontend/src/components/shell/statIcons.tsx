/**
 * Glyphs for the tinted stat strip.
 *
 * Separate module so StatStrip.tsx exports a component and nothing else -
 * react-refresh cannot hot-reload a file that mixes the two.
 */
const iconProps = {
  width: 16,
  height: 16,
  viewBox: '0 0 16 16',
  fill: 'none',
  stroke: 'var(--ink)',
  strokeWidth: 1.5,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
};

export const STAT_ICONS = {
  people: (
    <svg {...iconProps}>
      <circle cx="6" cy="5.5" r="2.25" />
      <path d="M2 13.5c0-2.2 1.8-3.5 4-3.5s4 1.3 4 3.5" />
      <path d="M11 4.2a2.25 2.25 0 0 1 0 4.3M12.2 13.5c0-1.6-.6-2.7-1.6-3.3" />
    </svg>
  ),
  gauge: (
    <svg {...iconProps}>
      <path d="M2.5 12a5.5 5.5 0 1 1 11 0" />
      <path d="M8 12 10.8 8.4" />
    </svg>
  ),
  chip: (
    <svg {...iconProps}>
      <rect x="4.5" y="4.5" width="7" height="7" rx="1.5" />
      <path d="M6.5 2.5v2m3-2v2m-3 9v-2m3 2v-2M2.5 6.5h2m-2 3h2m9-3h2m-2 3h2" />
    </svg>
  ),
};
