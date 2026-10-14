/**
 * Required clinical safety notice (requirement 5). Always rendered at the top of
 * the page, never collapsible, never behind a scroll.
 *
 * It used to be a saturated dark-red slab, which was the loudest thing on the
 * page. That is worse than useless: a warning rendered as an alarm gets tuned
 * out within a minute of use, and it was competing with the status colours that
 * do carry per-patient meaning. This version is quiet, permanent and legible -
 * a left rule and a tinted strip, holding exactly the same words.
 *
 * In the shell layout it sits above the content canvas and shares its width, so
 * it reads as a property of the whole page rather than of any one panel.
 */
export function DisclaimerBanner() {
  return (
    <aside
      role="note"
      className="mb-2.5 flex flex-wrap items-baseline gap-x-2 gap-y-0.5 rounded-[var(--radius)] px-3.5 py-2 text-[11.5px] leading-snug"
      style={{
        background: 'color-mix(in srgb, var(--risk-high) 7%, var(--surface))',
        color: 'var(--ink-secondary)',
        boxShadow: 'inset 3px 0 0 var(--risk-high)',
      }}
    >
      <strong className="font-semibold" style={{ color: 'var(--ink)' }}>
        Decision support &amp; education only.
      </strong>
      <span>
        Statistical risk estimates from a 303-patient research cohort. Not a diagnostic
        device. Does not replace coronary angiography, cardiac imaging, or clinical
        judgement.
      </span>
    </aside>
  );
}
