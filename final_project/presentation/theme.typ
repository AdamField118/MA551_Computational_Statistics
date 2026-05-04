// ─────────────────────────────────────────────────────────────────────────────
// theme.typ — WPI presentation design tokens
// ─────────────────────────────────────────────────────────────────────────────

// ── Palette ───────────────────────────────────────────────────────────────────
#let crimson = rgb("#AC2B37")    // WPI primary
#let wpi-black = rgb("#000000")
#let wpi-white = rgb("#FFFFFF")
#let muted = rgb("#999999")      // secondary / caption text
#let surface = rgb("#111111")    // slightly lighter black, for inset boxes

// ── Typography ────────────────────────────────────────────────────────────────
#let display-font = "Iosevka Extended"   // headings, wordmarks, section titles
#let body-font    = "Iosevka"            // body text, code, authors
#set cite(style: "springer-basic-author-date")
#show bibliography: set text(18pt)
#show cite: set text(rgb(200,125,0), style: "italic")

// ── Size scale ────────────────────────────────────────────────────────────────
#let sz = (
  display:  44pt,   // section-slide titles
  title:    38pt,   // title-slide main title
  subtitle: 28pt,   // title-slide subtitle
  h1:       26pt,   // slide headings
  body:     18pt,   // body text
  caption:  13pt,   // figure captions, small labels
  small:    11pt,   // footer, meta text
)
