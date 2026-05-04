// slides.typ - WPI Polylux slide layout functions
//
// Slide catalogue:
//   title-slide   - opening title card
//   section-slide - chapter / section break
//   content-slide - standard heading + body
//   two-col-slide - left / right split layout
//   figure-slide  - centred figure with caption
//   math-slide    - equation on a dark inset box
//   quote-slide   - crimson-background pull quote
//   ack-slide     - acknowledgements grid
//   end-slide     - closing slide with optional bibliography

#import "@preview/polylux:0.4.0": *
#import "./theme.typ": *

#let bar = place(
  top + left,
  dx: -30pt,
  dy: -30pt,
  rect(width: 100% + 60pt, height: 4pt, fill: crimson),
)

#let footer = place(
  bottom,
  grid(
    columns: (1fr, auto),
    align(
      left + horizon,
      text(fill: muted, size: sz.small, font: display-font)[WORCESTER POLYTECHNIC INSTITUTE],
    ),
    align(
      right + horizon,
      context text(fill: muted, size: sz.small, font: display-font)[
        #counter(page).display()
      ],
    ),
  ),
)

#let rule = {
  v(-6pt)
  line(length: 100%, stroke: 1.5pt + crimson)
  v(6pt)
}

#let left-stripe(col: crimson) = place(
  top + left,
  dx: -30pt,
  dy: -30pt,
  rect(width: 8pt, height: 100% + 60pt, fill: col),
)

#let _base(body) = slide[
  #set page(fill: wpi-black)
  #set text(fill: wpi-white, font: body-font, size: sz.body)
  #set list(marker: text(fill: crimson)[▸])
  #set enum(numbering: n => text(fill: crimson)[#n.])
  //#bar
  #footer
  #body
]



/// 1. title-slide
///
/// Opening title card.
///
/// - title    (content) Main title
/// - subtitle (content) Subtitle or short description
/// - authors  (array)   Sequence of (name, affiliation) pairs
/// - date     (content) Optional date string; pass `none` to omit
///
/// Example:
///   #title-slide([My Talk], [A subtitle], (("Adam Field", "Physics"),), date: [April 2026])
#let title-slide(title, subtitle, authors, date: none) = slide[
  #set page(fill: wpi-black)
  #set text(fill: wpi-white)

  #table(
    columns: 2,
    stroke: 0pt,
    inset: 0pt,
    column-gutter: 15pt,
    align: horizon,
    image("wpi.svg", height: 0.8in),
    text(
      fill: wpi-white,
      size: 20pt,
      weight: "bold",
      font: display-font,
    )[WORCESTER POLYTECHNIC INSTITUTE],
  )

  #place(horizon, dy: -20pt, {
    set text(font: body-font)
    text(fill: wpi-white, size: sz.title, weight: "bold")[#title]
    linebreak()
    text(fill: muted, size: sz.subtitle, weight: "bold")[#subtitle]
    if date != none {
      linebreak()
      v(8pt)
      text(fill: muted, size: sz.caption)[#date]
    }
  })

  #place(bottom, {
    set text(font: body-font, size: 20pt, weight: "semibold")
    table(
      columns: 2,
      stroke: 0pt,
      column-gutter: 10pt,
      inset: 0pt,
      row-gutter: 8pt,
      ..(for author in authors {
        (
          text(fill: wpi-white)[#author.at(0)],
          text(fill: muted)[#author.at(1)],
        )
      })
    )
  })

  #place(bottom + right, image("background.png", height: 30em), dx: 30pt, dy: 30pt)
]



/// 2. section-slide
///
/// Chapter / section break.  A large decorative number floats on the right;
///
/// - title    (content)       Section heading
/// - number   (content|none)  Short label, e.g. [01] or [II]; pass `none` to omit
/// - subtitle (content|none)  Optional short description below title
///
/// Example:
///   #section-slide([Methodology], number: [02], subtitle: [How we approach the problem])
#let section-slide(title, number: none, subtitle: none) = slide[
  #set page(fill: wpi-black)
  #set text(fill: wpi-white)

  #if number != none {
    place(
      right + horizon,
      dx: 30pt,
      text(fill: muted.transparentize(70%), size: 180pt, weight: "bold", font: display-font)[#number],
    )
  }

  #place(left + horizon, pad(left: 24pt, {
    if number != none {
      text(
        fill: crimson,
        size: sz.caption,
        weight: "bold",
        font: display-font,
        tracking: 2pt,
      )[SECTION #number]
      v(8pt)
    }
    text(fill: wpi-white, size: sz.display, weight: "bold", font: display-font)[#title]
    if subtitle != none {
      linebreak()
      v(6pt)
      text(fill: muted, size: sz.h1, font: body-font)[#subtitle]
    }
  }))
]

/// 3. content-slide
///
/// Standard slide: heading + free-form body (bullets, paragraphs, nested grids …).
///
/// - title (content) Slide heading
/// - body  (content) Trailing content block
///
/// Example:
///   #content-slide([Background])[
///     - Point one
///     - Point two
///   ]
#let content-slide(title, body) = _base[
  #text(size: sz.h1, weight: "bold", font: display-font)[#title]
  #rule
  #body
]

/// 4. two-col-slide
///
/// Side-by-side layout.  Both column bodies are passed as positional arguments.
///
/// - title   (content)   Slide heading
/// - left    (content)   Left column content
/// - right   (content)   Right column content
/// - left-w  (fraction)  Column width ratio, default 1fr
/// - right-w (fraction)  Column width ratio, default 1fr
///
/// Example:
///   #two-col-slide([Method], [#bullets], [#figure(...)], left-w: 2fr, right-w: 3fr)
#let two-col-slide(title, left, right, left-w: 1fr, right-w: 1fr) = _base[
  #text(size: sz.h1, weight: "bold", font: display-font)[#title]
  #rule
  #grid(
    columns: (left-w, right-w),
    column-gutter: 28pt,
    left,
    right,
  )
]

/// 5. figure-slide
///
/// Full-slide centred figure - image, plot, or any other visual.
///
/// - title   (content)       Slide heading
/// - caption (content|none)  Figure caption; pass `none` to omit
/// - body    (content)       Trailing content block (the figure itself)
///
/// Example:
///   #figure-slide([Results], caption: [Figure 1: Shear residuals.])[
///     #image("plot.pdf", width: 75%)
///   ]
#let figure-slide(title, caption: none, body) = _base[
  #text(size: sz.h1, weight: "bold", font: display-font)[#title]
  #rule
  #align(center + horizon)[
    #body
    #if caption != none {
      v(8pt)
      text(fill: muted, size: sz.caption, font: body-font)[#caption]
    }
  ]
]

/// 6. math-slide
///
/// Equation-focused slide.  The equation is displayed in a dark inset box.
/// Optional context text can appear above and/or below.
///
/// - title  (content)       Slide heading
/// - eq     (content)       The equation (math mode, e.g. $ ... $)
/// - before (content|none)  Text above the equation box
/// - after  (content|none)  Text below the equation box
///
/// Example:
///   #math-slide(
///     [Variational Loss],
///     $ cal(L) = integral_Omega (nabla f)^2 dif Omega + lambda ||f - g||^2 $,
///     before: [We minimise:],
///     after:  [$lambda$ controls regularisation strength.],
///   )
#let math-slide(title, eq, before: none, after: none) = _base[
  #text(size: sz.h1, weight: "bold", font: display-font)[#title]
  #rule
  #if before != none { before; v(14pt) }
  #align(center,
    block(
      fill: surface,
      radius: 8pt,
      inset: (x: 32pt, y: 22pt),
      text(fill: wpi-white, size: sz.h1)[#eq],
    )
  )
  #if after != none { v(14pt); after }
]

// 7. quote-slide
///
/// Full-bleed crimson background with a large pull quote.
///
/// - attribution (content|none) Person or source; pass `none` to omit
/// - body        (content)      Trailing content block - the quote text
///
/// Example:
///   #quote-slide(attribution: [Richard Feynman])[
///     The first principle is that you must not fool yourself.
///   ]
#let quote-slide(attribution: none, body) = slide[
  #set page(fill: crimson)
  #set text(fill: wpi-white, font: body-font)

  #place(center + horizon,
    pad(x: 60pt, {
      text(
        fill: wpi-white.transparentize(60%),
        size: 110pt,
        weight: "bold",
        font: display-font,
        baseline: 35pt,
      )["]
      v(-45pt)
      text(size: sz.h1, weight: "semibold")[#body]
      if attribution != none {
        v(18pt)
        text(fill: wpi-white.darken(15%), size: sz.body)[--- #attribution]
      }
    })
  )
]

// 8. ack-slide
///
/// Acknowledgements laid out as a two-column grid: names in crimson on the
/// left, descriptions on the right.
///
/// - items (array) Sequence of (name, description) pairs
///
/// Example:
///   #ack-slide((
///     ("Dr. Sayan Saha", "Research mentorship and direction."),
///     ("NSF",           "Grant support under award #XXXXXX."),
///   ))
#let ack-slide(items) = _base[
  #text(size: sz.h1, weight: "bold", font: display-font)[Acknowledgements]
  #rule
  #grid(
    columns: (auto, 1fr),
    column-gutter: 24pt,
    row-gutter: 16pt,
    ..(items.map(item => (
      text(fill: crimson, weight: "bold", size: sz.body)[#item.at(0)],
      text(fill: wpi-white, size: sz.body)[#item.at(1)],
    )).flatten())
  )
]


// ═════════════════════════════════════════════════════════════════════════════
// 9. end-slide
// ═════════════════════════════════════════════════════════════════════════════
/// Closing slide.  Repeats the WPI wordmark, centres a thank-you message and
/// a "Questions?" prompt.  Accepts an optional bibliography block at the bottom.
///
/// - thanks    (content) Primary closing message
/// - questions (content) Secondary prompt (defaults to "Questions?")
/// - bib       (content|none) Pass e.g. `bibliography("bib.bib", …)` or `none`
///
/// Example:
///   #end-slide(bib: bibliography("bib.bib", title: [#v(-1.2em)]))
#let end-slide(
  thanks: [Thank you for your time!],
  questions: [Questions?],
  bib: none,
) = slide[
  #set page(fill: wpi-black)
  #set text(fill: wpi-white)
  //#bar

  // Small wordmark in top-left
  #place(top + left, dy: 12pt,
    table(
      columns: 2,
      stroke: 0pt,
      inset: 0pt,
      column-gutter: 10pt,
      align: horizon,
      image("wpi.svg", height: 0.45in),
      text(fill: muted, size: sz.small, weight: "bold", font: display-font)[WORCESTER POLYTECHNIC INSTITUTE],
    )
  )

  // Centred closing text - shift up slightly if bibliography is present
  #place(center + horizon, dy: if bib != none { -85pt } else { 0pt },
    align(center, {
      text(size: sz.title, weight: "bold", font: display-font)[#thanks]
      v(-20pt)
      text(fill: crimson, size: sz.h1, weight: "bold", font: display-font)[#questions]
    })
  )

  #if bib != none {
    place(center + horizon,
      align(left,
        block(
          width: 70%,
          bib,
        )
      )
    )
  }
]