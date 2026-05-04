#import "@preview/polylux:0.4.0": *
#import "./slides.typ": *

#set page(paper: "presentation-16-9", margin: 30pt)

#title-slide(
  [Worcester Polytechnic Institute],
  [Presentation Template],
  (
    ("Joe Gorton", "CS & ECE Department"),
    ("Caz Cherniko", "CS & Physics Department"),
    ("Adam Field", "Physics Department"),
  ),
  date: [April 2026],
)

#section-slide([Motivation], number: [01],
  subtitle: [Why does this problem matter?])

#content-slide([Background])[
  A brief summary of prior work and relevant context.

  - *Key point one* - establish the setting
  - *Key point two* - supporting evidence or physical intuition
  - *Key point three* - what this work builds on

  Narrative text flows naturally here.  You can include inline math, e.g.
  the shear distortion tensor $gamma = gamma_1 + i gamma_2$, without a
  dedicated math slide. Here is a citation: @Brenner2008
]

#two-col-slide(
  [Method Overview],
  
  [
    *Architecture*

    - Input: galaxy stamp $I$, PSF stamp $P$
    - Two convolutional branches
    - Fused via cross-attention layer
    - Output: $hat(gamma)_1, hat(gamma)_2$

    *Loss*

    $ cal(L) = sum_i w_i (hat(gamma)_i - gamma_i)^2 $
  ],
  
  align(center + horizon,
    rect(
      width: 100%,
      height: 200pt,
      fill: surface,
      radius: 6pt,
      stroke: 1pt + muted,
    )
  ),
  left-w: 2fr,
  right-w: 3fr,
)

#figure-slide(
  [Results],
  caption: [Figure 1: Placeholder - replace with your plot or diagram.],
)[
  #rect(width: 70%, height: 210pt, fill: surface, radius: 6pt, stroke: 1pt + muted)
]

#math-slide(
  [Variational Loss],
  $ cal(L)[f] = integral_Omega lr(|nabla f|)^2 dif Omega + lambda lr(||f - g||)^2 $,
  before: [We minimise the Tikhonov-regularised functional:],
  after: [where $lambda > 0$ controls regularisation and $g$ is the observed signal.],
)

#quote-slide(attribution: [Alice Gaehring])[
  Say something profound.
]

#section-slide([Analysis & Discussion], number: [02])

#content-slide([Discussion])[
  Summary of main findings and their interpretation.

  + First conclusion: tied directly to the data
  + Second conclusion: broader implication
  + Open questions that remain

  #v(12pt)

  #block(fill: surface, radius: 6pt, inset: (x: 18pt, y: 12pt), width: 100%)[
    *Takeaway:* A highlighted callout box: useful for the single key message
    you want the audience to leave with.
  ]
]

#ack-slide((
  ("Dr. Sayan Saha",       "Research mentorship and direction on the ShearNet project at Northeastern University."),
  ("Dr. William Sanguinet","Advising on Finite Element Methods and Finite Volume Methods, C term 2026."),
  ("WPI Physics Dept.",    "Institutional support and computing resources."),
))

#end-slide(
  bib: bibliography("bib.bib", title: [#v(-1.2em)])
)