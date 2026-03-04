#import "@preview/biz-report:0.2.0": authorwrap, dropcappara, infobox, report  
#import "@preview/cmarker:0.1.8"
#show: report.with(
  title: "Business Report",
  publishdate: "November 2025",

  myvalues: "VALUE1 | VALUE2 | VALUE3 | VALUE4",
  mycolor: rgb("#1300a7"),
  myfont: "IBM Plex Sans"
)

#outline(
  title: "Impacted Stock Tickers",
  depth: 1,
  indent: auto // <--- Fix: Changed from `true` to `auto`
)

#let summaries = json(bytes(sys.inputs.summaries))
#for summary in summaries [
  // Creates a Heading (which populates the ToC)
  = #summary.ticker 
  
  // Renders the Gemini summary text
  #cmarker.render(summary.text)
  
  #v(1.5em) // Adds vertical spacing between entries
]