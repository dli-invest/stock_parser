#import "@preview/biz-report:0.2.0": authorwrap, dropcappara, infobox, report  

#show: report.with(
  title: "Business Report",
  publishdate: "November 2025",
  mylogo: image("mylogo.svg", width: 25%),
  myfeatureimage: image("techimage.svg", height: 6cm),
  myvalues: "VALUE1 | VALUE2 | VALUE3 | VALUE4",
  mycolor: rgb("#1300a7"),
  myfont: "IBM Plex Sans"
)
#let summaries = json(bytes(sys.inputs.summaries))
#for summary in summaries [
  #summary.text \
]
