#!/usr/bin/env python3
"""Generate the Phase 4 instructional SVG set with one visual grammar."""
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parents[2]

NAVY = "#17324D"
BLUE = "#2563A6"
TEAL = "#0F7C7B"
AMBER = "#C88719"
LIGHT = "#EEF3F7"
MID = "#9AA9B5"
INK = "#202A33"
WHITE = "#FFFFFF"


def svg(width, height, body, title, desc):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>
<defs><marker id="arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7 Z" fill="{NAVY}"/></marker><marker id="arrow-teal" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7 Z" fill="{TEAL}"/></marker></defs>
<rect width="100%" height="100%" fill="{WHITE}"/>
<g font-family="Arial, Helvetica, sans-serif" fill="{INK}" stroke-linecap="round" stroke-linejoin="round">{body}</g></svg>'''


def text(x, y, value, size=18, anchor="middle", weight="normal", fill=INK, family="Arial, Helvetica, sans-serif"):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}" font-family="{family}">{escape(value)}</text>'


def node(x, y, label, w=128, h=44, state="default", size=15):
    styles = {
        "default": (LIGHT, NAVY, ""), "active": ("#D9F1EF", TEAL, ""),
        "selected": ("#FFF1D3", AMBER, ""), "source": (NAVY, NAVY, ""),
        "sink": (TEAL, TEAL, ""), "inactive": (WHITE, MID, 'stroke-dasharray="6 5"')}
    fill, stroke, extra = styles[state]
    color = WHITE if state in {"source", "sink"} else INK
    return f'<rect x="{x-w/2}" y="{y-h/2}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2.4" {extra}/>' + text(x, y+5, label, size, fill=color)


def circle(x, y, label, state="default", r=25):
    styles = {"default": (LIGHT, NAVY), "active": ("#D9F1EF", TEAL), "selected": ("#FFF1D3", AMBER), "source": (NAVY, NAVY), "sink": (TEAL, TEAL)}
    fill, stroke = styles[state]
    color = WHITE if state in {"source", "sink"} else INK
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>' + text(x, y+6, label, 17, fill=color, weight="bold")


def edge(x1, y1, x2, y2, label="", selected=False, dashed=False, directed=True, label_dx=0, label_dy=-8):
    color = TEAL if selected else NAVY
    marker = ' marker-end="url(#arrow-teal)"' if selected and directed else ' marker-end="url(#arrow)"' if directed else ""
    dash = ' stroke-dasharray="7 6"' if dashed else ""
    out = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{4 if selected else 2.3}"{dash}{marker}/>'
    if label: out += f'<rect x="{(x1+x2)/2-24+label_dx}" y="{(y1+y2)/2-21+label_dy}" width="48" height="24" rx="5" fill="white"/>'+text((x1+x2)/2+label_dx,(y1+y2)/2-4+label_dy,label,14,weight="bold")
    return out


def write(chapter, filename, width, height, body, title, desc):
    out = ROOT / "assets" / "figures" / f"chapter{chapter:02d}" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg(width, height, body, title, desc), encoding="utf-8")


# Chapter 2: recursion and decomposition.
b = text(450,30,"Divide",18,weight="bold",fill=BLUE)
b += node(450,75,"[3, 7, 2, 9, 1, 5, 8, 4]",250)
levels=[[(260,"[3, 7, 2, 9]"),(640,"[1, 5, 8, 4]")],[(120,"[3, 7]"),(330,"[2, 9]"),(570,"[1, 5]"),(780,"[8, 4]")]]
for x,l in levels[0]: b+=node(x,165,l,150)+edge(450,98,x,142,directed=False)
for i,(x,l) in enumerate(levels[1]):
    parent=260 if i<2 else 640; b+=node(x,255,l,105)+edge(parent,188,x,232,directed=False)
vals=[3,7,2,9,1,5,8,4]
for i,v in enumerate(vals):
    x=70+i*108; parent=levels[1][i//2][0]; b+=circle(x,340,str(v),"selected" if v==9 else "default",20)+edge(parent,278,x,318,directed=False)
b+=text(450,395,"Combine maxima: 7, 9, 5, 8 → 9",18,weight="bold",fill=TEAL)
write(2,"find-max-recursion.svg",900,425,b,"Find-max recursion tree","The array splits into halves until single values; combining local maxima returns 9.")

b=node(450,55,"[38, 27, 43, 3]",210)
for x,l in [(270,"[38, 27]"),(630,"[43, 3]")]: b+=node(x,145,l,150)+edge(450,78,x,122,directed=False)
for x,l,p in [(190,"[38]",270),(350,"[27]",270),(550,"[43]",630),(710,"[3]",630)]: b+=node(x,235,l,82)+edge(p,168,x,212,directed=False)
b+=node(270,325,"[27, 38]",150,state="active")+node(630,325,"[3, 43]",150,state="active")
b+=edge(190,258,270,302,directed=False)+edge(350,258,270,302,directed=False)+edge(550,258,630,302,directed=False)+edge(710,258,630,302,directed=False)
b+=node(450,415,"[3, 27, 38, 43]",210,state="selected")+edge(270,348,450,392,directed=False)+edge(630,348,450,392,directed=False)
write(2,"merge-sort-decomposition.svg",900,455,b,"Merge-sort decomposition and combination","The array splits to singletons, then sorted halves merge into 3, 27, 38, 43.")

b=""
for level,(count,y,label) in enumerate([(1,65,"cn"),(2,155,"cn/2"),(4,245,"cn/4"),(8,335,"c")]):
    xs=[450] if count==1 else [90+(720/(count-1))*i for i in range(count)]
    for x in xs: b+=circle(x,y,label,"active" if level<3 else "default",27)
    if level:
        prev=[450] if count==2 else [90+(720/(count/2-1))*i for i in range(int(count/2))]
        for i,x in enumerate(xs): b+=edge(prev[i//2],y-63,x,y-29,directed=False)
    b+=text(870,y+6,f"level {level}: cn total",15,"end",weight="bold",fill=TEAL)
b+=text(450,395,"log₂ n + 1 levels × Θ(n) work per level = Θ(n log n)",18,weight="bold")
write(2,"merge-sort-work-tree.svg",940,425,b,"Merge-sort recursion-tree work","Each level contributes cn work and there are logarithmically many levels.")

b=text(235,32,"Balanced partitions",19,weight="bold",fill=TEAL)+text(705,32,"Unbalanced partitions",19,weight="bold",fill=AMBER)
for x,y,l in [(235,75,"n"),(155,155,"n/2"),(315,155,"n/2"),(115,235,"n/4"),(195,235,"n/4"),(275,235,"n/4"),(355,235,"n/4")]: b+=circle(x,y,l,"active" if y<200 else "default",25)
for a,c in [((235,100),(155,130)),((235,100),(315,130)),((155,180),(115,210)),((155,180),(195,210)),((315,180),(275,210)),((315,180),(355,210))]: b+=edge(*a,*c,directed=False)
for i,l in enumerate(["n","n−1","n−2","⋮","1"]):
    y=75+i*62; b+=circle(705,y,l,"selected" if i<3 else "default",25)
    if i: b+=edge(705,y-37,705,y-27,directed=False)
b+=text(235,330,"height Θ(log n); total Θ(n log n)",16,weight="bold")+text(705,330,"height Θ(n); total Θ(n²)",16,weight="bold")
write(2,"quicksort-partition-comparison.svg",940,365,b,"Balanced and unbalanced quicksort recursion","Balanced partitions have logarithmic height; repeatedly choosing an extreme pivot yields linear height.")

# Chapter 3: heap array correspondence.
b=""
coords=[(450,65),(270,155),(630,155),(170,245),(370,245),(530,245),(730,245)]
vals=[50,30,40,20,10,35,15]
for i,(x,y) in enumerate(coords):
    if i: p=(i-1)//2; b+=edge(coords[p][0],coords[p][1]+26,x,y-26,directed=False)
    b+=circle(x,y,str(vals[i]),"source" if i==0 else "default",26)+text(x,y+50,f"i={i}",13,fill=MID)
b+=text(450,318,"Array indices",17,weight="bold",fill=BLUE)
for i,v in enumerate(vals):
    x=210+i*80; b+=f'<rect x="{x-34}" y="340" width="68" height="48" fill="{LIGHT if i else "#DCE7F2"}" stroke="{NAVY}" stroke-width="2"/>'+text(x,370,str(v),17,weight="bold")+text(x,410,str(i),13,fill=MID)
write(3,"binary-heap-array-mapping.svg",900,435,b,"Binary heap and array mapping","A complete max-heap maps level by level to array indices zero through six.")

# Chapter 5: duplicated versus memoized Fibonacci states.
b=text(235,30,"Naive recursion",19,weight="bold",fill=AMBER)+text(705,30,"Memoized states",19,weight="bold",fill=TEAL)
na=[(235,70,"fib(5)"),(165,145,"fib(4)"),(305,145,"fib(3)"),(120,220,"fib(3)"),(210,220,"fib(2)"),(270,220,"fib(2)"),(350,220,"fib(1)")]
for i,(x,y,l) in enumerate(na): b+=node(x,y,l,88,36,"selected" if l=="fib(3)" and i>1 else "default",13)
for a,c in [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]: b+=edge(na[a][0],na[a][1]+19,na[c][0],na[c][1]-19,directed=False)
memo=[(705,70,"fib(5)"),(705,130,"fib(4)"),(705,190,"fib(3)"),(705,250,"fib(2)"),(650,310,"fib(1)"),(760,310,"fib(0)")]
for i,(x,y,l) in enumerate(memo):
    b+=node(x,y,l,92,36,"active" if i<4 else "default",13)
    if i: b+=edge(memo[i-1][0],memo[i-1][1]+19,x,y-19)
b+=text(235,300,"Repeated nodes recompute the same states",14,weight="bold")+text(705,365,"Each state is computed once; later requests are lookups",14,weight="bold")
write(5,"fibonacci-overlap.svg",940,395,b,"Overlapping Fibonacci subproblems","Naive recursion repeats fib(3) and fib(2), while memoization computes a linear set of unique states.")

# Chapter 7: known containments without asserting P versus NP.
b=f'<rect x="45" y="35" width="810" height="330" rx="28" fill="{LIGHT}" stroke="{NAVY}" stroke-width="3"/>'+text(85,70,"EXP",20,"start",weight="bold")
b+=f'<rect x="120" y="85" width="660" height="240" rx="24" fill="white" stroke="{BLUE}" stroke-width="3"/>'+text(155,120,"PSPACE",19,"start",weight="bold",fill=BLUE)
b+=f'<ellipse cx="450" cy="215" rx="245" ry="90" fill="#E8F5F4" stroke="{TEAL}" stroke-width="3"/>'+text(650,160,"NP",19,weight="bold",fill=TEAL)
b+=f'<ellipse cx="345" cy="215" rx="105" ry="58" fill="#FFF1D3" stroke="{AMBER}" stroke-width="3"/>'+text(345,221,"P",22,weight="bold")
b+=text(450,402,"Known: P ⊆ NP ⊆ PSPACE ⊆ EXP. Whether P = NP remains open.",18,weight="bold")
write(7,"complexity-class-containment.svg",900,430,b,"Known complexity-class containments","Nested regions show P inside NP inside PSPACE inside EXP, while explicitly marking P versus NP as open.")

# Chapter 9: network-flow visual sequence.
b=""
for x,y,l,s in [(90,180,"s","source"),(360,80,"a","default"),(360,280,"b","default"),(700,180,"t","sink")]: b+=circle(x,y,l,s,27)
for e in [(117,170,332,92,"10",True),(117,190,332,268,"5",True),(388,92,673,170,"10",True),(388,268,673,190,"15",True)]: b+=edge(*e)
b+=text(450,355,"Maximum flow = 10 on s→a→t plus 5 on s→b→t = 15",18,weight="bold",fill=TEAL)
write(9,"simple-flow-network.svg",800,390,b,"A two-path flow network","Two source-to-sink paths carry 10 and 5 units, achieving a maximum flow of 15.")

b=circle(180,155,"u","source",28)+circle(620,155,"v","sink",28)
b+=edge(210,140,590,140,"3",False,False,True,label_dy=-10)+edge(590,185,210,185,"7",True,True,True,label_dy=23)
b+=text(400,72,"Original edge carries 7/10",20,weight="bold")+text(400,245,"Forward residual capacity 3; backward cancellation capacity 7",17,weight="bold")
write(9,"residual-edge.svg",800,280,b,"Residual capacities for a partially used edge","An edge carrying seven of ten units leaves three forward residual units and seven backward cancellation units.")

b=""
for x,y,l,s in [(80,180,"s","source"),(300,85,"a","default"),(300,275,"b","default"),(690,180,"t","sink")]: b+=circle(x,y,l,s,27)
for e in [(107,168,272,97,"10/10",True),(107,192,272,263,"5/5",True),(328,97,662,168,"10/10",True),(328,263,662,192,"5/15",True)]: b+=edge(*e)
b+=f'<line x1="190" y1="45" x2="190" y2="320" stroke="{AMBER}" stroke-width="4" stroke-dasharray="10 7"/>'+text(210,65,"minimum cut",16,"start",weight="bold",fill=AMBER)
b+=text(400,355,"The source cut crosses capacities 10 and 5, matching the flow value 15",16,weight="bold")
write(9,"max-flow-min-cut.svg",800,385,b,"Flow and cut in the example network","Selected edges carry total flow 15; dashed boundary illustrates how cut capacity is computed.")

b=circle(70,190,"s","source",25)+circle(830,190,"t","sink",25)
left=[(250,85,"L1"),(250,190,"L2"),(250,295,"L3")]; right=[(650,85,"R1"),(650,190,"R2"),(650,295,"R3")]
for x,y,l in left+right: b+=circle(x,y,l,"default",24)
for x,y,l in left: b+=edge(95,190,x-25,y,"1")
for x,y,l in right: b+=edge(x+25,y,805,190,"1")
for i,j,sel in [(0,0,True),(0,1,False),(1,1,True),(2,1,False),(2,2,True)]: b+=edge(275,left[i][1],625,right[j][1],"1",sel)
b+=text(450,355,"Three highlighted unit-flow paths encode the matching {L1–R1, L2–R2, L3–R3}",16,weight="bold")
write(9,"bipartite-matching-flow.svg",900,385,b,"Bipartite matching as unit-capacity flow","Source and sink edges enforce one match per vertex; three highlighted compatibility edges form a matching.")

# Chapter 10: suffix ordering.
rows=[("$",6),("a$",5),("ana$",3),("anana$",1),("banana$",0),("na$",4),("nana$",2)]
b=text(190,38,'Suffixes of "banana$"',19,weight="bold",fill=BLUE)+text(650,38,"Lexicographic order",19,weight="bold",fill=TEAL)
suffixes=["banana$","anana$","nana$","ana$","na$","a$","$"]
for i,s in enumerate(suffixes): b+=node(190,82+i*42,s,220,32,"default",14)
for i,(s,pos) in enumerate(rows):
    y=82+i*42; b+=node(650,y,f"{s}  →  {pos}",220,32,"active" if i<4 else "default",14)
    b+=edge(315,82+pos*42,525,y,directed=True)
b+=text(650,390,"Suffix array = [6, 5, 3, 1, 0, 4, 2]",18,weight="bold")
write(10,"suffix-array-ordering.svg",900,420,b,"Suffix-array ordering for banana","All suffixes of banana with a sentinel are ordered lexicographically to produce indices 6, 5, 3, 1, 0, 4, 2.")

# Chapter 11: FFT multiplication pipeline.
b=""
steps=[(115,"Coefficient vectors","[1,2,3], [4,5,6]","default"),(340,"FFT","evaluate at roots","active"),(565,"Pointwise product","A(ωₖ)B(ωₖ)","selected"),(790,"Inverse FFT","[4,13,28,27,18]","active")]
for x,title,sub,state in steps:
    b+=node(x,145,title,175,55,state,15)+text(x,200,sub,14,family="Courier New, monospace")
for i in range(3): b+=edge(205+i*225,145,250+i*225,145)
b+=text(450,55,"Polynomial multiplication via the convolution theorem",21,weight="bold")+text(450,265,"Two transforms + linear pointwise multiplication + inverse transform",17,weight="bold",fill=TEAL)
write(11,"fft-convolution-pipeline.svg",900,300,b,"FFT polynomial-multiplication pipeline","Coefficient vectors are transformed, multiplied pointwise in the frequency domain, and inverse transformed to product coefficients.")

# Chapter 12: segment tree with query decomposition.
b=text(450,28,"Array",18,weight="bold",fill=BLUE)
arr=[1,3,5,7,9,11]
for i,v in enumerate(arr):
    x=250+i*80; b+=f'<rect x="{x-34}" y="42" width="68" height="42" fill="{("#FFF1D3" if 1<=i<=4 else LIGHT)}" stroke="{NAVY}" stroke-width="2"/>'+text(x,69,str(v),16,weight="bold")+text(x,104,str(i),12,fill=MID)
nodes=[(450,145,"[0–5] 36",180,"default"),(310,225,"[0–2] 9",140,"default"),(590,225,"[3–5] 27",140,"default"),(230,305,"[0–1] 4",115,"default"),(390,305,"[2] 5",100,"selected"),(530,305,"[3–4] 16",125,"selected"),(680,305,"[5] 11",100,"default"),(180,385,"[0] 1",90,"default"),(280,385,"[1] 3",90,"selected"),(500,385,"[3] 7",90,"default"),(570,385,"[4] 9",90,"default")]
for x,y,l,w,s in nodes: b+=node(x,y,l,w,38,s,13)
for a,c in [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6),(3,7),(3,8),(5,9),(5,10)]:
    b+=edge(nodes[a][0],nodes[a][1]+20,nodes[c][0],nodes[c][1]-20,directed=False)
b+=text(450,445,"Query [1,4] decomposes into [1], [2], and [3–4]: 3 + 5 + 16 = 24",17,weight="bold",fill=TEAL)
write(12,"segment-tree-query.svg",900,475,b,"Segment-tree range decomposition","The array 1,3,5,7,9,11 maps to a sum tree; highlighted disjoint nodes answer query indices one through four.")

print("Generated 14 SVG figures.")
