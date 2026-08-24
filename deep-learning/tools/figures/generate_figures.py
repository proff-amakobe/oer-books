#!/usr/bin/env python3
"""Generate the original vector figures for Deep Learning, Phase 2.5."""
from pathlib import Path
from xml.sax.saxutils import escape
import math

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY="#081B33"; INDIGO="#263F8F"; BLUE="#3157D5"; CYAN="#29A7C8"
SLATE="#5B6678"; LIGHT="#F4F7FA"; WHITE="#FFFFFF"; PALE="#E8EEF8"; GRAY="#AAB3C0"

class SVG:
    def __init__(self, w=1200, h=700, title=""):
        self.w,self.h=w,h
        self.a=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc">',
                f'<title id="title">{escape(title)}</title><desc id="desc">{escape(title)}</desc>',
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#081B33"/></marker><style>text{font-family:Arial,Helvetica,sans-serif;fill:#081B33}.t{font-size:28px;font-weight:700}.h{font-size:22px;font-weight:700}.b{font-size:18px}.s{font-size:15px;fill:#5B6678}.axis{stroke:#5B6678;stroke-width:2}.grid{stroke:#D8DEE8;stroke-width:1}.flow{stroke:#081B33;stroke-width:3;fill:none;marker-end:url(#arrow)}.dash{stroke-dasharray:9 7}.panel{fill:#F4F7FA;stroke:#AAB3C0;stroke-width:2}.module{fill:#E8EEF8;stroke:#263F8F;stroke-width:2}.accent{fill:#DDF4F8;stroke:#238BA8;stroke-width:2}</style></defs>',
                f'<rect width="{w}" height="{h}" fill="{WHITE}"/>']
    def rect(self,x,y,w,h,fill=LIGHT,stroke=SLATE,rx=12,sw=2): self.a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def line(self,x1,y1,x2,y2,cls="flow",stroke=None,sw=None): self.a.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"'+(f' style="stroke:{stroke};stroke-width:{sw or 2}"' if stroke else '')+'/>')
    def path(self,d,stroke=BLUE,sw=4,fill="none",dash=""): self.a.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" {dash}/>' )
    def circle(self,cx,cy,r,fill=WHITE,stroke=BLUE,sw=3): self.a.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def text(self,x,y,s,cls="b",anchor="middle",fill=None): self.a.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"'+(f' fill="{fill}"' if fill else '')+f'>{escape(s)}</text>')
    def poly(self,pts,fill=BLUE,stroke=NAVY,sw=2): self.a.append(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def save(self,name):
        self.a.append('</svg>'); (OUT/name).write_text('\n'.join(self.a),encoding='utf-8')

def axes(g,x,y,w,h,xlabel="Training progress",ylabel="Loss"):
    for i in range(1,4): g.line(x,y+i*h/4,x+w,y+i*h/4,"grid")
    g.line(x,y+h,x+w,y+h,"axis"); g.line(x,y,x,y+h,"axis")
    g.text(x+w/2,y+h+38,xlabel,"s"); g.text(x-48,y+h/2,ylabel,"s",anchor="middle")

def timeline():
    g=SVG(2200,700,"Timeline of neural-network milestones and two AI Winters")
    g.text(1100,55,"Neural Networks: Belief, Setback, and Vindication","t")
    x0,x1,y=100,2100,350
    def X(yr): return x0+(yr-1943)/(2020-1943)*(x1-x0)
    for a,b,label in [(1969,1982,"FIRST AI WINTER"),(1990,2006,"SECOND, MILDER WINTER (APPROX.)")]:
        g.rect(X(a),105,X(b)-X(a),440,"#E9EDF2","none",0,0); g.text((X(a)+X(b))/2,135,label,"s")
    g.line(x0,y,x1,y,"flow")
    events=[(1943,"McCulloch–Pitts","mathematical neuron"),(1958,"Perceptron","trainable classifier"),(1969,"Minsky–Papert","single-layer limits"),(1986,"Backpropagation","multilayer training"),(1990,"LeNet","handwritten digits"),(1997,"LSTM","long sequences"),(2012,"AlexNet","ImageNet breakthrough"),(2014,"GANs","generative learning"),(2017,"Transformer","attention architecture"),(2020,"GPT-3 + AlphaFold 2","language + proteins")]
    modern_positions={2012:(1670,205),2014:(1780,505),2017:(1930,205),2020:(2050,505)}
    for i,(yr,a,b) in enumerate(events):
        x=X(yr); up=i%2==0; lx,yy=modern_positions.get(yr,(x,205 if up else 505))
        g.line(x,y,lx,yy+(45 if up else -45),"axis"); g.circle(x,y,8,CYAN,NAVY,2)
        g.text(lx,yy,a,"h"); g.text(lx,yy+25,b,"s"); g.text(lx,yy+(50 if up else -28),str(yr),"s")
    g.save("fig-1-2-neural-network-timeline.svg")

def hierarchy():
    g=SVG(1400,620,"Four-stage hierarchy from raw pixels to object recognition")
    g.text(700,55,"Hierarchical Feature Learning","t")
    labels=[("RAW PIXELS","image values","▦"),("EDGES + GRADIENTS","local contrast","╱ ╲ ─"),("SHAPES + TEXTURES","curves · corners","○ △ ≋"),("OBJECT-LEVEL","dog face","DOG")]
    for i,(h,b,icon) in enumerate(labels):
        x=55+i*340; g.rect(x,150,290,320,PALE if i<3 else "#DDF4F8",INDIGO)
        g.text(x+145,195,h,"h");
        if i==3: g.text(x+145,225,"REPRESENTATION","h")
        g.text(x+145,320,icon,"t"); g.text(x+145,420,b,"b")
        if i<3: g.line(x+290,310,x+335,310)
    g.text(700,535,"Each learned representation becomes the input to the next layer.","b")
    g.save("fig-1-3-hierarchical-feature-learning.svg")

def landscape():
    g=SVG(1200,720,"Conceptual loss landscape with multiple low-loss regions and descent paths")
    g.text(600,50,"The Loss Landscape (Conceptual)","t")
    cx,cy=600,375
    for k in range(6):
        rx=390-k*55; ry=250-k*35
        g.a.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="{GRAY if k<3 else BLUE}" stroke-width="2"/>')
    g.a.append('<ellipse cx="790" cy="270" rx="95" ry="65" fill="none" stroke="#29A7C8" stroke-width="2" stroke-dasharray="7 6"/>')
    g.path("M190 155 C300 190 330 310 430 350 S520 395 585 380",INDIGO,5)
    g.path("M1040 570 C920 530 940 430 850 400 S760 350 690 375",CYAN,5)
    g.circle(190,155,9,WHITE,INDIGO); g.circle(1040,570,9,WHITE,CYAN)
    g.text(600,390,"low-loss region","h"); g.text(800,255,"another basin","s")
    g.text(600,680,"Contours indicate equal loss; paths are illustrative, not measured optimizer trajectories.","s")
    g.save("fig-3-1-loss-landscape.svg")

def training_loop():
    g=SVG(1200,720,"Four-phase training loop from forward pass through parameter update")
    g.text(600,55,"The Heartbeat of Training","t")
    items=[(600,155,"1  FORWARD PASS","inputs → predictions"),(910,355,"2  LOSS COMPUTATION","compare predictions to targets"),(600,555,"3  BACKWARD PASS","compute gradients by chain rule"),(290,355,"4  PARAMETER UPDATE","optimizer adjusts weights")]
    for i,(x,y,h,b) in enumerate(items):
        g.rect(x-170,y-65,340,130,PALE if i%2==0 else "#DDF4F8",INDIGO); g.text(x,y-8,h,"h"); g.text(x,y+28,b,"s")
    g.a.append('<path d="M770 155 C900 175 980 235 910 290" class="flow"/>')
    g.a.append('<path d="M910 420 C875 505 790 555 770 555" class="flow"/>')
    g.a.append('<path d="M430 555 C300 530 230 450 290 420" class="flow"/>')
    g.a.append('<path d="M290 290 C325 205 415 155 430 155" class="flow"/>')
    g.text(600,360,"repeat for each mini-batch","b")
    g.save("fig-3-3-training-loop.svg")

def learning_rates():
    g=SVG(1400,720,"Conceptual training-loss curves under four learning-rate choices")
    g.text(700,48,"Learning Rate Effects (Illustrative)","t")
    axes(g,90,110,1220,500)
    g.path("M100 170 C180 210 220 130 300 250 S430 120 510 330 S650 190 730 410 S900 250 1000 520 S1160 390 1290 585",NAVY,4)
    g.path("M100 170 C280 300 410 420 600 500 S950 565 1290 575",BLUE,5)
    g.path("M100 170 C390 230 700 315 1000 405 S1200 455 1290 480",SLATE,4,dash='stroke-dasharray="10 7"')
    g.path("M100 170 C160 165 220 190 300 240 C520 430 850 550 1290 580",CYAN,5)
    for yy,label,color in [(125,"too high: unstable",NAVY),(153,"just right: steady",BLUE),(181,"too low: slow",SLATE),(209,"warmup + cosine",CYAN)]: g.line(930,yy,980,yy,"axis",color,4); g.text(995,yy+6,label,"s",anchor="start")
    g.save("fig-3-4-learning-rate-effects.svg")

def batch_paths():
    g=SVG(1400,620,"Batch, stochastic, and mini-batch descent paths on matched contour plots")
    g.text(700,48,"Gradient Descent Strategies (Conceptual)","t")
    panels=[("BATCH","stable estimate",BLUE,[(80,90),(150,150),(220,205),(285,250)]),("STOCHASTIC (SGD)","noisy estimate",NAVY,[(80,90),(155,125),(130,185),(235,165),(205,235),(285,250)]),("MINI-BATCH","balanced estimate",CYAN,[(80,90),(145,145),(180,135),(225,210),(285,250)])]
    for i,(h,b,c,pts) in enumerate(panels):
        ox=45+i*450; oy=120; g.rect(ox,oy,410,390,WHITE,GRAY)
        for k in range(4): g.a.append(f'<ellipse cx="{ox+295}" cy="{oy+275}" rx="{150-k*30}" ry="{110-k*22}" fill="none" stroke="#D8DEE8" stroke-width="2"/>')
        d='M'+' L'.join(f'{ox+x} {oy+y}' for x,y in pts); g.path(d,c,5); g.circle(ox+285,oy+250,8,WHITE,c)
        g.text(ox+205,oy+35,h,"h"); g.text(ox+205,oy+365,b,"s")
    g.text(700,570,"More frequent estimates trade stability for responsiveness; paths are illustrative.","s")
    g.save("fig-3-5-batch-strategies.svg")

def training_curves():
    g=SVG(1500,860,"Four diagnostic patterns comparing training and validation loss")
    g.text(750,48,"Reading Training Curves","t")
    specs=[("UNDERFITTING","both losses remain high","increase capacity / train longer"),("GOOD FIT","both decline and track","retain checkpoint"),("MILD OVERFITTING","validation plateaus","regularize / early stop"),("SEVERE OVERFITTING","validation rises","stop and correct overfit")]
    for i,(h,b,fix) in enumerate(specs):
        ox=50+(i%2)*735; oy=100+(i//2)*370; g.rect(ox,oy,680,320,WHITE,GRAY)
        axes(g,ox+70,oy+70,550,170,"","")
        if i==0: p1="M120 130 C260 135 430 138 600 140"; p2="M120 118 C270 122 450 126 600 130"
        elif i==1: p1="M120 130 C260 200 400 245 600 270"; p2="M120 135 C270 195 420 235 600 250"
        elif i==2: p1="M120 130 C270 205 420 250 600 275"; p2="M120 135 C270 200 400 220 600 220"
        else: p1="M120 130 C270 210 430 260 600 280"; p2="M120 135 C270 205 400 205 600 145"
        def shift(d):
            nums=d.split(); out=[]
            for token in nums:
                if token[0] in 'MC' or token[0].isdigit(): out.append(token)
                else: out.append(token)
            return d
        # Paths use panel-local coordinates translated by a group.
        g.a.append(f'<g transform="translate({ox},{oy})"><path d="{p1}" fill="none" stroke="{BLUE}" stroke-width="5"/><path d="{p2}" fill="none" stroke="{CYAN}" stroke-width="5" stroke-dasharray="10 6"/></g>')
        g.text(ox+340,oy+35,h,"h"); g.text(ox+340,oy+275,b,"s"); g.text(ox+340,oy+300,"Response: "+fix,"s")
    g.line(560,60,610,60,"axis",BLUE,4); g.text(620,66,"training","s",anchor="start"); g.line(760,60,810,60,"axis",CYAN,4); g.text(820,66,"validation","s",anchor="start")
    g.save("fig-3-6-training-curves.svg")

def dropout():
    g=SVG(1600,650,"Full network, two dropout masks during training, and full network at inference")
    g.text(800,45,"How Dropout Works","t")
    titles=["FULL NETWORK","TRAINING STEP A","TRAINING STEP B","INFERENCE"]
    masks=[set(),{(1,1),(2,3),(1,4)},{(1,0),(2,1),(2,4)},set()]
    for p,title in enumerate(titles):
        ox=25+p*395; g.rect(ox,95,365,455,WHITE,GRAY); g.text(ox+182,130,title,"h")
        layers=[[0,1,2],[0,1,2,3,4],[0,1,2,3,4],[0,1]]
        xs=[ox+55,ox+145,ox+235,ox+320]
        coords={}
        for li,ns in enumerate(layers):
            for j in ns: coords[(li,j)]=(xs[li],195+j*58)
        for li in range(3):
            for a in layers[li]:
                for b in layers[li+1]:
                    if (li,a) not in masks[p] and (li+1,b) not in masks[p]: g.line(*coords[(li,a)],*coords[(li+1,b)],"axis",GRAY,1.4)
        for key,(x,y) in coords.items():
            off=key in masks[p]; g.circle(x,y,13,WHITE if not off else "#D8DEE8",BLUE if not off else SLATE,3)
            if off: g.line(x-9,y-9,x+9,y+9,"axis",SLATE,3)
        g.text(ox+182,585,"random masks" if p in (1,2) else ("all units active; scaled" if p==3 else "reference"),"s")
    g.save("fig-3-7-dropout.svg")

def gradients():
    g=SVG(1500,720,"Vanishing and exploding gradients across layers with diagnostic loss curves")
    g.text(750,45,"Vanishing and Exploding Gradients","t")
    for i,(title,color,grow) in enumerate([("VANISHING",BLUE,False),("EXPLODING",NAVY,True)]):
        ox=55+i*720; g.rect(ox,90,670,540,WHITE,GRAY); g.text(ox+335,130,title,"h")
        for j in range(6):
            x=ox+65+j*90; g.rect(x,200,62,90,PALE,INDIGO,8)
            g.text(x+31,255,f"L{j+1}","s")
            if j<5:
                width=(9-j*1.5) if grow else (2+j*1.5)
                g.line(x+75,245,x+89,245,"axis",color,max(1,width))
        g.text(ox+335,330,"gradient travels backward  ←","b")
        axes(g,ox+80,370,510,170,"","")
        d=(f"M{ox+90} 420 C{ox+240} 440 {ox+390} 460 {ox+575} 485" if not grow else f"M{ox+90} 510 C{ox+300} 500 {ox+400} 470 {ox+470} 430 L{ox+525} 380 L{ox+575} 370")
        g.path(d,color,5); g.text(ox+335,590,"slow learning" if not grow else "loss spike / NaN","s")
    g.save("fig-3-8-gradient-pathologies.svg")

def schedules():
    g=SVG(1400,700,"Conceptual learning-rate schedules: constant, step decay, cosine, and warmup plus cosine")
    g.text(700,48,"Learning Rate Schedules","t"); axes(g,95,110,1210,480,"epoch","learning rate")
    g.path("M110 200 L1290 200",NAVY,4)
    g.path("M110 180 L450 180 L450 310 L810 310 L810 440 L1290 440",INDIGO,4)
    pts=[]
    for i in range(81):
        x=110+i*(1180/80); y=180+300*(1-math.cos(math.pi*i/80))/2; pts.append(f'{x:.1f} {y:.1f}')
    g.path('M'+' L'.join(pts),BLUE,5)
    pts=[]
    for i in range(81):
        x=110+i*(1180/80); y=480-300*(i/12) if i<12 else 180+300*(1-math.cos(math.pi*(i-12)/68))/2; pts.append(f'{x:.1f} {y:.1f}')
    g.path('M'+' L'.join(pts),CYAN,5)
    for yy,label,color in [(130,"constant",NAVY),(157,"step decay",INDIGO),(184,"cosine annealing",BLUE),(211,"warmup + cosine",CYAN)]: g.line(930,yy,980,yy,"axis",color,4); g.text(995,yy+6,label,"s",anchor="start")
    g.text(700,650,"Schedule shapes are conceptual; no validation-accuracy values are implied.","s")
    g.save("fig-3-9-learning-rate-schedules.svg")

def mipds():
    g=SVG(1600,900,"MIPDS vision training engine with augmentation, model, loss, optimizer, validation, schedule, and checkpoints")
    g.text(800,48,"MIPDS Week 3: Structure + Training Engine","t")
    # Training lane
    g.rect(35,90,1530,470,LIGHT,GRAY); g.text(70,128,"TRAINING-TIME SYSTEM","h",anchor="start")
    boxes=[(75,230,210,110,"TRAINING IMAGES","labeled mini-batches",LIGHT,SLATE),(345,230,230,110,"DATA AUGMENTATION","flip · crop · color jitter","#DDF4F8",CYAN),(640,200,260,170,"VISION MODEL","convolutional features\nDropout after dense layers",PALE,INDIGO),(970,200,220,170,"CROSS-ENTROPY LOSS","class-weighted if needed",LIGHT,SLATE),(1260,200,230,170,"ADAM / ADAMW","updates model weights",PALE,INDIGO)]
    for x,y,w,h,a,b,fill,stroke in boxes:
        g.rect(x,y,w,h,fill,stroke); g.text(x+w/2,y+40,a,"h")
        for k,line in enumerate(b.split('\n')): g.text(x+w/2,y+73+k*24,line,"s")
    for a,b in zip(boxes,boxes[1:]): g.line(a[0]+a[2],a[1]+a[3]/2,b[0]-10,b[1]+b[3]/2)
    g.path("M1375 390 C1375 490 770 500 770 380",NAVY,3,dash='stroke-dasharray="10 7" marker-end="url(#arrow)"')
    g.rect(1110,420,270,90,"#DDF4F8",CYAN); g.text(1245,455,"LR SCHEDULE","h"); g.text(1245,485,"3–5 epoch warmup + cosine","s"); g.line(1245,420,1375,380)
    # Validation lane
    g.rect(35,595,1530,235,WHITE,GRAY); g.text(70,635,"VALIDATION + MODEL SELECTION","h",anchor="start")
    lower=[(170,"VALIDATION SET","held-out data"),(550,"VALIDATION LOOP","monitor generalization"),(950,"CHECKPOINT MANAGER","save best validation state"),(1360,"DEPLOYABLE MODEL","selected checkpoint")]
    for x,a,b in lower: g.rect(x-135,680,270,95,LIGHT if x!=925 else "#DDF4F8",SLATE if x!=925 else CYAN); g.text(x,717,a,"h"); g.text(x,748,b,"s")
    for a,b in zip(lower,lower[1:]): g.line(a[0]+135,727,b[0]-145,727)
    g.path("M770 595 L770 550 L770 380",SLATE,3,dash='stroke-dasharray="9 7" marker-end="url(#arrow)"')
    g.text(800,865,"Solid arrows: data/control flow · Dashed arrows: feedback or parameter update","s")
    g.save("fig-3-10-mipds-training-engine.svg")

for fn in [timeline,hierarchy,landscape,training_loop,learning_rates,batch_paths,training_curves,dropout,gradients,schedules,mipds]: fn()
print(f"Generated 11 SVG figures in {OUT}")
