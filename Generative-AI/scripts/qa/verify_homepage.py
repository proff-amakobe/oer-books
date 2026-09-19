#!/usr/bin/env python3
"""Check the W1 landing page locally or at a supplied deployed base URL."""
from pathlib import Path
import argparse,functools,http.server,threading,json,hashlib,urllib.parse
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--base');args=p.parse_args()
out=R/'output/reset/qa/w1';out.mkdir(parents=True,exist_ok=True)
server=None
if args.base:base=args.base
else:
 class Quiet(http.server.SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(R/'output/reset/html')))
 threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_address[1]}/'
result={'base':base,'viewports':[]}
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
 page=b.new_page(viewport={'width':1440,'height':1000});assert page.goto(base,wait_until='networkidle').status==200
 assert page.locator('.book-hero h1').inner_text()=='Generative AI'
 assert page.locator('.book-part').count()==5
 assert page.locator('.book-part li a').count()==15
 assert page.locator('.book-cover img').evaluate('(x)=>x.complete&&x.naturalWidth>0')
 cover=page.locator('.book-cover img').get_attribute('src');raw=page.request.get(urllib.parse.urljoin(base,cover)).body()
 assert raw==(R/'assets/cover/generative-ai-cover.png').read_bytes()
 for width in [1440,1024,768,390]:
  page.set_viewport_size({'width':width,'height':1000})
  for mode in ['light','dark']:
   if page.locator('body').evaluate('(x)=>x.classList.contains("quarto-dark")') != (mode=='dark'):
    page.evaluate('window.quartoToggleColorScheme()')
   page.wait_for_timeout(350)
   assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+2'),(width,mode)
   assert page.locator('.book-cover img').is_visible()
   contrast=page.locator('.book-landing a').evaluate_all(r"""xs=>xs.map(x=>{const rgb=s=>s.match(/[\d.]+/g).slice(0,3).map(Number);const lum=c=>c.map(v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4}).reduce((s,v,i)=>s+v*[.2126,.7152,.0722][i],0);let n=x,bg;while(n){bg=getComputedStyle(n).backgroundColor;if(bg!=='rgba(0, 0, 0, 0)'&&bg!=='transparent')break;n=n.parentElement;}let a=lum(rgb(getComputedStyle(x).color)),b=lum(rgb(bg));return {text:x.textContent,ratio:(Math.max(a,b)+.05)/(Math.min(a,b)+.05)};})""")
   assert all(x['ratio']>=4.5 for x in contrast),contrast
   assert page.locator('.book-cover img').get_attribute('alt')

   assert all(page.locator('.book-button').nth(i).bounding_box()['width']<=width for i in range(3))
   page.screenshot(path=str(out/f'{"live" if args.base else "local"}-{width}-{mode}.png'),full_page=True)
   result['viewports'].append({'width':width,'mode':mode,'status':'PASS'})
 page.locator('.book-button').first.focus();assert page.locator('.book-button').first.evaluate('(x)=>getComputedStyle(x).outlineStyle')!='none'
 links=page.locator('.book-landing a').evaluate_all('(xs)=>xs.map(x=>({href:x.href,text:x.textContent}))')
 result['links_checked']=0
 for link in links:
  if link['href'].startswith(base):
   res=page.request.get(link['href']);assert res.status==200,link
   if '.pdf' in link['href']:assert res.body().startswith(b'%PDF')
   if '.epub' in link['href']:assert res.body().startswith(b'PK')
   result['links_checked']+=1
 assert any(x['href']=='https://github.com/proff-amakobe/oer-books/tree/main/Generative-AI' for x in links)
 for name in ['og:title','og:description','og:image','twitter:card','twitter:image']:
  el=page.locator(f'meta[property="{name}"],meta[name="{name}"]');assert el.count()==1 and el.get_attribute('content'),name
 assert page.locator('meta[property="og:image"]').get_attribute('content')=='https://proff-amakobe.github.io/oer-books/Generative-AI/assets/cover/generative-ai-cover.png'
 assert page.locator('link[rel=canonical]').get_attribute('href')=='https://proff-amakobe.github.io/oer-books/Generative-AI/'
 data=json.loads(page.locator('script[type="application/ld+json"]').inner_text());assert data['@type']=='Book' and data['bookEdition']=='First Open Edition' and 'isbn' not in data
 page.get_by_role('link',name='Read Online',exact=True).click();assert page.url.endswith('original/01-foundations.html')
 page.goto(base+'preface.html',wait_until='networkidle');assert 'Learning by Designing' in page.locator('main').inner_text()
 b.close()
if server:server.shutdown()
result['status']='PASS';(out/('live.json' if args.base else 'local.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
