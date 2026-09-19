"""Verify raw author bytes and file set before and after any publication operation."""
from pathlib import Path
import csv,hashlib,argparse,json
R=Path(__file__).resolve().parents[1]
def verify(ci=False):
 rows=list(csv.DictReader((R/'editorial/ORIGINAL-MANUSCRIPT-LOCK.csv').open()))
 expected={r['relative_path']:r for r in rows if not(ci and Path(r['relative_path']).name=='.DS_Store')}
 actual={str(p.relative_to(R)):p for p in (R/'original').rglob('*') if p.is_file() and not(ci and p.name=='.DS_Store')}
 errors=[]
 for name,row in expected.items():
  p=actual.get(name)
  if p is None:errors.append('Missing: '+name);continue
  b=p.read_bytes()
  if len(b)!=int(row['byte_size']) or hashlib.sha256(b).hexdigest()!=row['sha256']:errors.append('Changed: '+name)
 errors+=['Added: '+name for name in actual.keys()-expected.keys()]
 result={'status':'FAIL' if errors else 'PASS','files_checked':len(expected),'changed_original_files':len(errors),'errors':errors,'scope':'Git publication excludes only .DS_Store' if ci else 'Entire author-supplied directory'}
 if errors:raise RuntimeError(json.dumps(result))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ci',action='store_true');args=p.parse_args();print(json.dumps(verify(args.ci),indent=2))
