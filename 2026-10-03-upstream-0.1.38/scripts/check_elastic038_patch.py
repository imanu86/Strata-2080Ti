from pathlib import Path
import zipfile,json,hashlib,subprocess,sys
BASE=Path(__file__).resolve().parent;R=BASE/'elastic038_20261003';dest=R/('apply-check-publish' if '--publish' in sys.argv else 'apply-check-final' if '--final' in sys.argv else 'apply-check-r3');dest.mkdir(exist_ok=False)
z=zipfile.ZipFile(BASE/'checkpoint038_20261003/upstream.zip');prefix=z.namelist()[0].split('/')[0]+'/'
meta=json.loads((R/'adaptation.json').read_text())
for rel in meta['files']:
    if prefix+rel in z.namelist():
        p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(prefix+rel))
p=subprocess.run(['git','-c','core.autocrlf=false','apply','--check',str(R/'elastic.patch')],cwd=dest,capture_output=True,text=True);assert p.returncode==0,p.stderr
p=subprocess.run(['git','-c','core.autocrlf=false','apply',str(R/'elastic.patch')],cwd=dest,capture_output=True,text=True);assert p.returncode==0,p.stderr
for rel,want in meta['files'].items():assert hashlib.sha256((dest/rel).read_bytes()).hexdigest()==want,rel
(R/'patch-apply-check.json').write_text(json.dumps({'pristine_upstream':meta['upstream'],'patch_sha256':hashlib.sha256((R/'elastic.patch').read_bytes()).hexdigest(),'all_applied_file_hashes_match':True,'files':len(meta['files'])},indent=2))
print('Patch applies to pristine pinned upstream; all resulting hashes match')
