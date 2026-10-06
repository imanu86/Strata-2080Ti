from pathlib import Path
import shutil,subprocess,json,hashlib,sys
BASE=Path(__file__).resolve().parent;R=BASE/('elastic-tail038_final_20261003' if '--final' in sys.argv else 'elastic-tail038_20261003');R.mkdir(exist_ok=False)
shutil.copytree(BASE/'elastic038_20261003/source',R/'source',ignore=shutil.ignore_patterns('*.rej','__pycache__'))
p=subprocess.run(['git','-c','core.autocrlf=false','apply',str(BASE/'checkpoint-tail038_20261003/checkpoint-tail.patch')],cwd=R/'source',capture_output=True,text=True);assert p.returncode==0,p.stderr
files={}
for p in (R/'source').rglob('*'):
    if p.is_file():files[str(p.relative_to(R/'source')).replace('\\','/')]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'preparation.json').write_text(json.dumps({'scope':'integration of the two separately prepared candidates; no Daily edit or branch/worktree','upstream':'99f3dbd0b21d1401b3769e0c0d963913607f380b','elastic_patch_sha256':hashlib.sha256((BASE/'elastic038_20261003/elastic.patch').read_bytes()).hexdigest(),'tail_patch_sha256':hashlib.sha256((BASE/'checkpoint-tail038_20261003/checkpoint-tail.patch').read_bytes()).hexdigest(),'source_files':files},indent=2))
print('Integration source prepared without conflicts')
