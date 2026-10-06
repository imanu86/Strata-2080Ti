from pathlib import Path
import json,hashlib,subprocess,shutil,zipfile,difflib,datetime,sys
BASE=Path(__file__).resolve().parent
REPO=Path(r'C:\Users\imanu\source\repos\moe-aggressive-commit')
DOC=REPO/'docs/porto/strata_adattivo'
E=BASE/'elastic038_20261003'; T=BASE/'checkpoint-tail038_20261003'; M=BASE/'elastic-tail038_final_20261003'; OLD=BASE/'checkpoint038_20261003'
UP='99f3dbd0b21d1401b3769e0c0d963913607f380b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def git(*a):return subprocess.check_output(['git',*a],cwd=REPO,text=True).strip()
assert git('branch','--show-current')=='research/ds4-iq1-subbit-tier-planner'
print(git('status','--short','--branch'))
for r in (E/'parity-r1',T/'parity-r1'):
    d=read(r/'results.json');assert d['status']=='completed' and d['gate_pass'] and d['oracle_pass']
assert read(E/'final-sanity/results.json')['bitwise']
assert read(M/'dialogue-r1/results.json')['long_decode_pass']
assert read(T/'timing-r1/results.json')['oracle_pass']
# Preserve exactly the revision used by the full elastic numeric gate.
z=zipfile.ZipFile(OLD/'upstream.zip');prefix=z.namelist()[0].split('/')[0]+'/'
meta=read(E/'adaptation.json');v1=E/'apply-check-r3';patch='';v1hash={}
for rel in meta['files']:
    p=v1/rel
    if not p.exists():continue
    a=z.read(prefix+rel).decode('utf-8') if prefix+rel in z.namelist() else ''
    b=p.read_text(encoding='utf-8');v1hash[rel]=sha(p)
    patch+=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+rel if prefix+rel in z.namelist() else '/dev/null',tofile='b/'+rel))
assert v1hash['src/program/generate.cpp']=='5b388b9713283ee32f88d2b0f0a5f6677084eb6ad40e1ee3bdd3793b8d647288'
(E/'elastic-full-gate-v1.patch').write_text(patch,encoding='utf-8',newline='\n')
(E/'full-gate-v1.json').write_text(json.dumps({'upstream':UP,'files':v1hash,'patch_sha256':sha(E/'elastic-full-gate-v1.patch'),'shipping':read(E/'qa-build.json'),'final_shipping':read(E/'qa-build-final.json'),'revision_limit':'Full 2560-row gate uses V1. Final CLI/mode/default refinements are checked by 42 CLI cases, CTest, unchanged numeric-body comparison and 64-row/fresh/follow bitwise sanity. V1 unit-test text erroneously labeled its largest successful mapping 8 MiB; final test corrects this to 6 MiB.'},indent=2))
OUT=DOC/'corse_2080ti/20261003_elastica_checkpoint_tail';OUT.mkdir(exist_ok=False,parents=True)
(OUT/'.gitattributes').write_text('* -text\n',encoding='utf-8',newline='\n')
files={}
def copy(src,rel):
    target=OUT/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target);files[rel]=sha(target)
def selected_tree(root,dest):
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix in ('.json','.jsonl','.log'):
            assert p.stat().st_size<10*2**20,p
            copy(p,str(Path(dest)/p.relative_to(root)))
for r,label in ((E,'elastic'),(T,'tail'),(M,'integration')):
    for p in r.iterdir():
        if p.is_file() and p.suffix in ('.json','.patch','.log'):copy(p,str(Path(label)/p.name))
    for sub in ('parity-r1','final-sanity','timing-r1','operational-r1','dialogue-r1'):
        if (r/sub).exists():selected_tree(r/sub,str(Path(label)/sub))
for name in ('fresh.ids.json','follow.ids.json','teacher_tail.ids.json','results.json'):
    copy(OLD/'model-tests-r3'/name,'fixture-original-grid/'+name)
for name in ('qa-readout-A.patch','dependency-pinned-check.json'):
    if (OLD/name).exists():copy(OLD/name,'baseline/'+name)
copy(Path(r'C:\Users\imanu\source\repos\Strata-2080Ti\tools\sm75\validate_model.py'),'scripts/validate_model.py')
for p in BASE.glob('*.py'):
    if any(x in p.name for x in ('elastic038','elastic_tail038','checkpoint_tail038')):copy(p,'scripts/'+p.name)
license_names=[n for n in z.namelist() if n.startswith(prefix) and n[len(prefix):] in ('LICENSE','LICENSE.md','LICENSE.txt')]
assert len(license_names)==1
(OUT/'LICENSE-Strata.txt').write_bytes(z.read(license_names[0]));files['LICENSE-Strata.txt']=sha(OUT/'LICENSE-Strata.txt')
copy(BASE/'elastic-tail-final-state.json','final-state.json')
manifest={'upstream':UP,'ggml_pin':'3cf03257f219afbe7334045ff7c6a06ac68c627d','generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'omitted':'Private multi-GB raw logits, executables, libraries, model weights, original download archive and CUDA cache are not published. Raw-logit hashes/lengths are recorded in results. No artificial saturation load.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(OUT/'README.md').write_text('''# Reproduction and provenance

The patches target Strata 0.1.38 commit `99f3dbd0b21d1401b3769e0c0d963913607f380b` and GGML pin `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Apply with `git -c core.autocrlf=false apply`. `elastic/elastic.patch` and `tail/checkpoint-tail.patch` are independent shipping contributions; QA readout patches are diagnostics only.

The full elastic numeric gate used `elastic-full-gate-v1.patch`; subsequent CLI/help, supported-mode and opt-in wait refinements are recorded separately and checked on the final pure build. The initial incorrect unit-test allocation label is preserved and explained in `full-gate-v1.json`.

Scripts preserve the workstation paths and run folders used for the recorded tests. For reproduction, adapt paths, install NumPy/psutil, obtain the same model/tokenizer/pack/profile locally, recreate the pinned plain source exports and build artifacts, and use a fresh output folder (runners deliberately refuse overwriting evidence). `validate_model.py` supplies the corpus and local numerical gate; its CLI is not used by these runners. Token ID fixtures and complete result JSON are included. The original grid results are retained as a failed reference, not as a passing contribution.

Operational results contain an initial malformed dialogue attempt whose 131k/250k guide outputs stopped after 48 tokens. The corrected `dialogue-r1` includes assistant turn delimiters, stronger long-output instructions and distinct subsequent user markers; both long outputs reach 1024 tokens. This does not establish that missing delimiters alone caused the first result.

No weights, executables or multi-GB raw logits are included. Raw dump hashes and sizes remain in the result metadata; `manifest.json` attests the exact published bytes. This is local regression evidence on a modified RTX 2080 Ti 22 GB, not a universal upstream quality standard or a Daily speed record.
''',encoding='utf-8',newline='\n')
report='''# Port elastica e checkpoint tail — 3 ottobre 2026

Le due contribuzioni sono preparate separatamente su upstream 0.1.38
(`99f3dbd0b21d1401b3769e0c0d963913607f380b`). Il gate locale è passato.
Le nuove PR saranno aperte dopo l'autorizzazione dei due branch prevista dalle
regole personali; il Daily operativo resta invariato.

## Cosa proponiamo

- **Elastica CUDA opt-in:** port della cache Daily originariamente realizzata
  da Claude Opus 5.5, con core fisso, coda ridimensionabile e indirizzi VMM
  stabili. Codex ha isolato il contributo e adattato il contesto 0.1.38,
  il link al driver CUDA e i token buffer al massimo chunk di prefill.
  CLI, layer split, peer/remote e resident CPU complement conservano la cache
  fissa; HIP ha fallback fisso, senza attestazione di compilazione/hardware AMD.
- **`--prompt-cache-tail` opt-in:** nuova alternativa Codex che salva al massimo
  un checkpoint aggiuntivo su un confine di chunk già raggiunto. Riutilizza la
  cache limitata esistente e non taglia i chunk. La patch checkpoint-grid del
  Daily conserva il FAIL numerico documentato: questa proposta non la dichiara
  corretta e non la sostituisce automaticamente nell'operativo.

## Parità e correttezza

Macchina: RTX 2080 Ti modificata a 22 GB, Ryzen 5800X, 96 GB RAM;
CUDA 12.6/MSVC 14.44. GGML pin `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
KV int8, capacità 262144 e 32768 residenti. Le prove isolate mantengono
5357 slot effettivi, geometria e residenza fisse; prompt fresh/follow circa
33,8k, speculative window 4, suffix draft 0, PCIe fraction 0.28.

| Contributo | Confronto | Teacher rows per braccio | Posizioni valutate | Esito |
|---|---|---:|---:|---|
| Elastica | upstream / port off / VMM stabile / upstream ripetuto | 2560 | 2557 | Tutti i logits bitwise, KL 0, top1 100% |
| Checkpoint tail | upstream / tail / upstream ripetuto | 2560 | 2557 | Tutti i logits bitwise, KL 0, top1 100% |

Corpora completi: codice 512, documento 1024, chat 1024. NLL/PPL identici;
anche i riferimenti ripetuti e le prime righe fresh/follow sono bitwise.
Oracoli completi: 8/8 per elastica e 6/6 per tail, su fixture ripetute;
non sono una misura generale della qualità del modello.

La parità dell'elastica isola l'arena mantenendo disattivati i resize tramite
intervalli molto lunghi. Le modifiche di residenza o chunk in auto possono
modificare gli arrotondamenti: il comportamento dinamico è verificato a parte.
Il gate completo usa la revisione V1 conservata. La versione finale aggiunge
parsing/help, esclusione dei tier non supportati e attesa aggiuntiva prima del
prestito solo quando l'elastica è attiva. Il corpo numerico resta verificato
identico salvo questa guardia, e il binario finale passa fresh/follow più
64 teacher rows bitwise contro V1. Non attribuire il gate completo direttamente
a un eseguibile diverso senza questa distinzione di provenienza.

Elastica finale: CTest **3/3**, inclusi **35 controlli CUDA VMM** su crescita,
shrink/regrowth, indirizzi/offset stabili, byte preservati, nuovi slot azzerati,
richieste invalide e riapertura fissa; **42 casi CLI** passati. La mappatura
massima del test è 6 MiB (l'etichetta iniziale 8 MiB è corretta nella revisione
finale). Non è una prova del rollback per OOM parziale del driver.

## Misura checkpoint tail

Tre coppie intercalate A1/B1, B2/A2, A3/B3, binari shipping, tutte le 12 risposte
complete corrette. Chunk 6144, intervallo checkpoint 16384, prefisso condiviso
32769 token: riuso **18432 → 30720**.

| Tempo al primo token | Mediana upstream | Mediana tail | Differenza |
|---|---:|---:|---:|
| Prompt fresco | 34,178 s | 34,497 s | +0,93% |
| Richiesta ramificata | 17,047 s | 4,470 s | −73,78% |

Spread dei riferimenti: 0,90% fresh e 1,91% follow. È una misura di questa
ramificazione da 33,8k, non un incremento universale né una nuova SOTA t/s.

## Prova combinata e conversazione profonda

Elastica auto + tail, capacità KV262144, prompt a 20k, 131072 e 250000:
**9/9 risposte di richiamo corrette**, primi logits finiti, crescita e shrink
reali della cache e relayout a chunk8192 osservati. Nessun carico artificiale.
Startup: 6432 slot, core3190/tail3242; finestra effettiva6, MTPmax4/lookup3,
PCIe fraction0.36. Queste impostazioni non equivalgono alle prove isolate.
La riserva500MiB è un obiettivo della policy, non una garanzia Windows.

Nel primo tentativo operativo i due prompt guide a131k/250k omettevano la
risposta assistant completa e il delimitatore di turno: uscite48token stop,
**non** prove long-decode. Il risultato è conservato in `operational-r1`.
La successiva sequenza `dialogue-r1` include le risposte assistant chiuse,
istruzioni lunghe rinforzate e un marcatore utente nuovo prima/dopo il decode:
**1024token/length a entrambe le profondità e 6/6 oracoli completi**,
inclusi i quattro distinti marcatori. Non è dimostrato che il delimitatore,
da solo, spieghi il risultato precedente.

Il binario combinato precede soltanto le ultime guardie peer/default dell'elastica;
il percorso single-GPU con flag attivo provato resta equivalente. AMD,
multi-GPU e 2080 Ti stock11GB non sono validati. I test non sostituiscono una
attestazione generale di qualità o di ogni possibile ridimensionamento.

## Evidenze e stato operativo

[Patch, script, log, fixture e manifest](corse_2080ti/20261003_elastica_checkpoint_tail/README.md).
I dump grezzi multi-GB restano privati; SHA e lunghezze sono nei risultati.
Tutti i fallimenti precedenti restano nei rispettivi report, senza essere
riclassificati come pass. SHA del Daily, server, configurazione, profilo e
lanciatore sono verificati invariati; motori di test chiusi.
Non installare la build upstream pura come SOTA: non contiene tutte le altre
patch SM75 del Daily. La promozione operativa va fatta integrando e verificando
il contributo nel suo ramo reale.
'''
(DOC/'PORT_ELASTICA_CHECKPOINT_TAIL_3_OTTOBRE.md').write_text(report,encoding='utf-8',newline='\n')
h=DOC/'HANDOFF.md';shutil.copyfile(h,BASE/'HANDOFF-before-elastic-tail038.md')
s=h.read_text(encoding='utf-8');at=s.index('\n')+1
s=s[:at]+'''
- **3 ottobre, port upstream038 elastica + nuovo checkpoint tail validati:**
  contributi separati; elastica A/off/VMM/A e tail A/B/A con2560teacher rows
  (2557posizioni) per braccio, tutti bitwise/KL0. Elastica35test CUDA/CTest3/3,
  CLI42; ultima revisione64teacher+fresh/follow bitwise contro V1.
  Tail:3coppie33.8k, TTFT branch17.047→4.470s(−73.78%), fresh+0.93%;
  non nuovaSOTA e non fix della grid originale, che conservaFAIL.
  Autoelastica+tail:9recall corretti, grow/shrink/relayout8192 reali.
  Conversazione chiusa correttamente a131k/250k:1024token entrambe,
  6oracoli inclusi4marcatori distinti; tentativo precedente48token conservato.
  Daily invariato e motori chiusi. PR pronte tecnicamente; autorizzazione dei
  due branch richiesta secondo AGENTS prima della creazione.
  [Misure, provenienza e limiti](PORT_ELASTICA_CHECKPOINT_TAIL_3_OTTOBRE.md).
'''+s[at:]
h.write_text(s,encoding='utf-8',newline='\n')
print('Published files',len(files),'bytes',sum((OUT/p).stat().st_size for p in files))
