from pathlib import Path
import zipfile,json,hashlib,difflib
BASE=Path(__file__).resolve().parent;R=BASE/'checkpoint-tail038_20261003';S=R/'source';R.mkdir(exist_ok=False)
z=zipfile.ZipFile(BASE/'checkpoint038_20261003/upstream.zip');prefix=z.namelist()[0].split('/')[0]+'/'
for name in z.namelist():
    rel=name.removeprefix(prefix)
    if not rel or name.endswith('/') or rel.startswith('.github/workflows/'):continue
    p=S/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
p=S/'src/program/generate.cpp';original=p.read_text(encoding='utf-8');s=original
needle='    int64_t prompt_cache_every = 16384;';assert s.count(needle)==1
s=s.replace(needle,needle+'\n    bool prompt_cache_tail = false;   // optional extra checkpoint at an existing near-tail chunk boundary')
needle='        else if (a == "--prompt-cache-every")';at=s.index(needle)
s=s[:at]+'        else if (a == "--prompt-cache-tail") o.prompt_cache_tail = true;\n'+s[at:]
needle='        int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;';assert s.count(needle)==1
s=s.replace(needle,needle+'\n        bool pp_tail_saved = false;')
needle='            if (o.prompt_cache_every > 0 && done >= pp_next_check) {';assert s.count(needle)==1
s=s.replace(needle,'''            const bool periodic_checkpoint = o.prompt_cache_every > 0 && done >= pp_next_check;
            // Keep the prefill's existing chunk geometry. One extra near-tail checkpoint
            // can serve a branch whose shared prefix ends before the final cached state.
            const bool tail_checkpoint = o.prompt_cache_tail && o.prompt_cache > 0 && o.prompt_cache_every > 0 &&
                !multi_gpu && !pp_tail_saved && pp_total - done > std::max<int64_t>(1, o.short_read) &&
                pp_total - done <= o.prefill_chunk;
            if (periodic_checkpoint || tail_checkpoint) {''')
needle='                pp_next_check = done + o.prompt_cache_every;';assert s.count(needle)==1
s=s.replace(needle,'''                if (periodic_checkpoint) pp_next_check = done + o.prompt_cache_every;
                if (tail_checkpoint) pp_tail_saved = true;''')
needle='            pp_next_check = reread_to > 0 ? INT64_MAX : resume + o.prompt_cache_every;';assert s.count(needle)==1
s=s.replace(needle,needle+'\n            pp_tail_saved = reread_to > 0;')
needle='                 "  --turn-token ID';at=s.index(needle)
s=s[:at]+'''                 "  --prompt-cache-tail  --serve, single GPU: one extra checkpoint near the prompt's end, at an existing\\n"
                 "                       chunk boundary (default off; requires --prompt-cache and --prompt-cache-every > 0)\\n"
'''+s[at:]
p.write_text(s,encoding='utf-8',newline='\n')
patch=''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/src/program/generate.cpp',tofile='b/src/program/generate.cpp'))
(R/'checkpoint-tail.patch').write_text(patch,encoding='utf-8',newline='\n')
(R/'preparation.json').write_text(json.dumps({'upstream':'99f3dbd0b21d1401b3769e0c0d963913607f380b','patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),'scope':'new optional one-file checkpoint alternative; no change to Prefill kernels or chunk partition; not the original Daily grid patch','original_grid_author':'Claude Opus 5.5','tail_alternative_author':'Codex','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2))
print('Tail checkpoint candidate prepared; no Prefill or engine changes')
