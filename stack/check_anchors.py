#!/usr/bin/env python3
"""The fork's patch stack: build the registry, or check it against a new upstream.

The stack is the ordered set of patches this fork carries on top of upstream Strata (see INDEX.md).  Each patch is one
lever: one `NN-name.patch` (git format-patch, applicable with `git am` or `git apply --3way`), one entry in
`manifest.json` (purpose, switch, origin, upstream status, measured value) and, derived from the patch itself, one
anchor per insertion point: 2-3 lines of the *unpatched* file, unique in that file, so a future upstream can be checked
without line numbers.  INDEX.md (short: metadata and anchors) and BLOCKS.md (the inserted blocks, anchor repeated on
top) are generated from manifest.json and the patches; nothing in them is written by hand.

    python stack/check_anchors.py --check <repo> --rev v0.1.44 [--json out.json]
        For every anchor of every patch: present (unique), ambiguous (more than one match), moved (matches only with
        whitespace folded) or missing in that upstream revision.  Then applies the patches in order in a temporary
        worktree of <repo> at <rev>: clean (`git apply --check`), 3-way (`git apply --3way` merged with no conflict) or
        rewrite (conflicts; the patch is skipped and the later ones are tried without it).  Read-only for <repo>'s
        own worktrees: everything happens in a throw-away worktree that is removed at the end.

    python stack/check_anchors.py --build <repo> --range <base>..<head>
        Regenerates the patches from the lever commits in <base>..<head> (commits that touch only stack/, docs/ or a
        README are skipped), in the order of manifest.json's entries, recomputes the anchors from each patch's hunks
        against its parent commit, and rewrites manifest.json (anchors only; the hand-written fields are kept),
        INDEX.md and BLOCKS.md.  Published history is never rewritten: a later fix to a lever is a commit whose subject
        starts with one of the entry's `follow_up_subjects`, and --build squashes it into that lever's patch.

    python stack/check_anchors.py --verify-docs
        Checks that INDEX.md and BLOCKS.md are exactly what manifest.json and the patches generate (CI-style).

Python 3.8+, git on PATH.  No other dependency.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'manifest.json'
INDEX = HERE / 'INDEX.md'
BLOCKS = HERE / 'BLOCKS.md'
ANCHOR_LINES = 3          # context lines kept in an anchor (fewer when the hunk has fewer)
ANCHOR_MAX = 6            # the anchor grows up to this many lines until it is unique in the file
BLOCK_FULL_MAX = 80       # longer hunks are shown truncated in BLOCKS.md (the .patch has them whole)
BLOCK_HEAD = 24


def git(args, cwd, check=True, text=True):
    r = subprocess.run(['git', *args], cwd=str(cwd), capture_output=True, text=text, encoding='utf-8' if text else None,
                       errors='replace' if text else None)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}:\n{r.stderr}")
    return r


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def save_manifest(m):
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


# ----------------------------------------------------------------------------------------------------------------- patches

class Hunk:
    def __init__(self, path, header_symbol, old_start, lines):
        self.path = path                  # the file in the post-image (b/ side)
        self.symbol = header_symbol       # git's funcname text after the @@ ... @@
        self.old_start = old_start
        self.lines = lines                # the hunk body: ' ', '+', '-' lines (without the newline)

    def pre_context_before_change(self):
        """The ' ' lines right before the first changed line (pre-image), as plain text lines."""
        out = []
        for l in self.lines:
            if l[:1] in ('+', '-'):
                break
            out.append(l[1:])
        return out

    def pre_context_after_change(self):
        """The ' ' lines right after the last changed line (pre-image)."""
        last = -1
        for i, l in enumerate(self.lines):
            if l[:1] in ('+', '-'):
                last = i
        return [l[1:] for l in self.lines[last + 1:] if l[:1] == ' ']

    def pre_lines_inside(self):
        """The '-' lines (pre-image text that the hunk rewrites)."""
        return [l[1:] for l in self.lines if l[:1] == '-']


def parse_patch(text):
    """Files and hunks of a format-patch / unified diff.  Returns ([(path, is_new, is_deleted, hunks)], message)."""
    files = []
    msg_end = text.find('\ndiff --git ')
    message = text[:msg_end] if msg_end >= 0 else text
    cur = None
    hunk = None
    for raw in text.splitlines():
        if raw.startswith('diff --git '):
            m = re.match(r'diff --git a/(.*?) b/(.*)$', raw)
            cur = {'path': m.group(2), 'new': False, 'deleted': False, 'hunks': [], 'binary': False}
            files.append(cur)
            hunk = None
            continue
        if cur is None:
            continue
        if raw.startswith('new file mode'):
            cur['new'] = True
        elif raw.startswith('deleted file mode'):
            cur['deleted'] = True
        elif raw.startswith('Binary files') or raw.startswith('GIT binary patch'):
            cur['binary'] = True
        elif raw.startswith('@@ '):
            m = re.match(r'@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@ ?(.*)$', raw)
            hunk = Hunk(cur['path'], m.group(2).strip(), int(m.group(1)), [])
            cur['hunks'].append(hunk)
        elif hunk is not None and raw[:1] in (' ', '+', '-'):
            hunk.lines.append(raw)
        elif hunk is not None and raw.startswith('\\ No newline'):
            pass
        elif raw.startswith('-- ') or raw.startswith('--'):
            # the format-patch trailer ("-- \n2.x") ends the diff
            if re.match(r'^-- ?$', raw):
                cur = None
                hunk = None
    return files, message


def anchor_for(hunk, pre_image_lines):
    """2-3 unique pre-image lines for this hunk: context before the change, grown until unique; else the rewritten
    lines; else the context after.  Returns (lines, status) with status 'unique', 'ambiguous' or 'none'."""
    text_lines = pre_image_lines

    def count(seq):
        if not seq:
            return 0
        n = len(seq)
        c = 0
        for i in range(len(text_lines) - n + 1):
            if text_lines[i:i + n] == seq:
                c += 1
        return c

    before = hunk.pre_context_before_change()
    inside = hunk.pre_lines_inside()
    after = hunk.pre_context_after_change()
    candidates = []
    # context before the change, last k lines (skip blank-only tails), k = ANCHOR_LINES..ANCHOR_MAX
    b = list(before)
    while b and not b[-1].strip():
        b.pop()
    for k in range(min(ANCHOR_LINES, len(b)), min(ANCHOR_MAX, len(b)) + 1):
        if k >= 1:
            candidates.append(b[-k:])
    # the rewritten lines themselves (a replacement hunk: the lines that go away are a fine anchor for "where")
    ins = [l for l in inside if l.strip()]
    for k in range(min(ANCHOR_LINES, len(ins)), min(ANCHOR_MAX, len(ins)) + 1):
        if k >= 1:
            candidates.append(ins[:k])
    # before + inside together
    if b and ins:
        candidates.append(b[-2:] + ins[:2])
    a = [l for l in after if l.strip()]
    for k in range(min(ANCHOR_LINES, len(a)), min(ANCHOR_MAX, len(a)) + 1):
        if k >= 1:
            candidates.append(a[:k])
    best = None
    for c in candidates:
        n = count(c)
        if n == 1:
            return c, 'unique'
        if n > 1 and best is None:
            best = c
    if best is not None:
        return best, 'ambiguous'
    if candidates:
        return candidates[0], 'none'
    return [], 'none'


def lever_commits(repo, rng, skip_subjects=()):
    """The commits of <base>..<head> that touch something outside stack/, oldest first; a commit whose subject starts
    with one of manifest.json's `skip_subjects` (documentation-only commits of the lab) is left out too."""
    out = []
    for sha in git(['rev-list', '--reverse', '--no-merges', rng], repo).stdout.split():
        files = git(['show', '--format=', '--name-only', sha], repo).stdout.split('\n')
        files = [f for f in files if f.strip()]
        subject = git(['log', '-1', '--format=%s', sha], repo).stdout.strip()
        if any(subject.startswith(s) for s in skip_subjects):
            continue
        # a lever changes code: a commit that touches only stack/, docs/ or a README is documentation, not a lever
        if any(not (f.startswith('stack/') or f.startswith('docs/') or f.split('/')[-1].lower().startswith('readme'))
               for f in files):
            out.append(sha)
    return out


def build(repo, rng):
    repo = Path(repo).resolve()
    m = load_manifest()
    entries = m['patches']
    # follow-ups: commits whose subject starts with an entry's `follow_up_subjects` belong to that lever, not to a new one
    follow_ups = {}
    fu_prefixes = []
    for e in entries:
        for pref in e.get('follow_up_subjects', []):
            fu_prefixes.append((pref, e['name']))
    all_commits = lever_commits(repo, rng, tuple(m.get('skip_subjects', [])))
    commits = []
    for sha in all_commits:
        subject = git(['log', '-1', '--format=%s', sha], repo).stdout.strip()
        owner = next((name for pref, name in fu_prefixes if subject.startswith(pref)), None)
        if owner:
            follow_ups.setdefault(owner, []).append(sha)
        else:
            commits.append(sha)
    if len(commits) != len(entries):
        raise SystemExit(f'{len(commits)} lever commits in {rng} but {len(entries)} manifest entries:\n' +
                         '\n'.join(f'  {s[:8]} ' + git(['log', '-1', '--format=%s', s], repo).stdout.strip()[:100]
                                   for s in commits))
    for old in HERE.glob('*.patch'):
        old.unlink()
    head = git(['rev-parse', rng.split('..')[1]], repo).stdout.strip()
    base = git(['rev-parse', rng.split('..')[0]], repo).stdout.strip()
    m['base']['commit'] = base
    m['stack_head'] = head
    for e, sha in zip(entries, commits):
        fname = f"{e['id']}-{e['name']}.patch"
        e['patch'] = fname
        e['commit'] = sha
        e['title'] = git(['log', '-1', '--format=%s', sha], repo).stdout.strip()
        patch_sha = sha
        if follow_ups.get(e['name']):
            # published history is never rewritten: a later fix to a lever is a follow-up commit, squashed here into the
            # lever's patch (cherry-picked onto the lever commit in a throw-away worktree)
            tmp = Path(tempfile.mkdtemp(prefix='stack-squash-'))
            wt = tmp / 'wt'
            try:
                git(['worktree', 'add', '--detach', '-q', str(wt), sha], repo)
                # the lever's author and dates, so that rebuilding the stack leaves an unchanged patch unchanged
                an, ae, ad = git(['log', '-1', '--format=%an%n%ae%n%aI', sha], repo).stdout.strip().split('\n')
                env = dict(os.environ, GIT_AUTHOR_NAME=an, GIT_AUTHOR_EMAIL=ae, GIT_AUTHOR_DATE=ad,
                           GIT_COMMITTER_NAME=an, GIT_COMMITTER_EMAIL=ae, GIT_COMMITTER_DATE=ad)
                for f in follow_ups[e['name']]:
                    r = subprocess.run(['git', 'cherry-pick', '-n', '-X', 'patience', f], cwd=str(wt), capture_output=True, text=True)
                    if r.returncode:
                        raise SystemExit(f"follow-up {f[:8]} does not squash onto {e['name']} ({sha[:8]}):\n{r.stderr}")
                # one commit from the lever's parent: the lever's diff plus the follow-ups
                git(['reset', '-q', '--soft', sha + '^'], wt)
                msg = git(['log', '-1', '--format=%B', sha], repo).stdout.rstrip() + '\n\nFollow-ups squashed into this patch:\n' + \
                    ''.join(f"- {f[:8]} {git(['log', '-1', '--format=%s', f], repo).stdout.strip()}\n" for f in follow_ups[e['name']])
                subprocess.run(['git', 'commit', '-q', '--no-verify', '-F', '-'], cwd=str(wt), input=msg, text=True,
                               capture_output=True, env=env)
                patch_sha = git(['rev-parse', 'HEAD'], wt).stdout.strip()
                e['follow_up_commits'] = follow_ups[e['name']]
            finally:
                subprocess.run(['git', 'worktree', 'remove', '--force', str(wt)], cwd=str(repo), capture_output=True)
                shutil.rmtree(tmp, ignore_errors=True)
                subprocess.run(['git', 'worktree', 'prune'], cwd=str(repo), capture_output=True)
        text = git(['format-patch', '-1', '--stdout', '--no-stat', '--no-signature', '--zero-commit', patch_sha], repo).stdout
        # the first line "From 000... Mon Sep 17 00:00:00 2001" is deterministic with --zero-commit
        (HERE / fname).write_text(text, encoding='utf-8', newline='\n')
        files, _ = parse_patch(text)
        anchors = []
        n = 0
        parent = sha + '^'
        for f in files:
            if f['new']:
                n += 1
                anchors.append({'block': f"{e['id']}-{n}", 'file': f['path'], 'symbol': '(new file)', 'anchor': [],
                                'status_in_base': 'new-file', 'hunk': 0})
                continue
            if f['deleted']:
                n += 1
                anchors.append({'block': f"{e['id']}-{n}", 'file': f['path'], 'symbol': '(deleted file)', 'anchor': [],
                                'status_in_base': 'deleted-file', 'hunk': 0})
                continue
            pre = git(['show', f"{parent}:{f['path']}"], repo).stdout.split('\n')
            for hi, h in enumerate(f['hunks'], 1):
                n += 1
                lines, st = anchor_for(h, pre)
                anchors.append({'block': f"{e['id']}-{n}", 'file': f['path'], 'symbol': h.symbol or '(file scope)',
                                'anchor': lines, 'status_in_base': st, 'hunk': hi})
        e['anchors'] = anchors
        e['files'] = [f['path'] for f in files]
    save_manifest(m)
    INDEX.write_text(render_index(m), encoding='utf-8', newline='\n')
    BLOCKS.write_text(render_blocks(m), encoding='utf-8', newline='\n')
    amb = sum(1 for e in entries for a in e['anchors'] if a['status_in_base'] == 'ambiguous')
    none = sum(1 for e in entries for a in e['anchors'] if a['status_in_base'] == 'none')
    print(f"{len(entries)} patches, {sum(len(e['anchors']) for e in entries)} insertion points "
          f"({amb} anchors ambiguous in their own base, {none} without a usable anchor); INDEX.md and BLOCKS.md written")


# ----------------------------------------------------------------------------------------------------------------- render

def md_escape(s):
    return s.replace('|', '\\|')


def render_index(m):
    b = m['base']
    out = [f"# {m.get('title', 'Patch stack')}", '',
           m.get('intro', '').rstrip(), '',
           f"Base: upstream `{b['repo']}` **{b['tag']}** (`{b['commit'][:8]}`). "
           f"Stack head: `{m.get('stack_head', '')[:8]}`. "
           f"{len(m['patches'])} patches, applied in this order with `git am stack/*.patch` "
           f"(or one at a time with `git apply --3way`). Check a new upstream with "
           f"`python stack/check_anchors.py --check <repo> --rev <tag>`.", '',
           '## Reading this file', '',
           '- **Switch / default**: how the lever is turned on, and what happens when nothing is set.',
           '- **Origin**: `fork` = written for this fork; `PR #n` = an upstream pull request (ours or somebody else\'s) '
           'carried as a patch while it is open.',
           '- **Upstream**: whether upstream already has it, has an open PR for it, or never saw it.',
           '- **Measured**: the value on this fork\'s hardware (RTX 2080 Ti 22 GB + RTX 3060 12 GB, `--layer-split 16`, '
           'Windows 11, 131k-token forced bench, B-A-B unless said otherwise). Details in the linked docs.',
           '- **Insertion points**: one per hunk: file, the enclosing symbol as git names it, and an anchor of 2-3 lines '
           'of the *unpatched* file, unique in that file at the time the patch was made. No line numbers. The inserted '
           'block is in `BLOCKS.md` under the same id; the applicable form is the `.patch`.', '',
           '## Upstream defaults this fork forces', '',
           m.get('forced_defaults', '').rstrip(), '',
           '## Patches', '']
    out.append('| id | name | what | switch (default) | origin | upstream | measured |')
    out.append('|---|---|---|---|---|---|---|')
    for e in m['patches']:
        out.append(f"| {e['id']} | [{e['name']}](#{e['id']}-{e['name']}) | {md_escape(e['summary'])} | "
                   f"{md_escape(e['switch'])} | {md_escape(e['origin'])} | {md_escape(e['upstream_status'])} | "
                   f"{md_escape(e['measured'])} |")
    out.append('')
    if m.get('retired'):
        out += ['## Retired', '', "Levers that were in the stack and were dropped (code stays on the fork's lab branches):", '',
                '| name | what | measured | why |', '|---|---|---|---|']
        for r in m['retired']:
            out.append(f"| {r['name']} | {md_escape(r['summary'])} | {md_escape(r['measured'])} | {md_escape(r['retired'])} |")
        out.append('')
    for e in m['patches']:
        out += [f"## {e['id']} {e['name']}", '',
                f"`{e['patch']}` — {e['title']}", '',
                f"- **What**: {e['summary']}",
                f"- **Switch (default)**: {e['switch']}",
                f"- **Origin**: {e['origin']}",
                f"- **Upstream**: {e['upstream_status']}",
                f"- **Measured**: {e['measured']}"]
        if e.get('depends_on'):
            out.append(f"- **Depends on**: {e['depends_on']}")
        if e.get('forces'):
            out.append(f"- **Upstream defaults it changes**: {e['forces']}")
        if e.get('notes'):
            out.append(f"- **Notes**: {e['notes']}")
        if e.get('docs'):
            out.append(f"- **Docs**: {e['docs']}")
        out.append(f"- **Files**: {', '.join('`' + f + '`' for f in e.get('files', []))}")
        out.append('')
        out.append(f"Insertion points ({len(e['anchors'])}):")
        out.append('')
        for a in e['anchors']:
            ref = f"[{a['block']}](BLOCKS.md#{a['block'].lower()})"
            if a['status_in_base'] in ('new-file', 'deleted-file'):
                out.append(f"- `{a['file']}` — {a['symbol']} → {ref}")
                continue
            flag = '' if a['status_in_base'] == 'unique' else f" *(anchor {a['status_in_base']} in the base file)*"
            out.append(f"- `{a['file']}` · `{a['symbol'][:90]}`{flag} → {ref}")
            out.append('  ```')
            for l in a['anchor']:
                out.append('  ' + l.rstrip())
            out.append('  ```')
        out.append('')
    return '\n'.join(out).rstrip() + '\n'


def render_blocks(m):
    out = [f"# {m.get('title', 'Patch stack')}: the inserted blocks", '',
           'Generated from the patches by `check_anchors.py --build`. One section per insertion point of `INDEX.md`, '
           'the anchor (unpatched lines) first, then the hunk as `git diff` shows it. Hunks longer than '
           f'{BLOCK_FULL_MAX} lines are cut after {BLOCK_HEAD}: the `.patch` has them whole.', '']
    for e in m['patches']:
        text = (HERE / e['patch']).read_text(encoding='utf-8')
        files, _ = parse_patch(text)
        hunks_by_file = {f['path']: f for f in files}
        out.append(f"## {e['id']} {e['name']}")
        out.append('')
        out.append(f"`{e['patch']}` — {e['title']}")
        out.append('')
        for a in e['anchors']:
            out.append(f"### {a['block']}")
            out.append('')
            out.append(f"`{a['file']}` · {a['symbol']}")
            out.append('')
            if a['status_in_base'] == 'new-file':
                f = hunks_by_file[a['file']]
                n = sum(len(h.lines) for h in f['hunks'])
                out.append(f"New file, {n} lines: see the patch.")
                out.append('')
                continue
            if a['status_in_base'] == 'deleted-file':
                out.append('Deleted file: see the patch.')
                out.append('')
                continue
            out.append('Anchor:')
            out.append('```')
            out += [l.rstrip() for l in a['anchor']]
            out.append('```')
            h = hunks_by_file[a['file']]['hunks'][a['hunk'] - 1]
            body = [l.rstrip() for l in h.lines]
            out.append(f"Block (hunk {a['hunk']} of `{a['file']}` in the patch, {len(body)} lines):")
            out.append('```diff')
            if len(body) > BLOCK_FULL_MAX:
                out += body[:BLOCK_HEAD]
                out.append(f"... {len(body) - BLOCK_HEAD} more lines: see {e['patch']}")
            else:
                out += body
            out.append('```')
            out.append('')
    return '\n'.join(out).rstrip() + '\n'


def verify_docs():
    m = load_manifest()
    ok = True
    for p, render in ((INDEX, render_index), (BLOCKS, render_blocks)):
        want = render(m)
        have = p.read_text(encoding='utf-8') if p.exists() else ''
        if want != have:
            ok = False
            print(f'{p.name}: out of date (regenerate with --build)')
    for e in m['patches']:
        if not (HERE / e['patch']).exists():
            ok = False
            print(f"{e['patch']}: missing")
    print('docs ok' if ok else 'docs NOT aligned')
    return ok


# ------------------------------------------------------------------------------------------------------------------ check

def fold(s):
    return re.sub(r'\s+', ' ', s).strip()


def anchor_status(anchor, file_lines):
    if not anchor:
        return 'n/a'
    n = len(anchor)
    c = sum(1 for i in range(len(file_lines) - n + 1) if file_lines[i:i + n] == anchor)
    if c == 1:
        return 'present'
    if c > 1:
        return f'ambiguous x{c}'
    fa = [fold(l) for l in anchor]
    ff = [fold(l) for l in file_lines]
    c = sum(1 for i in range(len(ff) - n + 1) if ff[i:i + n] == fa)
    if c >= 1:
        return 'moved (whitespace)' if c == 1 else f'moved, ambiguous x{c}'
    # a weaker signal: the longest single anchor line is still there somewhere
    longest = max(anchor, key=lambda l: len(l.strip()))
    if longest.strip() and any(fold(longest) == l for l in ff):
        return 'missing (one line survives)'
    return 'missing'


def check(repo, rev, json_out=None):
    repo = Path(repo).resolve()
    m = load_manifest()
    tmp = Path(tempfile.mkdtemp(prefix='stack-check-'))
    wt = tmp / 'wt'
    res = {'rev': rev, 'base': m['base'], 'patches': []}
    try:
        git(['worktree', 'add', '--detach', '-q', str(wt), rev], repo)
        head = git(['rev-parse', 'HEAD'], wt).stdout.strip()
        res['rev_commit'] = head
        print(f"checking {len(m['patches'])} patches against {rev} ({head[:8]}) in {wt}")
        env = dict(os.environ, GIT_AUTHOR_NAME='stack-check', GIT_AUTHOR_EMAIL='stack@check',
                   GIT_COMMITTER_NAME='stack-check', GIT_COMMITTER_EMAIL='stack@check')
        for e in m['patches']:
            r = {'id': e['id'], 'name': e['name'], 'anchors': {}, 'apply': None}
            # anchors against the pristine revision
            cache = {}
            for a in e['anchors']:
                if a['status_in_base'] in ('new-file', 'deleted-file'):
                    exists = (wt / a['file']).exists()
                    st = ('file already exists' if exists else 'ok (absent)') if a['status_in_base'] == 'new-file' \
                        else ('ok (present)' if exists else 'already gone')
                else:
                    p = wt / a['file']
                    if not p.exists():
                        st = 'file missing'
                    else:
                        if a['file'] not in cache:
                            cache[a['file']] = p.read_text(encoding='utf-8', errors='replace').split('\n')
                        st = anchor_status(a['anchor'], cache[a['file']])
                r['anchors'][a['block']] = st
            # apply, on top of the patches accepted so far
            patch = HERE / e['patch']
            c = subprocess.run(['git', 'apply', '--check', str(patch)], cwd=str(wt), capture_output=True, text=True)
            if c.returncode == 0:
                subprocess.run(['git', 'apply', '--index', str(patch)], cwd=str(wt), capture_output=True, text=True)
                r['apply'] = 'clean'
            else:
                t = subprocess.run(['git', 'apply', '--3way', str(patch)], cwd=str(wt), capture_output=True, text=True)
                conflicts = subprocess.run(['git', 'diff', '--name-only', '--diff-filter=U'], cwd=str(wt),
                                           capture_output=True, text=True).stdout.split()
                if t.returncode == 0 and not conflicts:
                    r['apply'] = '3-way'
                else:
                    r['apply'] = 'rewrite'
                    r['conflicts'] = conflicts or [l for l in (t.stderr or '').splitlines() if 'error' in l][:8]
                    subprocess.run(['git', 'reset', '-q', '--hard'], cwd=str(wt), capture_output=True)
                    subprocess.run(['git', 'clean', '-fdq'], cwd=str(wt), capture_output=True)
            if r['apply'] in ('clean', '3-way'):
                subprocess.run(['git', 'add', '-A'], cwd=str(wt), capture_output=True)
                subprocess.run(['git', 'commit', '-q', '--no-verify', '-m', f"stack {e['id']} {e['name']}"],
                               cwd=str(wt), capture_output=True, env=env)
            counts = {}
            for st in r['anchors'].values():
                k = st.split(' ')[0].rstrip(',')
                counts[k] = counts.get(k, 0) + 1
            r['anchor_counts'] = counts
            res['patches'].append(r)
            print(f"  {e['id']} {e['name']:<28} apply: {r['apply']:<8} anchors: " +
                  ', '.join(f'{k} {v}' for k, v in sorted(counts.items())) +
                  (f"  conflicts: {', '.join(r['conflicts'])}" if r.get('conflicts') else ''))
    finally:
        subprocess.run(['git', 'worktree', 'remove', '--force', str(wt)], cwd=str(repo), capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(['git', 'worktree', 'prune'], cwd=str(repo), capture_output=True)
    if json_out:
        Path(json_out).write_text(json.dumps(res, indent=2), encoding='utf-8')
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', metavar='REPO', help='a git checkout of upstream (any worktree of it)')
    g.add_argument('--build', metavar='REPO', help='the fork checkout holding the lever commits')
    g.add_argument('--verify-docs', action='store_true')
    ap.add_argument('--rev', help='with --check: the upstream revision to test (tag, branch or sha)')
    ap.add_argument('--range', help='with --build: <base>..<head> of the lever commits')
    ap.add_argument('--json', help='with --check: write the results here')
    a = ap.parse_args()
    if a.check:
        if not a.rev:
            ap.error('--check needs --rev')
        check(a.check, a.rev, a.json)
    elif a.build:
        if not a.range or '..' not in a.range:
            ap.error('--build needs --range <base>..<head>')
        build(a.build, a.range)
    else:
        sys.exit(0 if verify_docs() else 1)


if __name__ == '__main__':
    main()
