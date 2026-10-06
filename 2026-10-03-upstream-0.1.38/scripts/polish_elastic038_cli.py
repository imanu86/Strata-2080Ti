"""Validate new numeric arguments before VMM; numerical kernels are unchanged."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent;R=BASE/'elastic038_20261003';p=R/'source/src/program/generate.cpp';s=p.read_text(encoding='utf-8');before=s
start=s.index('        else if (a == "--elastic-reserve-mib")');end=s.index('        else if (a == "--prefill")',start)
old=s[start:end]
new='''        else if (a == "--elastic-reserve-mib" || a == "--elastic-chunk-mib" ||
                 a == "--elastic-period-ms" || a == "--elastic-stable-ms" ||
                 a == "--elastic-tail-mib" || a == "--elastic-shrink-step-mib") {
            const std::string value = next(a.c_str());
            int number = 0;
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
            const int minimum = a == "--elastic-tail-mib" ? -1 :
                (a == "--elastic-chunk-mib" || a == "--elastic-shrink-step-mib") ? 1 : 0;
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || number < minimum) {
                std::fprintf(stderr, "%s needs an integer >= %d within int32 range\\n", a.c_str(), minimum);
                return 2;
            }
            if (a == "--elastic-reserve-mib") o.elastic_reserve_mib = number;
            else if (a == "--elastic-chunk-mib") o.elastic_chunk_mib = number;
            else if (a == "--elastic-period-ms") o.elastic_period_ms = number;
            else if (a == "--elastic-stable-ms") o.elastic_stable_ms = number;
            else if (a == "--elastic-tail-mib") o.elastic_tail_mib = number;
            else o.elastic_shrink_step_mib = number;
        }
'''
s=s[:start]+new+s[end:]
needle='                 "  --prefill CHUNK';at=s.index(needle)
help='''                 "  --elastic           --serve, single CUDA GPU: resize the expert-cache tail (default off)\\n"
                 "  --elastic-reserve-mib N  free-memory target (default 500, >=0)\\n"
                 "  --elastic-chunk-mib N    VMM mapping chunk (default 64, >=1)\\n"
                 "  --elastic-period-ms N    minimum policy-check interval (default 1000, >=0)\\n"
                 "  --elastic-stable-ms N    sustained room before growth (default 5000, >=0)\\n"
                 "  --elastic-tail-mib N     initial tail limit (-1 auto, 0 no fixed core, >0 MiB)\\n"
                 "  --elastic-shrink-step-mib N  maximum reduction per check (default 1024, >=1)\\n"
'''
s=s[:at]+help+s[at:]
# Everything following argument parsing is exactly the measured candidate.
body=before[before.index('    bool split_auto ='): ] if '    bool split_auto =' in before else before[before.index('    int n_dev ='):]
assert s.endswith(body)
p.write_text(s,encoding='utf-8',newline='\n')
(R/'cli-polish.json').write_text(json.dumps({'before_generate_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_generate_sha256':hashlib.sha256(s.encode()).hexdigest(),'post_argument_parsing_body_unchanged':True,'unchanged_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'changes':'help strings and strict parsing for six elastic numeric flags; no numerical kernel or cache/prefill computation change','unit_display_correction':'initial test printed8MiB reservation limit; largest successful map is6MiB;35 functional checks unchanged'},indent=2))
print('Elastic CLI validated ranges; inference body unchanged')
