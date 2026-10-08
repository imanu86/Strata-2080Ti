# Baseline Daily + upstream 0.1.41 — 8 ottobre 2026

Il runtime corrente parte da upstream `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` e porta soltanto le funzionalità attive del Daily verificato `be4cf1c0a4448ff0ec5a357cd5c61ffdf15041ca`. Il Daily installato non è stato modificato. Gli esperimenti del fork precedente restano nella storia e nei documenti marcati ARCHIVIO, senza entrare nel runtime corrente.

I quattro blocchi Daily sono: prefisso SSD nativo con identità, lookahead e MTP completo; cache fisica VMM primaria con borrow e safe point della pipeline (esclusa dopo risoluzione di gruppi multipli); kernel B6 MMVQ con default Daily; dipendenze FP16/TC e QSA del prefill SM75. I controllori e i fix upstream restano presenti. Il default implicito SIMT riguarda soltanto CC7.5; override espliciti e altre architetture seguono upstream. Il sensore/helper elastico Windows e gli oracle/collector/multichat MTP sperimentali dell'8 ottobre non sono attivi.

## Verifiche effettuate

Hardware: RTX3060 12GB stadio0/layer0–15 e RTX2080Ti modificata22GB stadio1/layer16–47. Build privata CUDA75;86, 6 job BelowNormal. Test SSD CPU nativo e portable: 244 controlli ciascuno. Test GPU VMM:35 controlli, massimo6MiB; B6 mode1/reps50 sulle due GPU: zero mismatch. Questo non qualifica HIP/SYCL o altre architetture.

HTTP FREE greedy seed73, contesti nativi131264/131255,1024 token per client, KVint8/residente32768, riserve1393/384MiB. Il prefisso SSD autentico è131159 token, al confine nativo del turno: creazione fuori misura, riavvio e restore verificato, poi normale riuso RAM senza LAB. Singola dedicata split16/pipeline2/MTP/elastica:63.8/64.9 token/s decode,2048 token in34.375s wall. Multichat target-only upstream batch2/gruppi auto2x1/pipeline0, MTP globale disponibile al ritorno seriale, batch-MTP e elasticità fisica spenti: B1/A2/B2 =53.61/60.71/60.93 token/s wall. Nessun guadagno convincente; nessuna nuova velocità rivendicata. Il ritorno target-only da slot non qualifica la continuità KV MTP. I testi campionati sono leggibili, non una valutazione generale di qualità. Il Daily aveva anche una baseline FREE: 70.140349/60.926632 token/s di native decode, 1024 token per caso in14599.3/16807.1ms, con lo stesso eseguibile Daily SHA `eccd5db37302f033343156397f5f0132059776ca64f470b1056b2cece996732c` (evidenza `Strata-lab-90tps-130k/runs/DET-free-reference-01`). Gli ultimi63.8/64.9 sono anch'essi native decode: scarto descrittivo L01circa−9%, L02circa+6.5%. Non è soltanto una differenza di denominatore. Parità prestazionale e regressione non sono qualificate: i prompt erano131248/131239 contro131264/131255 e differiscono cache, ambiente e output. Serve un confronto omogeneo prima di attribuire gli scarti al port.

## Confronto con il protocollo originale Claude

Il confronto diretto corretto tra candidato e Daily usa run_long.py originale immutato: prompt IDs131248/131239, L01 poi L02,1024 token ciascuno, temperatura0/seed73, un nuovo processo per arm. L01 fa prefill completo e L02 riusa131072 token della cache RAM nello stesso processo. Non usa HTTP, creazione SSD o warmup separato. La tripla candidato–Daily–candidato conserva gli stessi argomenti, ambiente, helper e fixture del riferimento FREE storico. Il precedente banco HTTP resta una prova distinta, non il confronto omogeneo col banco Claude.

| Arm | L01 native decode t/s | L02 native decode t/s | Totale su somma tempi decode t/s |
|---|---:|---:|---:|
| B1 candidato |64.516|58.267|61.232|
| A Daily originale |64.565|59.959|62.177|
| B2 candidato |67.607|58.556|62.757|

Il Daily originale odierno64.56/59.96 non riproduce70.14/60.93 storici su L01; la causa non è isolata. Il tempo medio totale dei B è33040.2ms contro32938.3ms di A: +0.309% di tempo, −0.308% di throughput, sostanzialmente vicino in questa singola tripla. Su L01 il tempo medio B è−2.21%; su L02 +2.65%, con entrambi i B più lenti del Daily. Nessun guadagno convincente né garanzia di equivalenza generale.

Gli output FREE e l’allocazione dinamica della cache differiscono: stessi input/driver non significano percorso verificato identico o qualità equivalente. B1/B2 partono da4434 slot, A da3645; i resize restano quelli della policy originale. Prime divergenze ID, finestre/draft, eventi elasticità, tempi/pin e rawpath sono nel [rapporto Claude originale](sm75/claude-exact-bab-20261008.json). Sei richieste completate, cleanup owned0 e Daily invariato; nessun tuning o deployment Daily.

Il riepilogo S1 delle finestre richiede DECODE_TIMING, assente nel Daily: il primo audit del banco lo richiedeva erroneamente. L'esito grezzo è conservato e l'audit posthoc usa startuppl2 e messaggi runtime stage0gap; conteggi finestre/catture e per-tokenp95 rimangono UNKNOWN. Il rapporto completo è [qui](sm75/daily-upstream-041-native-http-20261008.json).

## Riproduzione della build

Usare MSVC2022/CUDA12.6/CMake/Ninja e il checkout ggml già verificato, senza scaricare dipendenze durante una misura. Esempio da ambiente vcvars64:

```text
cmake -S . -B build-daily041 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DCMAKE_CUDA_ARCHITECTURES="75;86" -DCMAKE_CUDA_FLAGS="-D_WINDOWS -Xcompiler=/EHsc" -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=<verified-llama.cpp>
cmake --build build-daily041 --parallel 6 --target strata prefix_cache_file_test prefix_cache_portable_test expert_cache_elastic_test b6_mmvq_bench
```

Il runner esterno usato, conservato nel workspace di prova `work/daily-upstream-041-20261008/bench-plan/runner`, pinna sorgenti/server/tools/binario, ownership PID+nascita, lock e deadline reale. `run_http.py prepare/check/run --profile single|multi --deadline <UTC>` esegue create/restart/warm/S1 oppure create/restart/warm/B1/A2/B2. Richiede asset e payload privati locali; non sono inclusi qui. Non usare il runner LAB storico o forzare131072 al posto del root nativo. I timer HTTP/native annidati non si sommano, SSEchunk non equivale a token. Creazione SSD e caricamento sono separati dalle misure.

Attribuzione: Niko1221 e contributori upstream; autori storici del Daily e relativi trailer nei commit conservati; port selettivo e banco GPT-6.1 Sol/OpenAI Codex. La storia pubblicata non viene riscritta.
