# lab/speed3 — esperti in VRAM, lato pesi (EXPERTDEC §5)

Base: `lab/speed1` (0.1.43 + pila del fork + tagli di nodo CUDA). Leva 3 della classifica roofline
(`ROOFLINE_2026-10-12.md` §5): il kernel degli esperti in VRAM sta al **34% della banda** (2080 Ti, T=2: 4,43 ms a
finestra), legato alle istruzioni di load da 2 byte e ai codebook, non ai byte. Stima del documento: −1,5/−2,0 ms sullo
stadio 1, −1,0/−1,3 sullo stadio 0.

## 1. Perché il gate/up è lento (dal sorgente)

`expdec_gu_kernel` (`src/kernels/cuda/iq_kernels.cu`): 16 sub-warp da 16 lane, una riga per sub-warp (R = 1), 5
chiamate per lane (k = t, t+32, t+64 in `s`; t+16, t+48 in `s16`), ogni chiamata decodifica 32 elementi di una riga
IQ (blocco da 256 elementi = 98 B IQ3_XXS / 110 IQ3_S / 66-90 IQ2) con `Split<TG>::load`: due `get_int_b2` sui codici
(4 `LD.U16`), uno sui segni/scale (2 `LD.U16`), `d` fp16 (1 `LD.U16`) e 8 lookup nel codebook (già in shared). Sono
**7 load da 2 byte per chiamata**, ciascuno un settore parziale: 35 per riga-lane, 560 per riga, a fronte di 980 byte
di riga. Le attivazioni sono già in shared (EXPDEC). Il kernel è quindi legato al numero di istruzioni di load e ai
settori a metà, non ai byte: 34% della banda.

## 2. Cosa fa questo ramo (opt-in, bitwise)

| variabile | kernel | idea |
|---|---|---|
| `STRATA_EXPDEC_WT=1` | `expdec_gu_wt_kernel<TG, MINB, MAGIC>` | i **16 × gu_row byte** della passata (16 righe contigue del gate o dell'up) copiati in shared dal blocco intero con `LDG.128` (gu_row int4, ~4 per thread, tutti in volo insieme), poi decodificati da lì con `LDS` (1 wavefront). Stessi byte, stessa mappa lane → chiamata, stesso albero di `ed_gu_rows`: **bitwise**. R = 1; shared dinamica 16 × gu_row (IQ3_S 17,6 KB) accanto agli 11,5 KB delle entrate e al codebook. |
| `STRATA_EXPDEC_MAGIC=1` | `apply_um` in ogni `Split` gate/up, `ed_gu_rows<…, MAGIC>` | `(float) sumi` sostituito da `__int_as_float(sumi + 0x4B400000) − 12582912.0f`: lo stesso float per \|sumi\| < 2^22 (qui ≤ ~1M), senza `I2F` (16 lane/clk per SM su Turing). Da solo (R = 1) o insieme a WT. |

I kernel di default e `expdec_gu_kernel` restano come sono (il parametro template `MAGIC` ha default false).
`expdec_bench`: campi `t` e `i` nelle scelte (`dec:t1`, `dec:i1`, `dec:t1:i1`, `dec:t1:b2`), memcmp con `def` come
sempre. Riga di log `strata expdec: … expdec_gu_wt_kernel (weights tiled in shared) R 1 MINB 4 I2F-magic …`.

Passi successivi se il tile rende: (a) prefetch del tile della passata seguente in registri (4 int4 per thread) prima
del calcolo, così la copia non è più in serie col decode; (b) layout con 2 byte di padding per blocco in shared perché
i codici diventino allineati a 4 byte (`LDS.32` al posto di due `LDS.U16`); (c) lo stesso tile per il down Q2_0 / IQ4_NL.

## 3. Verifica

1. `expdec_bench 300 <dev> 20 all def,dec,dec:t1,dec:i1,dec:t1:i1,dec:t1:b2 real` su CUDA0 e CUDA1: tutte le celle
   `identical`, µs e GB/s del gate/up per coppia e T (è il dato: il tetto è 616 GB/s sulla 2080 Ti).
2. SASS (`cuobjdump -sass`): nel `wt` i `LD.U16` del decode devono essere diventati `LDS.U16` (se restano `LD` generici
   il compilatore non ha inferito lo spazio shared: va passato il puntatore come `__shared__`).
3. Run engine breve con `STRATA_EXPDEC=1 STRATA_EXPDEC_WT=1 [STRATA_EXPDEC_MAGIC=1]`: `L02.txt` identico a off.
4. B-A-B notturno del coordinatore.

## 4. Risultati (build `build-speed3-3fa04f87`, sha256 `3f2c36fc…`, 12/10 12:44-12:57)

**Verdetto: bitwise sì, più lento. `STRATA_EXPDEC_WT` perde il 18-25% del gate/up sulla 2080 Ti e il 18-35% sulla
3060; `STRATA_EXPDEC_MAGIC` è neutro (±1%). Leva archiviata: il codice resta opt-in, spento, fuori dalle build da
banco.**

`expdec_bench 300 <dev> 20 all def,dec,dec:t1,dec:i1,dec:t1:i1,dec:t1:b2 real`: **240 celle per scheda, tutte
`identical`** (0 confronti diversi). Somma dei µs del gate/up sulle 10 coppie (gate/up IQ2_XXS/IQ2_XS/IQ3_XXS/IQ3_S/IQ2_S
× down IQ4_NL/Q2_0), rispetto a `dec` (EXPDEC attuale):

| scheda | T | def | **dec** | dec:t1 (WT) | dec:i1 (MAGIC) | dec:t1:i1 | dec:t1:b2 |
|---|---|---|---|---|---|---|---|
| 2080 Ti | 1 | 511 (+8%) | **471** | 590 (+25%) | 475 (+1%) | 592 (+26%) | 588 (+25%) |
| 2080 Ti | 2 | 899 (+15%) | **779** | 972 (+25%) | 783 (+1%) | 971 (+25%) | 966 (+24%) |
| 2080 Ti | 3 | 1019 (+22%) | **834** | 991 (+19%) | 836 (0%) | 991 (+19%) | 986 (+18%) |
| 2080 Ti | 4 | 1136 (+25%) | **911** | 1085 (+19%) | 915 (+1%) | 1087 (+19%) | 1084 (+19%) |
| 3060 | 1 | 624 (−4%) | **649** | 870 (+34%) | 651 (0%) | 880 (+36%) | 937 (+44%) |
| 3060 | 2 | 1157 (+4%) | **1111** | 1310 (+18%) | 1115 (0%) | 1288 (+16%) | 1500 (+35%) |
| 3060 | 3 | 1271 (+10%) | **1157** | 1416 (+22%) | 1144 (−1%) | 1411 (+22%) | 1557 (+35%) |
| 3060 | 4 | 1409 (+14%) | **1233** | 1484 (+20%) | 1225 (−1%) | 1462 (+19%) | 1693 (+37%) |

2080 Ti, T=2, µs del gate/up per coppia (`dec` → `dec:t1`): IQ2_XXS 67,8 → 72,3; IQ2_XS 73,1 → 79,5; IQ3_XXS 79,6 →
98,0; IQ3_S 89,5 → 100,0; **IQ2_S 77,4 → 134,5 (+74%)**. Con `dec` il gate/up IQ3_XXS a T=2 muove 25 MB in 79,6 µs =
315 GB/s (51% del picco), IQ2_XXS 249 GB/s (40%).

SASS sm_75 (`cuobjdump -sass` di `iq_kernels.cu.obj`, IQ2_S `<22, MINB 4>`):

| kernel | load dei pesi nel decode | copia del tile | `I2F` | `BAR.SYNC` | spill |
|---|---|---|---|---|---|
| `expdec_gu_kernel` | 100 `LD.E.U16` + 40 `LD.E.U8` (generici) | — | 50 | 2 | 1 `LDL`/2 `STL` |
| `expdec_gu_kernel` MAGIC | idem | — | **0** | 2 | 5 `LDL`/5 `STL` |
| `expdec_gu_wt_kernel` | **100 `LDS.U.U16` + 40 `LDS.U.U8`** | 7 `LD.E.128` + 7 `STS.128` | 50 | 4 | 0 |
| `expdec_gu_wt_kernel` MAGIC | idem | idem | **0** | 4 | 0 |

(IQ3_XXS: 140 `LD.E.U16` → 140 `LDS.U.U16`.) Il compilatore ha fatto quello che si voleva: i load da 2 byte del decode
sono diventati `LDS` e l'`I2F` è sparito. Eppure il kernel rallenta, quindi l'ipotesi del §1 era sbagliata:

- **Il gate/up non è legato ai load globali da 2 byte.** Spostarli in shared non toglie istruzioni (lo stesso numero di
  `LDS.U16`/`LDS.U8`, più i lookup del codebook `LDS.U.64`/`LDS.U.128` che c'erano già) e aggiunge la copia: il decode
  resta lo stesso flusso di istruzioni. Sulla 2080 Ti EXPDEC sta già al 40-51% della banda nominale sul gate/up.
- **La copia del tile è in serie col decode** (`LD.E.128` → `STS.128` → `BAR.SYNC` → decode → `BAR.SYNC`): senza la
  sovrapposizione fra passate il blocco paga latenza DRAM e decode uno dopo l'altro, e le 16 × gu_row byte di shared in più
  (IQ3_S 17,6 KB) tolgono blocchi residenti per SM. IQ2_S, il peggiore, ha un `gu_row` di 820 B: le 16 righe del tile
  cadono sugli stessi banchi più spesso.
- **`I2F` non è un collo**: toglierlo (50 → 0) non sposta i µs, né da solo né con WT.

Run engine brevi (L02, 64 token, giorno, banco rumoroso): `L02.txt` **identico** a `off` sia con `wt` sia con `wtm`
(md5 `d48bfa50…`, lo stesso dei run speed1/speed2); t/s 68,9 off, 67,2 wt, 66,0 wtm (indicativi). La riga
`expdec_gu_wt_kernel (weights tiled in shared) … [I2F-magic]` c'è per i tipi 16/17/21/22 sulla 3060 (CUDA0) e per
il tipo 18 sulla 2080 Ti (CUDA1).

I passi (a)-(c) del §2 non si fanno: (a) il prefetch in registri del tile seguente toglierebbe la serie ma non le
istruzioni, che sono il costo; (b) `LDS.32` al posto di due `LDS.U16` riduce i load da shared di ~⅓, ma il bench mostra
che anche coi load da shared il kernel non scende sotto EXPDEC. Se si riapre il gate/up, la direzione è meno istruzioni di
decode per byte (codebook in registri per IQ2, R = 2 righe per sub-warp a T ≥ 3), non lo spazio dei load.

## Stato

- 12/10 12:40: codice, bench e doc scritti.
- 12/10 12:57: compilato sì (`build-speed3-3fa04f87`), bench e run engine fatti: **bitwise sì, WT −18/−35%, MAGIC
  neutro. Archiviata**, nessuna build da banco con queste variabili.
