# EXPERTDEC: gli esperti VRAM del decode con le attivazioni in shared e più righe per lane (`STRATA_EXPDEC=1`)

Branch `lab/expertdec` del fork `imanu86/Strata-2080Ti`, da `lab/candidate` (`443ce1c9`). Obiettivo: dimezzare la
pendenza per riga degli esperti in VRAM a T = 2..4 (nsys `DET-qsarows-nsys2`, `docs/QSAROWS.md` §3: +1,04 ms per riga
sulla 3060, +1,31 sulla 2080 Ti, la prima voce del muro del grafo) con risultati **identici bit per bit** al kernel di
default. Tutto opt-in: `STRATA_EXPDEC=1`; spento, il motore lancia esattamente i kernel di prima.

## 1. Da dove viene la pendenza per riga

`native_gu_multi_kernel` / `native_down_multi_kernel` (`src/kernels/cuda/iq_kernels.cu`) decodificano **già** ogni
pezzo di peso una volta per blocco (`Split<TG>::load`) e lo applicano alle ≤ 4 entrate del gruppo
(`row_dot_80_sub16<TG, NC>`): la decodifica ripetuta per entrata non è la causa. Dai numeri di nsys sulla 2080 Ti
(L02 a 131k, 32 layer): gate/up 1,56 ms a T = 1 (≈ 11,7 esperti e 11,7 entrate per layer-finestra, dal
`VRAM hits 24.25` a T medio 2,15) e 4,21 ms a T = 4 (≈ 22 esperti distinti, 47 entrate). Con `t = a·G + b·E` per layer:
**a ≈ 2,6 µs per esperto, b ≈ 1,6 µs per entrata**: un'entrata in più costa il 60% di un esperto intero pur non
leggendo un byte dalla DRAM. Il micro-bench (§4, forme `real`/`single`) conferma sul kernel di default della 2080 Ti:
IQ3_S gate/up **a = 5,2 µs, b = 1,23 µs** per layer (la DRAM dell'esperto, 1,41 MB a 616 GB/s, vale 2,3 µs: il
lato pesi è al 44% della banda; il lato entrate è puro overhead di istruzioni).

Perché il lato entrate costa: per ogni (chiamata k, colonna c) la lane legge dalla q8_1 del token le 8 parole `qs` e
`ds` del blocco k con 9 `LDG.32` a stride 36 B. Le 16 lane di un sub-warp leggono 16 blocchi consecutivi: ogni
istruzione tocca 16-32 settori da 32 B della L1, cioè ~5-8 cicli di L1 per istruzione, ~50-70 cicli di L1 per
(warp-step, colonna) contro ~6 cicli di ALU per gli 8 `dp4a`. Il SASS del kernel di default IQ3_S (`cuobjdump -sass`,
sm_75) ha 355 `LDG.32` + 84 `LD.U16` contro 240 `IDP` (dp4a): due terzi delle istruzioni di memoria sono le
attivazioni. Il kernel down (`row_dot_40_sub8`, IQ4_NL) ha già l'hq del gruppo in shared (layout AoS da 36 B, 5
`LDS.32` senza conflitti per colonna): la sua pendenza è infatti minore (+0,36 contro +0,88 ms per riga sulla 2080 Ti).

Perché il lato pesi resta al 40-55% della banda: `Split<TG>::load` legge i blocchi (66-110 B, allineati a 2 B) con
`get_int_b2` = coppie di `LD.U16`, più il qh, i segni, lo scale e `d`: 8-9 istruzioni di load da 2 B per chiamata e
riga (IQ3_S), più 8 lookup nel codebook (random, da shared per IQ2_* e qui anche per IQ3_*). Sono tante istruzioni di
memoria per pochi byte: il limite è il numero di richieste alla L1/LSU, non i byte. Un carico a 16 B coalescato
(tile di pesi in shared) è il passo successivo, non fatto qui (§5).

I contatori ncu (settori per richiesta, stalli `lg_throttle`/`mio_throttle`) non si possono leggere su questa
postazione: `ERR_NVGPUCTRPERM`, i performance counter richiedono privilegi di amministratore non ottenibili da
remoto. Le conclusioni sopra vengono dal modello, dal SASS e dalla scomposizione a/b del bench.

## 2. Cosa fa EXPDEC (`expdec_gu_kernel<TG, R, MINB>`, `expdec_down_kernel<TD, R>`)

- **Attivazioni del gruppo in shared**, NC = 4 entrate per volta come il default: per il gate/up 80 blocchi × (8 parole
  + `ds`) per entrata, 11,5 KB per 4 entrate; per il down 20 blocchi. Layout SoA: le due metà da 16 B del blocco in due
  piani (`s_u[(2c + h)·80 + k]`), così una chiamata legge le 8 parole con **due `LDS.128`** senza conflitti di banco
  (16 k consecutivi = 256 B contigui per piano) e lo scale fp16 è già un float (`__low2float(ds)`, lo stesso valore
  che `apply` calcola). Per IQ4_NL le parole del blocco restano insieme e la chiamata legge due `LDS.64` (le coppie
  `iqs, iqs+1` e `iqs+4, iqs+5`). SASS del gate/up IQ3_S R=1: 40 `LDS.128` + 20 `LDS.32` al posto dei 355 `LDG.32`.
- **R righe per lane**: la lane decodifica i pesi di R righe allo stesso k e applica a ciascuna le stesse parole di
  attivazione, caricate una volta per NC × R prodotti. Per il gate/up il sub-warp resta di 16 lane (16 sub-warp × R
  righe per passata), per IQ4_NL di 8 lane, per Q2_0 di 16: l'albero per riga non cambia. Dal bench: gate/up **R = 1**
  (`STRATA_EXPDEC_R`, 1 o 2), down **R = 4** (`STRATA_EXPDEC_DR`, 1/2/4).
- **Più righe per blocco** (`rb`): il blocco itera su `rb` righe in passate di 16R (gate/up) o 32R / 16R (down) righe,
  così lo staging da 11,5 KB si ammortizza su 35-140 KB di pesi. `STRATA_EXPDEC_RB` / `STRATA_EXPDEC_DRB` lo fissano;
  `0` (default) sceglie il massimo che tenga almeno `STRATA_EXPDEC_MINBLK` (800) blocchi nel lancio
  (`ceil(righe / rb) × cap_groups`, con `cap_groups` = i gruppi possibili del lancio, n·K): T = 1 (cap 10) → 16R
  righe, T = 2 (cap 20) → 32, T = 4 (cap 40) → 64.
- **Codebook in shared** anche per IQ3_XXS (1 KB) e IQ3_S (2 KB): `Split<18>::load<true>` / `Split<21>::load<true>`
  leggono la stessa tabella dalla shared (IQ2_XXS/XS/S la mettevano già in shared i kernel di default).
- **Registri**: `__launch_bounds__(256, MINB)` del gate/up con MINB a runtime (`STRATA_EXPDEC_MINB`, default 4 dal
  bench 2): 4 = fino a 64 registri (4 blocchi per SM come il default, 8 B di stack), 2 = fino a 128 (2 blocchi per SM);
  il down resta a 2 (Q2_0 R 4: 106 registri).
- Formati: gate/up IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22) a n_embd 2560 (`n_ff` multiplo di
  64). Il down solo con `STRATA_EXPDEC_DOWN=1` (Q2_0 (42) a n_ff 640; IQ4_NL (20) anche con `STRATA_EXPDEC_D20=1`, dal
  bench non guadagna: il kernel di default ha già l'hq in shared): **di default EXPDEC prende il solo gate/up**, perché
  nel motore il down Q2_0 con EXPDEC è risultato ~10% più lento del default (§7) pur essendo 0,78 nel bench. Ogni ruolo
  decide da sé; un formato fuori lista tiene il kernel di default. `STRATA_OLD_IQ_MMVQ=1` spegne EXPDEC come spegne i
  multi.
- API (`iq_kernels.hpp`): `native_expert_set_expdec`, `native_expert_set_expdec_params` /
  `native_expert_expdec_params` (`NativeExpertExpdecParams`), `native_expert_expdec_takes`. Letti all'avvio
  dall'ambiente; i grafi catturati tengono i kernel catturati.

### Bit per bit: perché

Per ogni (riga, entrata) la somma è quella di `row_dot_80_sub16` (gate/up), `row_dot_40_sub8` (IQ4_NL) o
`row_dot_20_sub16` (Q2_0): la stessa lane fa le stesse chiamate k nello stesso ordine (gate/up: t, t+32, t+64 in `s`;
t+16, t+48 in `s16`; `s = __fadd_rn(s, s16)`; butterfly xor 8-4-2-1), `apply_u` è `apply` con le stesse operazioni
intere e float nello stesso ordine sulle stesse parole (il blocco q8_1 letto dalla shared invece che dalla globale), e
le entrate del gruppo sono indipendenti fra loro (quante per passata e in che ordine non cambia nulla). R righe per
lane e `rb` righe per blocco cambiano solo chi calcola cosa, non le somme. La contrazione FMA del compilatore
(`s += d * sumi`) è la stessa decisione sulla stessa forma di espressione nei due kernel: fa fede il bench, che su
tutte le celle (10 coppie × 4 T × 2 forme × 2 schede × ogni variante) dice `identical`.

## 3. Bench: `expdec_bench`

```
build-sm7586\expdec_bench [reps=300] [device=-1] [groups=20] [coppie=all] [scelte=def,dec,dec:r2,dec:b4,dec:r2:b4] [forma=real]
```

- Coppie: le 10 del modello (5 gate/up × 2 down) o `21/20,16/42`. Scelte: `def` (riferimento) e
  `dec[:r<1|2>][:d<1|2|4>][:b<2|4>][:w<0|1>][:x<0|1>][:g<n>][:h<n>][:m<n>]` (r righe per lane del gate/up, d del down,
  b MINB del gate/up, w EXPDEC anche sul down (il default del motore è il solo gate/up), x anche per IQ4_NL, g/h righe
  per blocco con 0 = auto, m minimo di blocchi; i campi mancanti dal motore, cioè dall'ambiente `STRATA_EXPDEC_*` o dai
  default). Le misure dei §4 e §6 sono state fatte con il down acceso (`dec` di allora = `dec:w1` di oggi). Forme: `real` = ogni token sceglie 10 esperti fra
  `groups` distinti (T = 1: 10 gruppi × 1 entrata; T = 2: 20 × 1; T = 4: 20 gruppi, 40 entrate); `single` = 10 T gruppi
  di una entrata; `both` stampa anche la scomposizione **a** = (single T=4 − real T=2) / 20 (µs per esperto) e
  **b** = (real T=4 − real T=2) / 20 (µs per entrata).
- Riga: `dev T forma gu/d scelta all_us gu_us down_us GB/s pct identical|DIFFERENT`: la chiamata intera (gate/up +
  SwiGLU + q8_1 + down, mediana di 5 round × reps), poi le fasi gate/up e down da sole; memcmp dell'uscita contro
  `def`; exit 1 se qualcosa differisce. I blob ruotano su ≥ 96 MB (pesi freddi dalla DRAM come nel decode).
- Comando della misura: `expdec_bench 300 -1 20 all def,dec,dec:r2,dec:b4,dec:r2:b4 both` (entrambe le schede,
  ~10 min). Nel log della postazione i device sono 0 = 2080 Ti, 1 = 3060 (senza `CUDA_VISIBLE_DEVICES`).

## 4. Esito del bench, iterazione 1 (build `87e0102f`, 12/10)

Scelte dell'iterazione 1: `dec:R` = R righe per lane per entrambi i ruoli, MINB 2, IQ4_NL compreso. **Bitwise:
`identical` su tutte le 640 celle** (10 coppie × 4 T × 2 forme × 4 scelte × 2 schede). Tempi: `def` in µs della
fase, le altre come rapporto su `def` (< 1 = più veloce); forma `real`.

**2080 Ti (stadio 1), gate/up**

| coppia | T | def | dec:1 | dec:2 | dec:4 |
|---|---|---|---|---|---|
| IQ2_XXS (16/20) | 1 | 46 | 0.99 | 0.89 | 1.07 |
| IQ2_XXS (16/20) | 2 | 73 | 0.94 | 0.88 | 1.09 |
| IQ2_XXS (16/20) | 3 | 86 | 0.86 | 0.81 | 1.03 |
| IQ2_XXS (16/20) | 4 | 97 | 0.83 | 0.79 | 0.94 |
| IQ2_XS (17/20) | 1 | 60 | 0.74 | 0.78 | 0.91 |
| IQ2_XS (17/20) | 2 | 79 | 0.93 | 0.93 | 1.15 |
| IQ2_XS (17/20) | 3 | 92 | 0.87 | 0.86 | 1.08 |
| IQ2_XS (17/20) | 4 | 103 | 0.84 | 0.83 | 0.98 |
| IQ3_XXS (18/20) | 1 | 54 | 0.88 | 0.91 | 1.19 |
| IQ3_XXS (18/20) | 2 | 94 | 0.85 | 0.86 | 1.16 |
| IQ3_XXS (18/20) | 3 | 106 | 0.80 | 0.81 | 1.08 |
| IQ3_XXS (18/20) | 4 | 118 | 0.80 | 0.80 | 0.99 |
| IQ3_S (21/20) | 1 | 62 | 0.82 | 0.92 | 1.10 |
| IQ3_S (21/20) | 2 | 112 | 0.79 | 0.86 | 0.99 |
| IQ3_S (21/20) | 3 | 123 | 0.74 | 0.83 | 0.97 |
| IQ3_S (21/20) | 4 | 136 | 0.75 | 0.80 | 0.89 |
| IQ2_S (22/20) | 1 | 62 | 0.78 | 0.75 | 0.97 |
| IQ2_S (22/20) | 2 | 84 | 0.92 | 0.92 | 1.18 |
| IQ2_S (22/20) | 3 | 96 | 0.87 | 0.86 | 1.10 |
| IQ2_S (22/20) | 4 | 109 | 0.84 | 0.83 | 1.01 |

**2080 Ti (stadio 1), down**

| coppia | T | def | dec:1 | dec:2 | dec:4 |
|---|---|---|---|---|---|
| IQ4_NL (16/20) | 1 | 26 | 1.06 | 1.02 | 1.13 |
| IQ4_NL (16/20) | 2 | 44 | 1.02 | 1.03 | 1.07 |
| IQ4_NL (16/20) | 3 | 49 | 0.99 | 1.03 | 1.10 |
| IQ4_NL (16/20) | 4 | 55 | 0.95 | 1.00 | 1.09 |
| Q2_0 (16/42) | 1 | 25 | 0.85 | 0.76 | 0.70 |
| Q2_0 (16/42) | 2 | 34 | 1.03 | 0.86 | 0.84 |
| Q2_0 (16/42) | 3 | 41 | 0.94 | 0.78 | 0.76 |
| Q2_0 (16/42) | 4 | 45 | 0.92 | 0.78 | 0.78 |

**2080 Ti, scomposizione** (µs per layer; a = per esperto, b = per entrata), coppie con down IQ4_NL / Q2_0:

| coppia | scelta | a gu | a down | b gu | b down |
|---|---|---|---|---|---|
| 21/20 | def | 5.20 | 1.81 | 1.23 | 0.59 |
| 21/20 | dec:1 | 3.55 | 1.75 | 0.70 | 0.40 |
| 21/20 | dec:2 | 3.88 | 1.77 | 0.69 | 0.52 |
| 18/20 | def | 4.34 | 1.80 | 1.19 | 0.57 |
| 18/20 | dec:1 | 3.16 | 1.75 | 0.70 | 0.38 |
| 16/20 | def | 3.36 | 1.80 | 1.18 | 0.58 |
| 16/20 | dec:1 | 2.69 | 1.74 | 0.57 | 0.40 |
| 21/42 | def | 5.22 | 1.46 | 1.22 | 0.63 |
| 21/42 | dec:2 | 3.81 | 1.07 | 0.67 | 0.29 |
| 21/42 | dec:4 | 4.67 | 1.01 | 0.55 | 0.36 |
| 22/42 | def | 3.83 | 1.48 | 1.19 | 0.64 |
| 22/42 | dec:2 | 2.99 | 1.05 | 0.64 | 0.29 |

**3060 (stadio 0)**: gate/up fra 0,86 e 1,27 di `def` (peggio a T = 1-2 con R = 1, ~0,9 a T = 3-4 su IQ3_S), rumoroso
(la 3060 regge il desktop: il primo punto di ogni coppia oscilla del 10-20%); down IQ4_NL 0,97-1,14; down Q2_0 con
R = 4 **0,67-0,79**, con R = 2 0,77-0,89. Scomposizione IQ3_S: a gu 7,2 → 5,4-5,6, b gu 1,85 → 0,6; Q2_0 a down
3,2-3,9 → 1,8-2,1.

Lettura:

1. La pendenza per entrata del gate/up è **dimezzata** (b 1,2 → 0,6-0,7 µs sulla 2080 Ti) e il costo per esperto
   scende del 25-30% (a 5,2 → 3,55 su IQ3_S, perché anche a T = 1 l'unica colonna non passa più dalla L1): la parte che
   resta è il lato pesi (3,55 µs contro 2,3 di DRAM), cioè le 8-9 richieste da 2 B per chiamata e riga.
2. R = 1 batte R = 2 sul gate/up (116-126 registri contro 128 e stack, stessi 2 blocchi per SM): il riuso delle
   attivazioni fra righe non paga quanto la pressione dei registri costa. Sul down Q2_0 (90-106 registri) R = 4 vince
   su entrambe le schede. Sul down IQ4_NL EXPDEC non guadagna (il default è già in shared): resta il default.
3. Stima sul motore (2080 Ti, 32 layer, ~11,7 entrate e ~3,4 esperti distinti in più per riga): gate/up
   11,7 × 1,23 + 3,4 × 5,2 = 32 µs/layer → 1,0 ms per riga col default (misurato +0,88), 11,7 × 0,70 + 3,4 × 3,55 =
   20 µs → **0,65 ms** con `dec:1` (−35%); down Q2_0 −25%, IQ4_NL ±0. Pendenza totale attesa 1,31 → ~0,95-1,0 ms per
   riga: **non ancora il dimezzamento**, che richiede il lato pesi (§5).

## 5. Iterazione 2 e passi successivi

Iterazione 2 (questo HEAD): knob separati per ruolo (`STRATA_EXPDEC_R` default 1, `STRATA_EXPDEC_DR` default 4,
IQ4_NL dietro `STRATA_EXPDEC_D20`), MINB a runtime (`STRATA_EXPDEC_MINB` 2 | 4) per provare 4 blocchi per SM con 64
registri. Esito: vedi §6.

Passo successivo per il lato pesi (non fatto): i blocchi dei pesi di un tile di righe (16 righe × 1100 B = 17,6 KB,
contigui) copiati in shared con `LDG.32`/`LDG.128` coalescati e decodificati da lì (`LDS.U16` a 1 wavefront invece di
`LD.U16` a 5-8 settori), eventualmente a doppio buffer per step k (14 KB per step a 32 righe); oppure la conversione
int→float con la costante magica (`__int_as_float(sumi + 0x4B400000) - 12582912.0f`, esatta per |sumi| < 2^22, qui
sempre) al posto di `I2F` (16 lane/clk per SM su Turing). Entrambi bitwise per costruzione, da verificare col bench.

## 6. Esito del bench, iterazione 2 (build `1a41387e`, 12/10) e scelta finale

`expdec_bench 300 -1 20 all def,dec,dec:r2,dec:b4,dec:r2:b4,dec:d1:x1,dec:d2:x1,dec:d4:x1 both`: **bitwise
`identical` su tutte le 1280 celle**. `dec` = gate/up R 1 MINB 2, down Q2_0 R 4, IQ4_NL default; `dec:b4` = MINB 4 (64
registri, 4 blocchi per SM); `dec:r2` = gate/up R 2; `dec:d<n>:x1` = EXPDEC anche sul down IQ4_NL con R n. Rapporti su
`def` (µs della fase, forma `real`):

**2080 Ti, gate/up** (def in µs a T 1/2/3/4, poi `dec` e `dec:b4`)

| formato | def T1-4 | dec | dec:b4 |
|---|---|---|---|
| IQ2_XXS | 42 / 74 / 86 / 97 | 1.00 / 0.93 / 0.86 / 0.84 | 0.98 / 0.92 / 0.86 / 0.84 |
| IQ2_XS | 60 / 80 / 92 / 104 | 0.74 / 0.91 / 0.87 / 0.83 | 0.75 / 0.92 / 0.87 / 0.84 |
| IQ3_XXS | 61 / 95 / 107 / 119 | 0.77 / 0.86 / 0.79 / 0.79 | 0.79 / 0.84 / 0.79 / 0.77 |
| IQ3_S | 72 / 112 / 124 / 138 | 0.71 / 0.79 / 0.74 / 0.75 | 0.73 / 0.79 / 0.75 / 0.74 |
| IQ2_S | 62 / 83 / 96 / 109 | 0.78 / 0.92 / 0.87 / 0.85 | 0.75 / 0.92 / 0.86 / 0.84 |

`dec:r2` è pari o peggio tranne IQ2_XXS/IQ2_S (0,79-0,83 a T = 4). **2080 Ti, down**: Q2_0 `dec` 0.69 / 0.85 / 0.77 /
0.79 (R = 4; R 1-2 via `x1` 0,86-1,02); IQ4_NL `dec` = 1,00 (default), con `x1` 0,95-1,04: resta il default.

**3060, gate/up**: `dec:b4` 0.87-0.95 a T ≥ 2 su tutti i formati (IQ3_S 0.99 / 0.89 / 0.87, IQ3_XXS 0.92 / 0.90 /
0.82), `dec` (MINB 2) 0,91-1,11: sulla 3060 i 4 blocchi per SM contano. **3060, down**: Q2_0 `dec` 0.76 / 0.73 /
0.71 / 0.65; IQ4_NL 0,97-1,07 (default).

Scomposizione (µs per layer, a = per esperto, b = per entrata) con la scelta finale `dec:b4`:

| scheda | coppia | ruolo | a def → dec:b4 | b def → dec:b4 |
|---|---|---|---|---|
| 2080 Ti | 21/20 | gate/up IQ3_S | 5.20 → 3.73 | 1.26 → 0.64 |
| 2080 Ti | 18/20 | gate/up IQ3_XXS | 4.39 → 3.27 | 1.22 → 0.62 |
| 2080 Ti | 16/20 | gate/up IQ2_XXS | 3.42 → 2.65 | 1.18 → 0.67 |
| 2080 Ti | 16/42 | down Q2_0 | 1.45 → 1.02 | 0.59 → 0.39 |
| 2080 Ti | 16/20 | down IQ4_NL | 1.81 → 1.80 | 0.58 → 0.59 |
| 3060 | 21/20 | gate/up IQ3_S | 6.25 → 5.35 | 1.52 → 0.49 |
| 3060 | 16/42 | down Q2_0 | 3.77 → 1.84 | 1.15 → 0.54 |

**Scelta finale** (default di `STRATA_EXPDEC=1` da questo HEAD): gate/up R 1 con MINB 4 (`STRATA_EXPDEC_MINB=4`),
down Q2_0 R 4, down IQ4_NL col kernel di default. Stima sul motore (2080 Ti, 32 layer, ~11,7 entrate e ~3,4 esperti
distinti in più per riga): gate/up IQ3_S 11,7 × 1,26 + 3,4 × 5,2 = 32 µs/layer → 11,7 × 0,64 + 3,4 × 3,73 = 20 µs
(**−38%**, la pendenza +0,88 ms per riga → ~0,55); down Q2_0 11,9 → 7,9 µs/layer (−34%), IQ4_NL ±0 → la pendenza down
+0,36 → ~0,28; router +0,08 invariato. **Pendenza esperti attesa: +1,31 → ~0,9 ms per riga sulla 2080 Ti (−30%),
+1,04 → ~0,9 sulla 3060 (−10/−15%)**. Non è il dimezzamento chiesto: la parte che resta è il costo per esperto del
gate/up (3,7 µs contro 2,3 di DRAM), cioè il lato pesi (§5), più ~1 ms piatto a T = 1 che EXPDEC già toglie in parte
(IQ3_S a T = 1: 0,71-0,73).

Costo di T = 1: nessuna perdita sulla 2080 Ti (0,71-1,00); sulla 3060 `dec:b4` a T = 1 è 0,89-1,09 (rumore del
desktop compreso). Nel motore il 42% delle finestre è a T = 1.

Pronto per il B-A-B end-to-end (lo lancia l'owner): stesso exe, braccio B con `STRATA_EXPDEC=1`, braccio A senza; testo
identico atteso (bitwise), da guardare `strata decode timing` per T e il profilo `VRAM hits`.

## 7. Nel motore: prova che i kernel girano e dove va il tempo (nsys, 12/10)

Il B-A-B dell'owner (`DET-x2-expdec-{a,b1,b2}`, build `ff8614a6`, config del candidato, B con `STRATA_EXPDEC=1`) è
neutro: −0,3% (forbice −2,2 / +1,7), 26,06 ms/finestra su L02 in entrambi i bracci; classi fresh 27,9 → 27,75 / 27,5
ms, speculative 18,8 → 18,6 / 18,9. Il log di B non diceva nulla di EXPDEC: da `e4885f1d` `native_expert_grouped`
scrive una riga `strata expdec: CUDAn <ruolo> type T: expdec_*_kernel R .. MINB .., N rows per block, grid X x Y (cap G
groups), captured into a graph` la prima volta che incontra un (ruolo, formato) (e `not covered: the default kernel
runs` per i formati fuori lista, qui il down IQ4_NL).

Due catture nsys sullo stesso build `e4885f1d` (`DET-x2-expdec-nsysA` = default, `-nsysB` = `STRATA_EXPDEC=1`; banco
forzato L01 16 token + L02 200 token = 96 finestre di L02, stesse opzioni e `STRATA_FORCE_*` del B-A-B; `nsys profile
--trace=cuda --cuda-graph-trace=node`; `nsys_rows.py` sul `.sqlite` in `D:\ds4_work\strata\nsys\`; cartelle in
`20261004_routed_keep\`).

**I kernel girano.** In B: `expdec_gu_kernel` 6 720 lanci sulla 2080 Ti (64 per finestra = 32 layer × 2 gruppi) e 5 472
sulla 3060; `expdec_down_kernel` 3 570 (34 per finestra: i 17 layer Q2_0 dello stadio 1) e 4 446; il down IQ4_NL resta
`native_down_multi_kernel` (30 per finestra). Nel log di B le sette righe `strata expdec:` (gate/up 16/17/18/21/22 e
down 42 presi, down 20 non coperto), tutte `captured into a graph`. In A zero lanci `expdec_*`.

**Quanto rendono nel motore** (mediane per finestra, ms, T = 1 / 2 / 3 / 4; dev1 = 2080 Ti = stadio 1, dev0 = 3060 =
stadio 0):

| | A (default) | B (EXPDEC) |
|---|---|---|
| 2080 Ti gate/up kernel | 1,60 / 2,58 / 3,51 / 4,16 | **1,42 / 2,07 / 2,75 / 3,20** (−11 / −20 / −22 / −23%) |
| 2080 Ti gate/up, pendenza per riga | +0,86 | **+0,60** |
| 2080 Ti down (Q2_0 + IQ4_NL) | 0,73 / 1,12 / 1,50 / 1,78 | 0,82 / 1,20 / 1,60 / 1,88 (Q2_0 expdec 0,42 / 0,61 / 0,83 / 0,97 + IQ4_NL default 0,40 / 0,59 / 0,77 / 0,91) |
| 2080 Ti famiglia esperti | 3,08 / 4,48 / 5,81 / 6,89 | 3,01 / 4,09 / 5,19 / 6,03 |
| 2080 Ti `wait_flag_ge` (GPU che aspetta l'host) | 2,45 / 2,23 / 4,52 / 6,97 | 1,58 / 1,88 / 5,37 / 6,35 |
| 2080 Ti buchi fra i nodi del grafo | 5,7 / 5,7 / 6,2 / 5,4 | 4,9 / 4,9 / 4,4 / 5,1 |
| 2080 Ti muro del grafo | 19,6 / 22,5 / 27,8 / 32,9 | 18,5 / 20,7 / 26,5 / 30,4 |
| 3060 gate/up kernel | 0,77 / 1,43 / 1,97 / 2,23 | 0,80 / 1,21 / 1,83 / 1,84 |
| 3060 down | 0,54 / 0,95 / 1,36 / 1,62 | 0,51 / 0,78 / 1,08 / 1,18 |
| 3060 muro del grafo | 13,0 / 14,5 / 16,4 / 19,3 | 10,9 / 12,7 / 12,9 / 17,1 |

(Sotto nsys tutto è più lento del banco pulito: `decode timing` 45,4 ms/finestra in A e 41,3 in B contro 26 nel
B-A-B, `wait_flag` 2-7 ms contro 1,5-2,6 di `DET-qsarows-nsys2`; valgono i confronti A/B, non gli assoluti.)

**Lettura.**

1. Il gate/up rende come nel micro-bench: −20/−23% a T ≥ 2 sulla 2080 Ti, pendenza +0,86 → +0,60 ms per riga (il bench
   prevedeva 4,16 × 0,74 = 3,08 a T = 4; misurato 3,20). Il down Q2_0 invece nel motore è **~10% più lento** del default
   (28,5 µs per lancio contro ~25,6 a T = 4) dove il bench dava 0,78: la differenza è la geometria. La regola `rb` auto
   conta i blocchi su `cap_groups` = n·K (10 / 20 / 30 / 40), ma i gruppi reali di una finestra sono ~12 / 18 / 22 / 24:
   a T = 4 sceglie 128 righe per blocco (down) e 64 (gate/up) dove il bench, con `cap` = gruppi veri, lavorava a 64 e
   32. **Verificato con `-nsysB2`** (`STRATA_EXPDEC_RB=32 STRATA_EXPDEC_DRB=64`): gate/up 1,35 / 2,10 / 2,75 / 3,20 e
   down Q2_0 0,42 / 0,61 / 0,82 / 1,04, identici a B: la geometria non c'entra. Il down Q2_0 nel motore resta ~10%
   peggio (probabile la convivenza con la catena del drafter sulla stessa scheda e i 106 registri contro 61: 2 blocchi
   per SM invece di 4; non misurato oltre). Da `e4885f1d`+1 il down è dietro `STRATA_EXPDEC_DOWN=1` e **di default
   EXPDEC prende il solo gate/up**.
2. Il risparmio di kernel per finestra a T medio ≈ 2 è ~0,4-0,5 ms sullo stadio 1 (gate/up −0,5, down +0,1) e ~0,4 sullo
   stadio 0: sul banco pulito è il **2% di 26 ms**, sotto la forbice del B-A-B (±2%). Sotto nsys il muro dello stadio 1
   è sceso di 1,1-2,5 ms per finestra, ma lì conta anche il rumore dell'host (`wait_flag`).
3. Perché non può essere di più: sullo stadio 1 gli esperti VRAM sono il 15-20% del muro (3-7 ms su 20-33), e il resto
   del tempo di una finestra non è nei kernel: 26 ms/finestra del B-A-B contro ~20 ms di muro del grafo a T = 2 (QSAROWS:
   ~6 ms fra verdetto, lancio della catena, F0→L1 e GPU-reach), 1,5-2,6 ms di `wait_flag` (l'host che pianifica il layer
   mentre la GPU aspetta; qui 2-7 ms), 3,5-5 ms di buchi fra i 1397 nodi del grafo. Un kernel esperti al 100% della
   banda toglierebbe al massimo altri ~1 ms per finestra (−4%); le voci piatte dell'host e del grafo valgono 2-3 volte
   tanto.

**Proposta.** (a) Tenere EXPDEC acceso nel candidato (bitwise, −20/−23% sul kernel gate/up a T ≥ 2, nessun costo
altrove) col nuovo default = solo gate/up; l'effetto atteso sul banco è −0,5 ms per finestra (~2%), misurabile solo con
più run o con nsys, non con un B-A-B singolo a ±2%. (b) Non aspettarsi dal lato esperti più del 2-4% per finestra: il
prossimo ms per finestra sta nei buchi fra i nodi (fusioni come `CHAINFUSE`), nell'attesa dell'host per layer
(`CRITPATH`/`DEVPLAN`) e nei ~6 ms host fra una finestra e l'altra. (c) Il lato pesi (§5, tile in shared) resta la via
per il gate/up a T = 1 (~1 ms piatto). (d) Il down Q2_0: capire perché nel motore perde (nsys con la catena del drafter
spenta, o `STRATA_EXPDEC_DR=2`), prima di riaccenderlo.

## Stato

Build di `ff8614a6` (codice; i commit successivi toccano solo questa pagina): compilata con `build_lock.py` (Release,
CUDA 75;86, test spenti), rc 0. `strata.exe` sha256
`4efc9f7422645b7b8b33f11c08fc3677c257956f27c2fd4a7ebf28d75dd81f16`, copiata con `build.json` in
`moe-aggressive-commit\docs\porto\strata_adattivo\corse_2080ti\20261004_routed_keep\build-expertdec-ff8614a6\`; accanto
`source\serve` e `source\tools` del worktree e `strata-vision.exe` di `build-candidate-5dea5258`. **Compilato sì,
micro-bench sì (`expdec_bench`, 2 iterazioni, bitwise), installato no, provato nel motore no.**

Build di `e4885f1d` (= `ff8614a6` + la riga di log `strata expdec:`): `strata.exe` sha256
`affd6c1ec680ab8336e0929aa80254266eda179db9974fa45bc9a38346100a0c`, in `build-expertdec-e4885f1d\` con `build.json`;
provata nel motore solo nelle catture nsys `DET-x2-expdec-nsys{A,B,B2}` (§7). Il B-A-B lo lancia l'owner.
