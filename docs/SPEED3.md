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

## Stato

- 12/10 12:40: codice, bench e doc scritti; **compilato no / provato no** (laboratorio in pausa).
