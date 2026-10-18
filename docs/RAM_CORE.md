# RAM_CORE: nucleo fisso in VRAM senza copia in RAM (`STRATA_RAM_CORE=1`)

Branch `lab/core-ram` del fork `imanu86/Strata-2080Ti`, da `lab/upsync`. Opzione di laboratorio, spenta di default, solo
`--serve` con l'arena residente. Obiettivo dell'owner: almeno 10 GB di RAM in meno senza perdere velocità, perché il PC
non è un server dedicato.

## L'idea

L'arena in RAM (`ArenaExpertSource`, ~40 GiB) tiene tutti i 24.576 esperti, anche quelli che le due schede hanno già in
VRAM. La dedup V5 (`lab/wip-0809`, `ram_dedup.*`) e quella di Fable (`lab/ram-dedup`) liberavano la RAM di *tutti* i
residenti e la ricopiavano dalla VRAM (D2H) quando un esperto usciva. Nella misura V5 ha dato −17 GiB ma B 50,3/45,5 contro
A 58,2/44,6 t/s: le copie di ritorno cadono proprio quando servono (scambi dell'adattamento, prestiti al prompt, shrink).

Qui niente copie di ritorno. Su ogni scheda c'è un **nucleo fisso** di slot, i più bassi, che all'avvio il riempimento dal
profilo (`--expert-profile`) assegna agli esperti più chiamati. Quel nucleo non cambia mai:

- **adattamento**: nessuno dei due tier (il blocco `adapt` e quello asincrono `a_pick`) sceglie una vittima nel nucleo,
  nemmeno con `STRATA_ELASTIC_CORE_ADAPT`/pipeline, che altrimenti rende scambiabile il core dell'elastica;
- **prestiti alla prefill**: il prestito prende solo slot sopra il nucleo (`PfPart::rc_floor`, `el_lend_floor` per le
  ripianificazioni della 3060);
- **elastica**: lo shrink non scende sotto il nucleo e non ci sposta esperti della coda (`el_fixed`);
- **K/V elastica** (`--kv-grow`): `kvg.floor` non scende sotto il nucleo;
- comando `VRAM`: il restringimento manuale tiene almeno il nucleo.

Poiché il nucleo non esce mai, la sua copia in RAM non serve: dopo l'avvio, prima della prima richiesta, le pagine intere
dei loro blob vengono restituite (`VirtualUnlock` + `VirtualFree(MEM_DECOMMIT)`; su Linux `munlock` + `MADV_DONTNEED`). Lo
spazio di indirizzi resta riservato, quindi `blob()` resta `base + offset`. Si restituiscono solo le pagine interne (dopo i
primi 256 byte del blob e prima della pagina condivisa col successivo): una copia del vicino arrotondata a 256 B non tocca
mai pagine tolte. Il minimo del working set, alzato dal lock dell'arena, viene abbassato di quanto restituito.

La parte variabile (coda dell'elastica, parte prestabile, slot scambiabili) resta com'è, con la sua copia RAM.

**Rete di sicurezza.** Se un percorso chiedesse comunque `blob()` di un esperto del nucleo, l'arena lo rilegge dal file del
modello (experts.bin o le tre parti del GGUF nativo), lo rimette nella sua posizione e conta l'evento: `riletti` nel log.
Deve restare 0; l'output resta comunque identico perché i byte riletti sono gli stessi.

**Output.** Greedy identico bit per bit: cambia solo dove stanno i byte, non quali esperti girano in GPU. Un nucleo più
grande del default sulla 2080 Ti cambia però *quali* esperti l'adattamento può scambiare, quindi da un confronto A/B senza
banco forzato possono uscire testi diversi pur restando entrambi corretti.

## Il nucleo per scheda

| Scheda | Default | Variabile |
|---|---|---|
| CUDA0 (3060, stadio 0) | la parte del core elastico (`core N slots fixed`) che la prefill non prende in prestito: `min(el_core, primo slot prestato)` | `STRATA_RAM_CORE_SLOTS_DEV0=N` |
| CUDA1 (2080 Ti, stadio 1) | 3500 slot (~6,1 GiB), comunque sotto il primo slot prestato | `STRATA_RAM_CORE_SLOTS_DEV1=N` |

Con un valore esplicito il nucleo viene chiesto **prima** di pianificare il prestito: se supera la parte non prestata, il
prestito si riduce (chunk più piccolo, prompt più lento). Sulla 3060 il valore esplicito non supera mai il core elastico,
perché sopra la coda può restringersi. Con l'opzione accesa ma senza le condizioni (arena mappata, large pages, peer/remote
expert, niente tabella di residenza) il log dice `strata ram core: NOT armed: <motivo>` e tutto resta come prima.

**Il prefisso registrato.** Sotto WDDM con due schede i primi ~8 GiB dell'arena (10 layer: 0-9) sono registrati con
`cudaHostRegister`; quelle pagine non si possono restituire e i loro esperti restano in RAM (`tenuti nel prefisso
registrato` nel log). Riguarda solo la 3060: i suoi layer 10-15 si liberano, i layer 0-9 no.

## Log

All'avvio, dopo il riempimento e prima della prima richiesta:

```
strata ram core:   dev0: nucleo fisso slot [0, N) di S (N esperti, X GiB in VRAM); a senza copia RAM, b tenuti nel prefisso registrato (...)
strata ram core:   dev1: nucleo fisso slot [0, M) di S1 (...)
strata ram core:   pagine restituite in T ms; privata P0 -> P1 GiB, working set W0 -> W1 GiB
strata ram core: dev0 a esperti (X GiB) senza copia RAM, dev1 c (Y GiB); arena da A a B GiB; riletti 0 (avvio; fuori posto 0; scambi dell'adattamento 0; privata ..., working set ...)
```

La stessa riga a fine richiesta (`fine richiesta`): `riletti` e `fuori posto` (esperti del nucleo non più nel loro slot)
devono restare 0; `scambi dell'adattamento` sono gli scambi fatti dal blocco `adapt` dalla riga precedente.

## Come si misura

1. **RAM**: privata e working set di `strata.exe` dopo la richiesta di riscaldamento, A (senza) contro B (con
   `STRATA_RAM_CORE=1`): `Get-Process strata | Select PrivateMemorySize64, WorkingSet64`, la colonna «Commit» del Task
   Manager, e le righe `strata ram core:` (privata prima/dopo, arena da A a B). Obiettivo: ≥ 10 GB in meno.
2. **Velocità**: B-A-B col banco forzato (`STRATA_FORCE_IDS` / `STRATA_FORCE_WINDOWS` dalla traccia di A): stessi token,
   stesse finestre; si confrontano le righe DONE (ms di decode, t/s) e le `T <id>`, che devono essere identiche.
3. **Cache**: `strata serve: decode expert cache hit rate` per richiesta, A contro B. Il nucleo toglie flessibilità
   all'adattamento sulla 2080 Ti: è il primo numero da guardare se B rallenta.
4. **Adattamento**: `scambi dell'adattamento` per richiesta nella riga `strata ram core`, contro A (in A la riga non c'è:
   si confrontano hit rate e tempi; con `STRATA_RAM_CORE=1 STRATA_RAM_CORE_SLOTS_DEV0=0 STRATA_RAM_CORE_SLOTS_DEV1=0` si ha
   lo stesso motore con il conteggio e nessun nucleo).
5. `riletti 0` e `fuori posto 0` in ogni riga.

## Stima del risparmio (configurazione del 7/10: 3060 ~3.800-4.600 slot da ~1,43 MiB, 2080 Ti 9.150 slot da ~1,78 MiB)

- 2080 Ti, 3500 slot: ~6,1 GiB (tutti fuori dal prefisso registrato).
- 3060: nucleo = core elastico non prestato, ~1.200-2.000 slot (2,8 GiB al massimo); se ne liberano solo i layer 10-15,
  ~0,6-1,1 GiB.
- Totale ai default: **~6,7-7,2 GiB (7,2-7,7 GB)**, sotto l'obiettivo.
- Per arrivare a ≥ 10 GB: `STRATA_RAM_CORE_SLOTS_DEV1=5500` (~9,6 GiB sulla 2080 Ti: restano 3.650 slot che ruotano,
  2.170 dei quali prestabili) → ~10,2-10,7 GiB in tutto. Il primo slot prestato della 2080 Ti è ~6.980, quindi il prestito
  non cambia.

## Rischi

- **Meno flessibilità dell'adattamento**: gli esperti del nucleo non sono più vittime. Se una conversazione usa poco un
  esperto caldo del profilo, il suo slot resta occupato. Sulla 3060 la pipeline rendeva scambiabile tutto il core elastico
  (`STRATA_ELASTIC_CORE_ADAPT`): ora la parte non prestata non lo è più.
- **Interazione coi prestiti**: un nucleo esplicito più grande della parte non prestata riduce il prestito e rallenta i
  prompt lunghi (sulla 3060 il prestito del core aveva portato un prompt da 131k da 134 a ~85 s).
- **Shrink elastico**: la 3060 non può scendere sotto `max(el_core, nucleo)`; con la sola coda come margine, sotto
  pressione di VRAM la coda finisce prima.
- **Percorsi sconosciuti**: un percorso che legge la RAM di un esperto residente senza passare da `blob()` (puntatore
  calcolato a mano, alias device) leggerebbe pagine tolte. Oggi l'alias device esiste solo nel prefisso registrato, che non
  si tocca; la rete di sicurezza copre `blob()`/`blob_stable()`.
- **Prefisso registrato**: limita il risparmio sulla 3060. Spostare o ridurre la registrazione (`STRATA_ARENA_PIN_GIB`)
  libererebbe di più, ma cambia la quota PCIe e le copie DMA dei prompt: da misurare a parte.

## Stato

Compilato sul ramo `lab/core-ram` (Release, sm_75+86, `build_lock.py`); non installato, non provato sulle schede. Prima
verifica: una richiesta di riscaldamento con
`STRATA_RAM_CORE=1` che deve mostrare `riletti 0`, `fuori posto 0` e righe `T <id>` identiche alla traccia di riferimento.
