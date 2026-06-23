# Edge Detection tramite Trasformata Discreta di Fourier (DFT) con CUDA

Progetto sviluppato per il corso di **Sistemi di Elaborazione Accelerata**.

## Abstract
Il progetto si focalizza sull'implementazione e sull'ottimizzazione parallela su GPU (architettura CUDA) dell'algoritmo di *Edge Detection* applicato a immagini e flussi video, sfruttando la Trasformata Discreta di Fourier (DFT) e la sua inversa (IDFT). Partendo da una baseline sequenziale in C, sono state applicate iterativamente ottimizzazioni aritmetiche e architetturali. La versione finale ottimizzata e l'estensione video basata su CUDA Streams hanno permesso di nascondere interamente i tempi di trasferimento PCIe e I/O, massimizzando il throughput della GPU (scenari *compute-bound*) e raggiungendo uno speedup massimo superiore a 2000x rispetto alla CPU.

## Autori
* **Francesco Moretti**
* **Giacomo Sampaoli**

## Struttura del Progetto e Ottimizzazioni
Il codice segue un percorso di ottimizzazione incrementale analizzato tramite *Nsight Systems* e *Nsight Compute*:

1. **Baseline Sequenziale:** Versione Naïve in C su CPU per il calcolo di DFT, IDFT e filtraggio.
2. **CUDA Naïve:** Prima parallelizzazione elementare su GPU.
3. **Ottimizzazioni Aritmetiche:** Passaggio da precisione `double` a `float` e introduzione delle istruzioni FMA (*Fused Multiply-Add*).
4. **Dimensionamento Blocchi & Shared Memory:** Ottimizzazione delle dimensioni dei blocchi e riduzione degli accessi in memoria globale tramite memoria condivisa e coalescenza.
5. **Loop Unrolling & Cache L1:** Riduzione dell'overhead dei cicli di clock interni al kernel.
6. **Estensione Video (CUDA Streams):** Elaborazione asincrona dei fotogrammi mediante *double buffering* e *kernel fusion*, azzerando i tempi morti causati dal bus PCIe e dal file system.

## Requisiti e Strumenti
* **Hardware:** GPU NVIDIA compatibile con CUDA.
* **Software:** CUDA Toolkit (compilatore `nvcc`), GCC.
* **Profilazione:** NVIDIA Nsight Systems e NVIDIA Nsight Compute.

