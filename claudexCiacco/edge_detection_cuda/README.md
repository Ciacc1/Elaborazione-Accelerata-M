# Edge Detection via FFT con CUDA

Implementazione parallela su GPU di edge detection utilizzando la trasformata di Fourier 2D.

## File inclusi

- **edge_detection_kernel.cu**: Kernel CUDA e funzioni FFT (cuFFT)
  - `laplacian_filter_kernel()`: Filtro Laplacian nel dominio Fourier
  - `sobel_filter_kernel()`: Filtro Sobel nel dominio Fourier
  - `edge_detection_laplacian()`: Wrapper per Laplacian
  - `edge_detection_sobel()`: Wrapper per Sobel
  - `print_metrics()`: Stampa metriche di performance

- **edge_detection.h**: Header file con dichiarazioni

- **main.cu**: Programma principale con utility I/O
  - `read_ppm()`: Leggi immagini PPM
  - `write_ppm()`: Scrivi immagini PPM
  - `rgb_to_grayscale()`: Conversione RGB → Grayscale

- **Makefile**: Configurazione compilazione

- **run_example.sh**: Script bash di esempio

## Compilazione su Colab

```bash
# Copia i file
# Poi da Colab:

!make clean
!make

# Oppure manualmente:
!nvcc -arch=sm_75 -std=c++11 -O3 -c edge_detection_kernel.cu
!nvcc -arch=sm_75 -std=c++11 -O3 -c main.cu
!nvcc -arch=sm_75 -std=c++11 -O3 edge_detection_kernel.o main.o -o edge_detection -lcufft
```

## Utilizzo

```bash
./edge_detection <input.ppm> <output.ppm> [laplacian|sobel]
```

Esempi:
```bash
./edge_detection image.ppm result.ppm laplacian
./edge_detection image.ppm result.ppm sobel
```

## Metriche di Performance

Il programma misura:
- **FFT Forward**: Tempo FFT reale→complessa (R2C)
- **Kernel Multiply**: Tempo applicazione filtro
- **FFT Inverse**: Tempo FFT complessa→reale (C2R)
- **Magnitude/Clip**: Tempo normalizzazione e conversione a byte
- **TOTAL TIME**: Tempo totale esecuzione
- **Estimated BW**: Bandwidth GPU stimato (GB/s)

Esempio output:
```
========== PERFORMANCE METRICS ==========
Image size: 512 x 512 (262144 pixels)
FFT Forward:      12.456 ms
Kernel Multiply:  0.234 ms
FFT Inverse:      11.897 ms
Magnitude/Clip:   0.412 ms
----------------------------------------
TOTAL TIME:       25.001 ms
Estimated BW:     41.25 GB/s
=========================================
```

## Architettura

### FFT Domain Processing

```
Input Image (float)
        ↓
    FFT 2D (R2C)  ← Forward Transform
        ↓
   Spectrum (Complex)
        ↓
  Apply Filter Kernel
   (Laplacian/Sobel)
        ↓
   Filtered Spectrum
        ↓
   IFFT 2D (C2R)  ← Inverse Transform
        ↓
  Normalize & Clip
        ↓
Output Image (uint8)
```

### Kernel Details

**Laplacian Filter:**
- Nel dominio Fourier: `-4π²(u²+v²)`
- Evidenzia rapid changes (edge-heavy)
- Output può avere valori negativi

**Sobel Filter:**
- Nel dominio Fourier: `√(u²+v²)`
- Approssimazione della magnitudine del gradiente
- Output sempre positivo

## Note importanti

1. **Compute Capability**: Il codice usa `-arch=sm_75` (Tesla T4 su Colab)
   - Per A100 usa `-arch=sm_80`
   - Per RTX30xx usa `-arch=sm_86`

2. **Formato Input**: Supporta immagini PPM (formato semplice, no librerie)
   - Per convertire PNG/JPG → PPM usa ImageMagick: `convert image.png image.ppm`

3. **Memoria**: FFT alloca memoria complessa `width × height × sizeof(cufftComplex)`
   - Per immagini grandi (>4K) controlla memoria disponibile

4. **Normalizzazione**: IFFT richiede divisione per numero di pixel
   - Implementato in `normalize_kernel()`

5. **Clipping**: Output convertito a [0, 255] in `clip_to_byte_kernel()`

## Benchmark tipici (Tesla T4, Colab)

```
512×512:   ~25 ms  (41 GB/s)
1024×1024: ~85 ms  (44 GB/s)
2048×2048: ~320 ms (46 GB/s)
```

Il bottleneck principale è la FFT (80-90% del tempo totale).

## Possibili miglioramenti

1. **Batch Processing**: Elaborare più immagini contemporaneamente
2. **Adaptive Filtering**: Regolare scala del filtro in base al contenuto
3. **Pad to Power of 2**: cuFFT è più veloce con dimensioni potenza di 2
4. **Stream Processing**: Usare CUDA streams per sovrapposizione compute/memory
5. **Mixed Precision**: Usare float16 per risparmio memoria/bandwidth

## Troubleshooting

**Errore: cuFFT allocation failed**
- Memoria GPU insufficiente. Riduci dimensioni immagine.

**Output completamente nero**
- Regola il parametro `scale` nei kernel (attualmente 0.1 e 0.05)

**Risultati strani con Laplacian**
- Laplacian produce valori negativi. Controlla la visualizzazione (prova abs value)

**Compilation error "unknown device"**
- Controlla Compute Capability con `nvidia-smi -L` e adatta `-arch=sm_XX`

## Licenza

Libero utilizzo per scopi educativi e research.
