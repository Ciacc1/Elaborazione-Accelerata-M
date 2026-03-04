#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <time.h>

#define PI 3.14159265358979323846

typedef struct {
    double real;
    double imag;
} MyComplex;

typedef struct {
    int width;
    int height;
    int max_value;
    unsigned char *data;
} PGMImage;

PGMImage readPGM(const char *filename) {
    PGMImage img;
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Errore nell'aprire il file");
        exit(1);
    }

    char format[3];
    fscanf(file, "%s", format);
    if (format[0] != 'P' || format[1] != '5') {
        fprintf(stderr, "Formato non supportato.\n");
        exit(1);
    }

    fscanf(file, "%d %d", &img.width, &img.height);
    fscanf(file, "%d", &img.max_value);
    fgetc(file);

    img.data = (unsigned char *)malloc(img.width * img.height);
    fread(img.data, 1, img.width * img.height, file);
    fclose(file);

    return img;
}

void writePGM(const char *filename, PGMImage img) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Errore nell'aprire il file per la scrittura");
        exit(1);
    }

    fprintf(file, "P5\n%d %d\n%d\n", img.width, img.height, img.max_value);
    fwrite(img.data, 1, img.width * img.height, file);
    fclose(file);
}



void dft2D(unsigned char *in, MyComplex *out, int width, int height) {
    
    MyComplex sum;
    double angle;
    double pixel;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            
            sum.real = 0;
            sum.imag = 0;
            
            for (int v = 0; v < height; v++) {
                for (int u = 0; u < width; u++) {

                    angle = -2.0f * PI * ((double)x * u / width + (double)y * v / height);
                    pixel = in[v * width + u];

                    sum.real += pixel * cos(angle);
                    sum.imag += pixel * sin(angle);

                }
            }

            out[y * width + x] = sum;
        }
    }
}


void idft2D(MyComplex *in ,unsigned char *out, int width, int height) {
    
    double sum;
    double angle;
    MyComplex coeff;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            
            sum = 0;
            
            for (int v = 0; v < height; v++) {
                for (int u = 0; u < width; u++) {

                    angle = 2.0f * PI * ((double)x * u / width + (double)y * v / height);

                    coeff = in[v * width + u];
                    sum += coeff.real * cos(angle) - coeff.imag * sin(angle);
                }
            }

            sum /= (width * height);
            out[y * width + x] = (unsigned char)(sum < 0 ? 0 : (sum > 255 ? 255 : sum));
        
        }
    }
}

void filtro(MyComplex *dft, int width, int height, float cutoff) {
    
    int Xmezzi = width / 2;
    int Ymezzi = height / 2;
    float d;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            d = sqrt((x - Xmezzi) * (x - Xmezzi) + (y - Ymezzi) * (y - Ymezzi));
            
            if (d > cutoff) {
                dft[y * width + x].real = 0;
                dft[y * width + x].imag = 0;
            }

        }
    }
}


PGMImage dftToImage(MyComplex *dft, int width, int height) {
    PGMImage img;
    img.width = width;
    img.height = height;
    img.max_value = 255;
    img.data = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    double magnitude;
    double maxMagnitude = 0.0;
    double *magnitudes = (double *)malloc(width * height * sizeof(double));

    // Calcola la magnitudine per ogni pixel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            double real = dft[idx].real;
            double imag = dft[idx].imag;
            
            magnitude = sqrt(real * real + imag * imag);
            magnitudes[idx] = magnitude;
            
            if (magnitude > maxMagnitude) {
                maxMagnitude = magnitude;
            }
        }
    }

    // Normalizza e applica logaritmo per migliore contrasto
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            
            // Applica log per comprimere la dinamica dei valori
            double logMagnitude = log(1.0 + magnitudes[idx]);
            double logMaxMagnitude = log(1.0 + maxMagnitude);
            
            // Normalizza a [0, 255]
            double normalized = (logMagnitude / logMaxMagnitude) * 255.0;
            
            img.data[idx] = (unsigned char)(normalized > 255 ? 255 : (unsigned char)normalized);
        }
    }

    free(magnitudes);
    return img;
}

 // Sposta le componenti a frequenza zero al centro dell'immagine
 // Utile per visualizzare meglio lo spettro
 void fftshift(MyComplex *dft, int width, int height) {
     MyComplex *temp = (MyComplex *)malloc(width * height * sizeof(MyComplex));  
     int half_width = width / 2;
     int half_height = height / 2;  
     for (int y = 0; y < height; y++) {
         for (int x = 0; x < width; x++) {
             int src_x = (x + half_width) % width;
             int src_y = (y + half_height) % height;
             temp[y * width + x] = dft[src_y * width + src_x];
         }
     }  
     for (int i = 0; i < width * height; i++) {
         dft[i] = temp[i];
     }  
     free(temp);
 }


int main() {
    const char *inputFile = "immaginiInput/Bologna-256.pgm";
    //const char *outputFile = "output_sequenziale-256.pgm";
    const char *dftImageFile = "output_dft_spectrum-256.pgm";
    const char *dftShiftedImageFile = "output_dft_spectrum_shifted-256.pgm";
    
    float raggioFiltro = 130.0f; // 130 per 256 px, 260 per 512 px

    clock_t start, end;
    double tempo;

    PGMImage img = readPGM(inputFile);
    printf("Immagine caricata di dimensioni H:%d W:%d\n", img.height, img.width);

    MyComplex *dft = (MyComplex *)malloc(img.width * img.height * sizeof(MyComplex));

    //-------------------------------------- DFT
    start = clock();

    dft2D(img.data, dft, img.width, img.height);
    
    end = clock();
    tempo = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Tempo DFT: %f secondi\n", tempo);
    //--------------------------------------

    // Salva lo spettro DFT come immagine (prima del shift)
    printf("Conversione dello spettro DFT in immagine...\n");
    PGMImage dftImage = dftToImage(dft, img.width, img.height);
    writePGM(dftImageFile, dftImage);
    printf("Spettro salvato in %s\n", dftImageFile);
    free(dftImage.data);

    // Copia DFT per il shift
    MyComplex *dft_shifted = (MyComplex *)malloc(img.width * img.height * sizeof(MyComplex));
    for (int i = 0; i < img.width * img.height; i++) {
        dft_shifted[i] = dft[i];
    }

    // Applica FFT shift e salva
    fftshift(dft_shifted, img.width, img.height);
    PGMImage dftImageShifted = dftToImage(dft_shifted, img.width, img.height);
    writePGM(dftShiftedImageFile, dftImageShifted);
    printf("Spettro centrato salvato in %s\n", dftShiftedImageFile);
    free(dftImageShifted.data);
    free(dft_shifted);

    //-------------------------------------- FILTRO
    start = clock(); 
    
    filtro(dft, img.width, img.height, raggioFiltro);

    end = clock();
    tempo = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Tempo filtro: %f secondi\n", tempo);
    //--------------------------------------

    //-------------------------------------- iDFT
    //start = clock();

    //idft2D(dft, img.data, img.width, img.height);

    //end = clock();
    //tempo = ((double)(end - start)) / CLOCKS_PER_SEC;
    //printf("Tempo iDFT: %f secondi\n", tempo);
    //-------------------------------------- 

    //writePGM(outputFile, img);

    free(dft);
    free(img.data);

    printf("Risultato salvato in %s\n", dftImageFile);

    return 0;
}