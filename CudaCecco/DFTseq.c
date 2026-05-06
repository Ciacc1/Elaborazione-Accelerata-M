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

            float dx = (x < width / 2) ? x : (width - x);
            float dy = (y < height / 2) ? y : (height - y);
            float d = sqrtf(dx * dx + dy * dy);

            // EDGE DETECTION (Filtro Passa-Alto)
            if (d < cutoff) {
                dft[y * width + x].real = 0.0f;
                dft[y * width + x].imag = 0.0f;
            }

        }
    }
}


int main() {
    const char *inputFile = "immagini/512.pgm";
    const char *outputFile = "output_sequenziale-512.pgm";
    float raggioFiltro = 30.0f; 

    clock_t start, end;
    double tempo;

    PGMImage img = readPGM(inputFile);
    printf("Immagine caricata di dimensioni H:%d W:%d\n", img.height, img.width);

    MyComplex *dft = (MyComplex *)malloc(img.width * img.height * sizeof(MyComplex));

    //-------------------------------------- DFT
    start = clock();

    dft2D(img.data, dft, img.width, img.height);
    
    end = clock();
    tempo = ((double)(end - start)) / CLOCKS_PER_SEC ;
    printf("Tempo DFT: %f\n", tempo);
    //-------------------------------------- 

    //-------------------------------------- FILTRO
    start = clock(); 
    
    filtro(dft, img.width, img.height, raggioFiltro);

    end = clock();
    tempo = ((double)(end - start)) / CLOCKS_PER_SEC ;
    printf("Tempo filtro: %f\n", tempo);
    //--------------------------------------

    //-------------------------------------- iDFT
    start = clock();

    idft2D(dft, img.data, img.width, img.height);

    end = clock();
    tempo = ((double)(end - start)) / CLOCKS_PER_SEC ;
    printf("Tempo iDFT: %f\n", tempo);
    //-------------------------------------- 

    writePGM(outputFile, img);

    free(dft);
    free(img.data);

    printf("Risultato salvato in %s\n", outputFile);

    return 0;
}
