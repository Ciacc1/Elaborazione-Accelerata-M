#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <dirent.h>
#include <sys/stat.h>
#include <vector>
#include <string.h>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#define TILE_W 16
#define TILE_H 16
#define NUM_SLOTS 8 // Numero di buffer in memoria pinned per evitare starvation di I/O

typedef struct {
    unsigned char *h_in;
    unsigned char *h_out;
} BufferSlot;

typedef struct {
    int slot_idx;
    int frame_idx;
    std::string in_path;
    std::string out_path;
} PipelineTask;

// Classe Thread-Safe Queue per la gestione della pipeline multithread
template <typename T>
class SafeQueue {
private:
    std::queue<T> q;
    std::mutex m;
    std::condition_variable cv;
public:
    void push(T val) {
        std::lock_guard<std::mutex> lock(m);
        q.push(val);
        cv.notify_one();
    }
    
    T pop() {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this] { return !q.empty(); });
        T val = q.front();
        q.pop();
        return val;
    }
};

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// ─────────────────────────────────────────────────────────────────────────────
// Funzioni di I/O PGM ottimizzate per buffer preallocati
// ─────────────────────────────────────────────────────────────────────────────

void readPGM_to_buffer(const char *filename, unsigned char* buffer) {
    FILE *file = fopen(filename, "rb");
    if (!file) { fprintf(stderr, "Errore apertura file lettura: %s\n", filename); exit(1); }
    char format[3];
    (void)fscanf(file, "%s", format);
    int width, height, max_value;
    (void)fscanf(file, "%d %d", &width, &height);
    (void)fscanf(file, "%d",    &max_value);
    fgetc(file);
    (void)fread(buffer, 1, width * height, file);
    fclose(file);
}

void writePGM_from_buffer(const char *filename, const unsigned char* buffer, int width, int height) {
    FILE *file = fopen(filename, "wb");
    if (!file) { fprintf(stderr, "Errore apertura file scrittura: %s\n", filename); exit(1); }
    fprintf(file, "P5\n%d %d\n255\n", width, height);
    fwrite(buffer, 1, width * height, file);
    fclose(file);
}

// ─────────────────────────────────────────────────────────────────────────────
// CUDA Kernels per cuFFT
// ─────────────────────────────────────────────────────────────────────────────

__global__ void uchar_to_complex(const unsigned char* in, cufftComplex* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        out[y * width + x].x = (float)in[y * width + x];
        out[y * width + x].y = 0.0f;
    }
}

__global__ void filter_cufft(cufftComplex* dft, int width, int height, float cutoff2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float dx = (x < width / 2) ? (float)x : (float)(width - x);
        float dy = (y < height / 2) ? (float)y : (float)(height - y);
        
        if (dx*dx + dy*dy < cutoff2) {
            dft[y * width + x].x = 0.0f;
            dft[y * width + x].y = 0.0f;
        }
    }
}

__global__ void complex_to_uchar(const cufftComplex* in, unsigned char* out, int width, int height, float invN) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float val = in[y * width + x].x * invN; 
        out[y * width + x] = (unsigned char)(fmaxf(0.0f, fminf(255.0f, val)));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Thread Functors per Asynchronous I/O
// ─────────────────────────────────────────────────────────────────────────────

void reader_thread_func(const std::vector<std::string>& files, const std::string& out_dir,
                        BufferSlot* slots, SafeQueue<int>& free_slots, SafeQueue<PipelineTask>& read_queue) {
    for (size_t i = 0; i < files.size(); i++) {
        int slot_idx = free_slots.pop(); // Attende che ci sia un buffer libero
        
        readPGM_to_buffer(files[i].c_str(), slots[slot_idx].h_in);
        
        char path_edge[512];
        snprintf(path_edge, sizeof(path_edge), "%s/edge_%04d.pgm", out_dir.c_str(), (int)i);
        
        PipelineTask task = { slot_idx, (int)i, files[i], std::string(path_edge) };
        read_queue.push(task);
    }
}

void writer_thread_func(int total_frames, int width, int height, BufferSlot* slots,
                        SafeQueue<PipelineTask>& write_queue, SafeQueue<int>& free_slots) {
    for (int i = 0; i < total_frames; i++) {
        PipelineTask task = write_queue.pop(); // Attende un fotogramma elaborato dalla GPU
        
        writePGM_from_buffer(task.out_path.c_str(), slots[task.slot_idx].h_out, width, height);
        
        free_slots.push(task.slot_idx); // Libera lo slot per il Reader thread
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Execution
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <cartella_input_pgm> <cartella_output>\n", argv[0]);
        return 1;
    }

    if (mkdir(argv[2], 0755) != 0 && errno != EEXIST) {
        perror("Errore creazione cartella output"); return 1;
    }

    DIR *dir = opendir(argv[1]);
    if (!dir) { perror("Errore apertura cartella input"); return 1; }
    struct dirent *entry;
    std::vector<std::string> files;
    while ((entry = readdir(dir)) != NULL) {
        std::string name(entry->d_name);
        if (name.find(".pgm") != std::string::npos)
            files.push_back(std::string(argv[1]) + "/" + name);
    }
    closedir(dir);
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        fprintf(stderr, "Nessun file PGM trovato.\n");
        return 1;
    }

    // Legge le dimensioni del primo frame
    FILE *f_first = fopen(files[0].c_str(), "rb");
    char fmt[3]; (void)fscanf(f_first, "%s", fmt);
    int W, H, max_v; (void)fscanf(f_first, "%d %d %d", &W, &H, &max_v);
    fclose(f_first);
    int N = W * H;

    printf("=== ULTIMATE HYPER-OPTIMIZED VIDEO PIPELINE ===\n");
    printf("Risoluzione: %dx%d (%d pixel)\n", W, H, N);
    printf("Frame totali: %d\n", (int)files.size());
    printf("Architettura: Triple Pipelining (Read Thread || GPU Multi-Stream || Write Thread)\n");
    printf("Motore Matematico: cuFFT O(N log N)\n\n");

    dim3 threads(TILE_W, TILE_H);
    dim3 blocks((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);

    float cutoff2 = 30.0f * 30.0f;   
    float invN    = 1.0f / (float)N;  

    // Code di sincronizzazione inter-thread
    SafeQueue<int> free_slots;
    SafeQueue<PipelineTask> read_queue;
    SafeQueue<PipelineTask> write_queue;

    // Allocazione dello Slot Ring in memoria Host Pinned
    BufferSlot slots[NUM_SLOTS];
    for (int i = 0; i < NUM_SLOTS; i++) {
        CHECK(cudaMallocHost(&slots[i].h_in, N));
        CHECK(cudaMallocHost(&slots[i].h_out, N));
        free_slots.push(i); // All'inizio tutti gli slot sono liberi
    }

    // Allocazioni Device (2 stream paralleli)
    unsigned char *d_in[2], *d_edge[2];
    cufftComplex  *d_comp[2];
    cudaStream_t stream[2];
    cufftHandle plan[2];

    for (int b = 0; b < 2; b++) {
        CHECK(cudaMalloc(&d_in[b],   N));
        CHECK(cudaMalloc(&d_edge[b],     N));
        CHECK(cudaMalloc(&d_comp[b],     N * sizeof(cufftComplex)));
        CHECK(cudaStreamCreate(&stream[b]));
        cufftPlan2d(&plan[b], H, W, CUFFT_C2C);
        cufftSetStream(plan[b], stream[b]);
    }

    // Avvio dei Thread dedicati esclusivamente all'I/O su Disco
    std::thread reader_thread(reader_thread_func, std::ref(files), std::string(argv[2]), slots, std::ref(free_slots), std::ref(read_queue));
    std::thread writer_thread(writer_thread_func, (int)files.size(), W, H, slots, std::ref(write_queue), std::ref(free_slots));

    // Timer ad alta precisione CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // ── LOOP PRINCIPALE (GESTITO INTERAMENTE DALLA GPU) ─────────────────────
    PipelineTask active_tasks[2];
    bool stream_active[2] = {false, false};

    for (int i = 0; i < (int)files.size(); i++) {
        int sm = i % 2; // Alterna gli stream CUDA

        // Se lo stream corrente ha un'elaborazione pendente, attendiamo che finisca 
        // e la passiamo immediatamente al Writer Thread senza bloccare il disco.
        if (stream_active[sm]) {
            CHECK(cudaStreamSynchronize(stream[sm]));
            write_queue.push(active_tasks[sm]);
            stream_active[sm] = false;
        }

        // Estrae il prossimo fotogramma già letto dal disco in memoria pinned
        PipelineTask task = read_queue.pop();
        int slot = task.slot_idx;

        // Sottomissione asincrona totale sulla GPU
        CHECK(cudaMemcpyAsync(d_in[sm], slots[slot].h_in, N, cudaMemcpyHostToDevice, stream[sm]));
        
        uchar_to_complex<<<blocks, threads, 0, stream[sm]>>>(d_in[sm], d_comp[sm], W, H);
        cufftExecC2C(plan[sm], d_comp[sm], d_comp[sm], CUFFT_FORWARD);
        filter_cufft<<<blocks, threads, 0, stream[sm]>>>(d_comp[sm], W, H, cutoff2);
        cufftExecC2C(plan[sm], d_comp[sm], d_comp[sm], CUFFT_INVERSE);
        complex_to_uchar<<<blocks, threads, 0, stream[sm]>>>(d_comp[sm], d_edge[sm], W, H, invN);
        
        CHECK(cudaMemcpyAsync(slots[slot].h_out, d_edge[sm], N, cudaMemcpyDeviceToHost, stream[sm]));

        active_tasks[sm] = task;
        stream_active[sm] = true;
    }

    // Svuotamento della pipeline per gli ultimi frame rimasti negli stream
    for (int sm = 0; sm < 2; sm++) {
        if (stream_active[sm]) {
            CHECK(cudaStreamSynchronize(stream[sm]));
            write_queue.push(active_tasks[sm]);
        }
    }

    // Attende la terminazione dei thread di I/O
    reader_thread.join();
    writer_thread.join();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("\nElaborazione completata con successo!\n");
    printf("Tempo totale di esecuzione: %.3f secondi.\n", ms / 1000.0f);
    printf("Fps medi raggiunti: %.2f fps\n", (float)files.size() / (ms / 1000.0f));

    // ── CLEANUP ──────────────────────────────────────────────────────────────
    for (int b = 0; b < 2; b++) {
        cufftDestroy(plan[b]);
        cudaFree(d_in[b]); cudaFree(d_edge[b]); cudaFree(d_comp[b]);
        cudaStreamDestroy(stream[b]);
    }
    for (int i = 0; i < NUM_SLOTS; i++) {
        cudaFreeHost(slots[i].h_in);
        cudaFreeHost(slots[i].h_out);
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
