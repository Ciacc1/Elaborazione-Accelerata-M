#!/bin/bash

# Script di esempio per compilare ed eseguire in Colab

echo "========== CUDA Edge Detection via FFT =========="

# Controlla se CUDA è disponibile
echo "Checking CUDA installation..."
/usr/local/cuda/bin/nvcc --version

# Compila
echo -e "\nCompiling..."
make clean
make

if [ ! -f ./edge_detection ]; then
    echo "Compilation failed!"
    exit 1
fi

echo -e "\nCompilation successful!"

# Se non hai un'immagine di test, creane una semplice (PPM)
if [ ! -f input.ppm ]; then
    echo -e "\nCreating test image (512x512 gradient)..."
    python3 << 'EOF'
import struct

width, height = 512, 512

with open("input.ppm", "wb") as f:
    # Header PPM
    f.write(b"P6\n")
    f.write(f"{width} {height}\n".encode())
    f.write(b"255\n")
    
    # Crea un'immagine con gradiente e pattern
    for y in range(height):
        for x in range(width):
            r = int((x / width) * 255)
            g = int((y / height) * 255)
            b = int(((x + y) / (width + height)) * 255)
            f.write(struct.pack('BBB', r, g, b))

print(f"Test image created: {width}x{height}")
EOF
fi

# Esegui con filtro Laplacian
echo -e "\nRunning Laplacian edge detection..."
./edge_detection input.ppm output_laplacian.ppm laplacian

# Esegui con filtro Sobel
echo -e "\nRunning Sobel edge detection..."
./edge_detection input.ppm output_sobel.ppm sobel

echo -e "\nDone! Output files:"
echo "  - output_laplacian.ppm"
echo "  - output_sobel.ppm"

# Visualizza risultati (se sei in Colab)
python3 << 'EOF'
try:
    from PIL import Image
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    img_orig = Image.open("input.ppm")
    img_laplacian = Image.open("output_laplacian.ppm")
    img_sobel = Image.open("output_sobel.ppm")
    
    axes[0].imshow(img_orig)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(img_laplacian, cmap='gray')
    axes[1].set_title("Laplacian")
    axes[1].axis('off')
    
    axes[2].imshow(img_sobel, cmap='gray')
    axes[2].set_title("Sobel")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("edge_detection_results.png", dpi=100, bbox_inches='tight')
    plt.show()
    
    print("\nResults saved as edge_detection_results.png")
except ImportError:
    print("PIL/matplotlib not available for visualization")
EOF
