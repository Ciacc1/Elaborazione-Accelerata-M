#!/usr/bin/env python3
"""
Converti qualsiasi formato di immagine (JPG, PNG, BMP, GIF, etc.) in PPM
Prerequisito: pip install pillow
"""

import sys
from PIL import Image

def convert_to_ppm(input_file, output_file):
    """Converte un'immagine al formato PPM"""
    try:
        # Apri l'immagine con Pillow (supporta JPEG, PNG, BMP, GIF, etc.)
        img = Image.open(input_file)
        
        print(f"Immagine originale: {img.format}, Mode: {img.mode}, Size: {img.size}")
        
        # Converti a RGB se necessario (per supportare tutti i formati)
        if img.mode not in ('RGB', 'L', 'RGBA'):
            img = img.convert('RGB')
        
        print(f"Dopo conversione: Mode: {img.mode}, Size: {img.size}")
        
        # Salva in formato PPM
        # P6 = PPM RGB, P5 = PPM Grayscale
        img.save(output_file, format='PPM')
        
        print(f"✓ Convertito con successo in {output_file}")
        print(f"Ora esegui: ./image_to_pgm {output_file} output.pgm")
        return True
        
    except FileNotFoundError:
        print(f"Errore: File '{input_file}' non trovato")
        return False
    except Exception as e:
        print(f"Errore: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Uso: python3 convert_to_ppm.py <input_image> <output.ppm>")
        print("\nFormati supportati: JPG, PNG, BMP, GIF, TIFF, WebP, etc.")
        print("\nEsempi:")
        print("  python3 convert_to_ppm.py foto.jpg foto.ppm")
        print("  python3 convert_to_ppm.py immagine.png immagine.ppm")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if convert_to_ppm(input_file, output_file):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
