#!/bin/bash
# Converte il file pgm dato in ingresso in un file png e lo inserisce nella cartella out_png

# Controllo input
if [ $# -ne 1 ]; then
    echo "Uso: $0 file.pgm"
    exit 1
fi

input="$1"

# Controlla che il file esista
if [ ! -f "$input" ]; then
    echo "File non trovato: $input"
    exit 1
fi

# Crea cartella output se non esiste
mkdir -p out_png

# Estrae solo il nome file senza percorso
filename=$(basename "$input")

# Rimuove estensione .pgm
name="${filename%.pgm}"

# Percorso output
output="out_png/${name}.png"

# Conversione
magick "$input" "$output"

echo "Creato: $output"