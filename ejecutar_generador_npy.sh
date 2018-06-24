

export GOOGLE_APPLICATION_CREDENTIALS='/media/storage/proyectos/claves/Deep-1-cf9564fb42a8.json'

python3 Utilidades/generador_npy.py \
    --ruta_in ./data500/train \
    --ruta_out1 data500/trainx.npy \
    --ruta_out2 data500/trainy.npy