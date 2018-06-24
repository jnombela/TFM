

export GOOGLE_APPLICATION_CREDENTIALS='./Deep-1-cf9564fb42a8.json'

python3 TFM/Utilidades/generador_npy.py \
    --ruta_in data/train \
    --ruta_out1 data/trainx.npy \
    --ruta_out2 data/trainy.npy \
    --ruta_cla_bu Deep-1-cf9564fb42a8.json
