import os

#--> pilih dataset
opt : int = 2

#--> buat folder model
os.makedirs(f"model{opt}", exist_ok=True)

#--> konfigurasi save file
dataset_name   : str = f"data/landmarks_{opt}.csv"
binarizer_name : str = f"model{opt}/label_binarizer.pkl"
scaler_name    : str = f"model{opt}/scaler.pkl"
model_name     : str = f"model{opt}/model.h5"

#--> konfigurasi camera
cameras : int = 1