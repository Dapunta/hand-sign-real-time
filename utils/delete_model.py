import os
from config import scaler_name, binarizer_name, model_name

#--> hapus model
list_file = [f'model/{i}' for i in [scaler_name, binarizer_name, model_name]]
for file in list_file:
    try:
        os.remove(file)
        print(f"File '{file}' berhasil dihapus.")
    except FileNotFoundError:
        print(f"File '{file}' tidak ditemukan.")
    except PermissionError:
        print(f"Tidak memiliki izin untuk menghapus '{file}'.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")