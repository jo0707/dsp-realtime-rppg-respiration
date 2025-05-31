# Real-Time Respiration and rPPG Signal Extraction

Proyek ini merupakan bagian dari tugas akhir mata kuliah **IF3024 - Pengolahan Sinyal Digital** yang diampu oleh bapak [Martin Clinton Tosima Manullang, Ph.D](https://mctm.web.id/) di Program Studi Teknik Informatika ITERA. Aplikasi ini bertujuan untuk mengekstraksi sinyal respirasi dan sinyal **remote-photoplethysmography (rPPG)** secara real-time dari input video webcam.

## 🎯 Tujuan Proyek

Mengembangkan sistem yang mampu:

-   Mengambil input video dari webcam
-   Mengekstrak sinyal respirasi dan sinyal rPPG secara real-time
-   Menampilkan visualisasi sinyal secara langsung menggunakan `matplotlib` dan/atau `cv2`

---

## 👨‍💻 Anggota Kelompok

| Nama Lengkap         | NIM       | GitHub ID     |
| -------------------- | --------- | ------------- |
| Joshua Palti Sinaga  | 122140141 | @jo0707       |
| Irma Amelia Novianti | 122140128 | @irmaamelia45 |

---

## 📘 Logbook Mingguan

| Week | Task |
| :---: | :---: |
| Week 1 (5 - 11 Mei 2025) | -   Pembuatan Repository |
| Week 2 (12 - 18 Mei 2025) | - |
| Week 3 (19 - 25 Mei 2025) | -   Percobaan rPPG POS dari wajah (branch respiration-and-heartbeat) |
| Week 4 (26 - 31 Mei 2025) | -   Implementasi rPPG POS dari wajah untuk detak jantung <br> -   Implementasi deteksi respirasi menggunakan bahu <br> -   Penggabungan rPPG dan respirasi dalam satu GUI <br> -   Penambahan fitur deteksi manual menggunakan tombol pada GUI untuk komparasi hasil secara real-time <br> -   Implementasi Lucas-Kanade Optical Flow untuk deteksi wajah dan bahu <br> -   Memperbaiki tampilan GUI <br> -   Membuat Laporan Akhir |

## ⚙️ Instalasi

1. Clone repository ini:

    ```bash
    git clone https://github.com/jo0707/dsp-realtime-rppg-respiration.git
    cd dsp-realtime-rppg-respiration
    ```

2. Aktifkan virtual environment (opsional):

    ```bash
    uv venv
    source venv/bin/activate  # Linux/Mac
    .venv\Scripts\activate    # Windows
    ```

3. Instal dependensi:

    ```bash
    pip install -r requirements.txt # pip
    uv pip install -r requirements.txt # uv
    ```

4. Jalankan aplikasi:
    ```bash
    python main.py
    ```

## 📚 Laporan 
Berikut ini adalah laporan program yang kami buat:
[Laporan](https://www.overleaf.com/read/fffpwtswrmvj#411fc5)