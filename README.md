# GUIDANCE (OpenCV Computer Vision Game)

**GUIDANCE** adalah game interaktif *real-time* berbasis **Pengolahan Citra Video** yang dibangun menggunakan **Python** dan **OpenCV**. Game ini memanfaatkan kamera *webcam* untuk melacak gerakan sarung tangan berwarna (kuning/krem) secara *real-time* sebagai papan pemantul bola (*paddle*).

---

## Fitur Utama

- **100% Pengolaha Citra Video Klasik:** Tanpa menggunakan library *Deep Learning* (seperti MediaPipe, TensorFlow, atau PyTorch). Aman digunakan untuk pemenuhan tugas/proyek yang melarang penggunaan pustaka kecerdasan buatan berbasis *Neural Network*.
- **Pendeteksian Warna HSV Adaptif:** Memanfaatkan ruang warna HSV yang kebal terhadap perubahan intensitas pencahayaan ruangan.
- **Dua Mode Permainan:**
  1. **Random Mode (Wave 1–10):** Mode bertahan hidup dengan target Pokémon acak dari folder `pokemons/` di setiap *wave*, serta pertarungan bos epik melawan **Groudon** di Wave 10.
  2. **Fixed Stage:** Mode level bertema dengan tata letak khusus (contoh: *Rock Farming* dengan gambar batu kustom `orirock.png` dan stage *Theresa*).
- **Pengontrol Kontrol Interaktif (HSV Sliders):** Menu pengaturan interaktif untuk menyesuaikan ambang batas warna (*thresholding*) saat aplikasi sedang berjalan.
- **Navigasi Hands-Free:** Pemilihan menu dilakukan dengan menahan sarung tangan di atas tombol virtual selama 1 detik (atau menggunakan tombol *keyboard*).

---

## Cara Kerja Sistem (System Mechanics)

### 1. Sistem Permainan (Game Mechanics)
* **Kontrol Paddle:** Gerakan sarung tangan pemain dilacak melalui titik pusat massa (*centroid*), kemudian dihitung kecepatannya secara *real-time*. Kecepatan ini digunakan dalam simulasi fisika benturan untuk memantulkan bola.
* **Fisika Tumbukan & Gravitasi:** Menggunakan *sub-stepping* 4 tahap per *frame* agar bola berkecepatan tinggi tidak menembus batas objek yang tipis (*tunneling*).
* **Boss Battle (Wave 10 - Groudon):**
  * Posisi Groudon dikunci di bagian bawah layar ($y = 380 - 430$).
  * Generator rintangan hanya memunculkan objek di area atas layar ($y < 220$), menjaga area bawah tetap bersih agar pemain dapat menyerang Groudon tanpa terhalang.
  * Groudon memicu efek gempa bumi (*earthquake*) yang mengguncang layar dan menjatuhkan batu penghalang dari atas.

### 2. Pendeteksian Warna (Color Detection)
Sistem mengubah *frame* gambar dari BGR ke ruang warna **HSV** (*Hue, Saturation, Value*). Komponen *Hue* mewakili jenis warna dasar, sedangkan *Saturation* dan *Value* mengatur kepekatan serta kecerahan.
* Rentang warna default dikonfigurasi agar sensitif terhadap sarung tangan kuning pastel/krem dalam pencahayaan ruangan:
  * **Hue:** `12 – 35`
  * **Saturation:** `35 – 220`
  * **Value:** `70 – 240`

### 3. Masking & Pembersihan Citra
1. **Gaussian Blur:** Menghaluskan citra input untuk mengurangi *noise*.
2. **Color Thresholding (`cv2.inRange`):** Mengubah *frame* menjadi citra biner (hitam & putih). Piksel di dalam rentang warna sarung tangan bernilai 255 (putih), sisanya 0 (hitam).
3. **Morfologi Citra:**
   * **Opening (`cv2.MORPH_OPEN`):** Menghapus bintik-bintik kecil piksel acak di latar belakang.
   * **Closing (`cv2.MORPH_CLOSE`):** Menutup lubang-lubang kecil di dalam area sarung tangan.
4. **Batas Area Minimum:** Jika area piksel putih kurang dari 800px, sistem mengabaikannya untuk mencegah deteksi palsu.

### 4. Pelacakan Gerakan (Motion Tracking)
Pelacakan posisi dilakukan dengan fungsi momen citra biner (`cv2.moments`):
$$\bar{x} = \frac{M_{10}}{M_{00}}, \quad \bar{y} = \frac{M_{01}}{M_{00}}$$

Kecepatan sarung tangan ($\mathbf{v}_p$) dihitung dari selisih posisi pusat antara *frame* saat ini ($\mathbf{c}$) dan *frame* sebelumnya ($\mathbf{p}_c$) dibagi interval waktu ($\Delta t$):
$$\mathbf{v}_p = \text{clip}\left(0.6 \cdot \mathbf{v}_p + 0.4 \cdot \frac{\mathbf{c} - \mathbf{p}_c}{\Delta t}, -900, 900\right)$$

---

## Struktur Folder Proyek

```text
.
├── bounce.py              # Script utama game
├── hsv_config.json        # Konfigurasi nilai HSV yang tersimpan
├── orirock.png            # Asset gambar batu untuk Rock Farming stage
├── groudon.png            # Asset boss Groudon (opsional/placeholder)
├── pokeball.png           # Asset bola pemantul
└── pokemons/              # Folder gambar target Pokémon (Random Mode)
    ├── Bagon.png
    ├── Beldum.png
    ├── Lotad.png
    ├── Mudkip.png
    ├── Quagsire.png
    ├── Spheal.png
    ├── Torchic.png
    ├── Trapinch.png
    └── Treecko.png
