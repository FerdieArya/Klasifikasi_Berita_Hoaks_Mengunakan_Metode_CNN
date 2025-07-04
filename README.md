# Klasifikasi_Berita_Hoaks_Mengunakan_Metode_CNN
Berikut adalah kodingan untuk Algoritma Klarifikasi Berita Mengunakan Metode CNN:
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# 1. DATASET
# Minimal 20 data: 10 berita asli (label 0) dan 10 berita hoaks (label 1)
# Anda bisa mengganti atau menambah data ini sesuai kebutuhan.
berita = [
    # Berita Asli (Label 0)
    "Pemerintah resmikan jalan tol baru untuk tingkatkan konektivitas",
    "Presiden Jokowi hadiri KTT G20 bahas perubahan iklim global",
    "Bank Indonesia catat inflasi bulan Juni berada di level 3 persen",
    "Kemenkes galakkan program vaksinasi booster untuk lansia",
    "Timnas sepak bola Indonesia menang 2-0 melawan Malaysia",
    "Harga saham gabungan ditutup menguat di akhir pekan",
    "BMKG prediksi cuaca cerah berawan di sebagian besar wilayah Jawa",
    "Ekspor komoditas pertanian Indonesia alami peningkatan signifikan",
    "Polisi berhasil amankan pelaku kejahatan siber internasional",
    "Pameran teknologi terbaru akan digelar di Jakarta Convention Center",

    # Berita Hoaks (Label 1)
    "Ditemukan pisang raksasa hasil rekayasa genetik dari planet Mars",
    "Air rebusan bawang putih bisa sembuhkan semua jenis kanker dalam 24 jam",
    "Pesan berantai berhadiah mobil mewah hanya dengan membagikan tautan",
    "Ada razia STNK besar-besaran serentak seluruh Indonesia besok",
    "Bill Gates ciptakan virus untuk jual vaksin dan tanam microchip",
    "Beredar foto editan monyet berhidung panjang di hutan Kalimantan",
    "Telur ayam palsu terbuat dari plastik dan lilin dijual di pasar",
    "Bahaya minum es bisa sebabkan pembekuan darah dan radang usus",
    "Kode rahasia di KTP elektronik bisa cairkan dana bantuan pemerintah",
    "Panggilan telepon dari nomor tidak dikenal akan menguras pulsa otomatis"
]

labels = np.array([0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1])

# 2. PREPROCESSING DATA
# Mengubah teks menjadi urutan angka (sequences) agar bisa diproses model
vocab_size = 1000  # Ukuran kosakata
max_length = 20    # Panjang maksimal setiap kalimat
embedding_dim = 16 # Dimensi untuk embedding layer

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(berita)
sequences = tokenizer.texts_to_sequences(berita)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# 3. MEMBAGI DATA TRAINING & TESTING
# Memisahkan data untuk melatih model dan untuk menguji performa model
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# 4. MEMBUAT MODEL CNN (Convolutional Neural Network)
model = Sequential([
    # Mengubah angka menjadi vektor, menangkap makna kata
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    
    # Layer Konvolusi untuk mengekstrak fitur/pola dari teks
    Conv1D(128, 5, activation='relu'),
    
    # Layer Pooling untuk merangkum fitur yang paling penting
    GlobalMaxPooling1D(),
    
    # Layer Fully Connected untuk klasifikasi
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid') # Output: 0 (asli) atau 1 (hoaks)
])

# Mengompilasi model dengan optimizer dan loss function
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Menampilkan ringkasan arsitektur model
print("--- Arsitektur Model CNN ---")
model.summary()
print("\n")


# 5. TRAINING MODEL
print("--- Proses Training Model ---")
num_epochs = 30
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), verbose=2)
print("\n")

# 6. EVALUASI MODEL
print("--- Hasil Evaluasi Model ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Akurasi pada data testing: {accuracy*100:.2f}%")
print(f"Loss pada data testing: {loss:.4f}")

# 7. CONTOH PREDIKSI
print("\n--- Contoh Prediksi Berita Baru ---")
contoh_berita_baru = ["Pemerintah akan bagikan uang 100 juta untuk semua warga"]
new_sequences = tokenizer.texts_to_sequences(contoh_berita_baru)
padded_new = pad_sequences(new_sequences, maxlen=max_length, padding='post', truncating='post')

prediction = model.predict(padded_new)
print(f"Teks Berita: '{contoh_berita_baru[0]}'")
print(f"Skor Prediksi: {prediction[0][0]:.4f}")
if prediction[0][0] > 0.5:
    print("Hasil: Prediksi adalah BERITA HOAKS")
else:
    print("Hasil: Prediksi adalah BERITA ASLI")

```
Hasil Output: 

```
--- Arsitektur Model CNN ---
/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/embedding.py:90: UserWarning: Argument `input_length` is deprecated. Just remove it.
  warnings.warn(
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ embedding (Embedding)           │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv1d (Conv1D)                 │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_max_pooling1d            │ ?                      │             0 │
│ (GlobalMaxPooling1D)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ ?                      │   0 (unbuilt) │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 0 (0.00 B)
 Trainable params: 0 (0.00 B)
 Non-trainable params: 0 (0.00 B)


--- Proses Training Model ---
Epoch 1/30
1/1 - 4s - 4s/step - accuracy: 0.5000 - loss: 0.6927 - val_accuracy: 0.5000 - val_loss: 0.6933
Epoch 2/30
1/1 - 0s - 121ms/step - accuracy: 0.5000 - loss: 0.6872 - val_accuracy: 0.5000 - val_loss: 0.6941
Epoch 3/30
1/1 - 0s - 138ms/step - accuracy: 0.5000 - loss: 0.6817 - val_accuracy: 0.5000 - val_loss: 0.6942
Epoch 4/30
1/1 - 0s - 131ms/step - accuracy: 0.5000 - loss: 0.6760 - val_accuracy: 0.5000 - val_loss: 0.6937
Epoch 5/30
1/1 - 0s - 125ms/step - accuracy: 0.5625 - loss: 0.6706 - val_accuracy: 0.5000 - val_loss: 0.6936
Epoch 6/30
1/1 - 0s - 138ms/step - accuracy: 0.5625 - loss: 0.6650 - val_accuracy: 0.5000 - val_loss: 0.6935
Epoch 7/30
1/1 - 0s - 110ms/step - accuracy: 0.7500 - loss: 0.6592 - val_accuracy: 0.5000 - val_loss: 0.6935
Epoch 8/30
1/1 - 0s - 113ms/step - accuracy: 0.8125 - loss: 0.6532 - val_accuracy: 0.5000 - val_loss: 0.6934
Epoch 9/30
1/1 - 0s - 117ms/step - accuracy: 0.9375 - loss: 0.6471 - val_accuracy: 0.5000 - val_loss: 0.6931
Epoch 10/30
1/1 - 0s - 165ms/step - accuracy: 0.9375 - loss: 0.6407 - val_accuracy: 0.5000 - val_loss: 0.6927
Epoch 11/30
1/1 - 0s - 114ms/step - accuracy: 1.0000 - loss: 0.6340 - val_accuracy: 0.5000 - val_loss: 0.6924
Epoch 12/30
1/1 - 0s - 113ms/step - accuracy: 1.0000 - loss: 0.6271 - val_accuracy: 0.5000 - val_loss: 0.6921
Epoch 13/30
1/1 - 0s - 109ms/step - accuracy: 1.0000 - loss: 0.6196 - val_accuracy: 0.5000 - val_loss: 0.6918
Epoch 14/30
1/1 - 0s - 141ms/step - accuracy: 1.0000 - loss: 0.6116 - val_accuracy: 0.5000 - val_loss: 0.6916
Epoch 15/30
1/1 - 0s - 163ms/step - accuracy: 1.0000 - loss: 0.6031 - val_accuracy: 0.5000 - val_loss: 0.6914
Epoch 16/30
1/1 - 0s - 305ms/step - accuracy: 1.0000 - loss: 0.5941 - val_accuracy: 0.5000 - val_loss: 0.6913
Epoch 17/30
1/1 - 0s - 300ms/step - accuracy: 1.0000 - loss: 0.5845 - val_accuracy: 0.5000 - val_loss: 0.6912
Epoch 18/30
1/1 - 0s - 301ms/step - accuracy: 1.0000 - loss: 0.5742 - val_accuracy: 0.5000 - val_loss: 0.6911
Epoch 19/30
1/1 - 0s - 205ms/step - accuracy: 1.0000 - loss: 0.5633 - val_accuracy: 0.5000 - val_loss: 0.6911
Epoch 20/30
1/1 - 0s - 294ms/step - accuracy: 1.0000 - loss: 0.5518 - val_accuracy: 0.5000 - val_loss: 0.6912
Epoch 21/30
1/1 - 0s - 292ms/step - accuracy: 1.0000 - loss: 0.5397 - val_accuracy: 0.5000 - val_loss: 0.6913
Epoch 22/30
1/1 - 0s - 201ms/step - accuracy: 1.0000 - loss: 0.5272 - val_accuracy: 0.5000 - val_loss: 0.6917
Epoch 23/30
1/1 - 0s - 297ms/step - accuracy: 1.0000 - loss: 0.5140 - val_accuracy: 0.5000 - val_loss: 0.6922
Epoch 24/30
1/1 - 0s - 285ms/step - accuracy: 1.0000 - loss: 0.5004 - val_accuracy: 0.5000 - val_loss: 0.6926
Epoch 25/30
1/1 - 0s - 125ms/step - accuracy: 1.0000 - loss: 0.4863 - val_accuracy: 0.5000 - val_loss: 0.6934
Epoch 26/30
1/1 - 0s - 109ms/step - accuracy: 1.0000 - loss: 0.4719 - val_accuracy: 0.5000 - val_loss: 0.6942
Epoch 27/30
1/1 - 0s - 141ms/step - accuracy: 1.0000 - loss: 0.4569 - val_accuracy: 0.5000 - val_loss: 0.6953
Epoch 28/30
1/1 - 0s - 114ms/step - accuracy: 1.0000 - loss: 0.4416 - val_accuracy: 0.5000 - val_loss: 0.6968
Epoch 29/30
1/1 - 0s - 128ms/step - accuracy: 1.0000 - loss: 0.4260 - val_accuracy: 0.5000 - val_loss: 0.6986
Epoch 30/30
1/1 - 0s - 133ms/step - accuracy: 1.0000 - loss: 0.4102 - val_accuracy: 0.5000 - val_loss: 0.7006


--- Hasil Evaluasi Model ---
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step - accuracy: 0.5000 - loss: 0.7006
Akurasi pada data testing: 50.00%
Loss pada data testing: 0.7006

--- Contoh Prediksi Berita Baru ---
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 127ms/step
Teks Berita: 'Pemerintah akan bagikan uang 100 juta untuk semua warga'
Skor Prediksi: 0.6169
Hasil: Prediksi adalah BERITA HOAKS
```
