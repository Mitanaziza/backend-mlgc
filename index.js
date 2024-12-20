const express = require('express');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const admin = require('firebase-admin');
const tf = require('@tensorflow/tfjs-node');
const { Storage } = require('@google-cloud/storage'); // Import Google Cloud Storage
const path = require('path');
const fs = require('fs');

// Setup Firebase Admin SDK tanpa menggunakan key.json
admin.initializeApp({
  credential: admin.credential.applicationDefault() // Menggunakan kredensial default yang sudah diatur di Google Cloud
});

const db = admin.firestore();

// Setup Multer untuk menangani upload gambar dengan ukuran maksimal 1MB
const upload = multer({
  limits: { fileSize: 1000000 }, // Maksimal 1MB
}).single('image'); // Menggunakan field 'image' pada form-data

// Setup Google Cloud Storage
const storage = new Storage();
const bucketName = 'bucket-mlgc-mita'; // Ganti dengan nama bucket Anda
const modelFileName = 'model.json'; // Ganti dengan path model di bucket Anda

// Fungsi untuk mengunduh model dari bucket
async function downloadModelFromBucket() {
  const tempModelPath = 'model.json'; // Path sementara untuk model yang diunduh
  const options = {
    destination: tempModelPath, // Tempat penyimpanan model yang diunduh
  };

  try {
    // Mengunduh model dari bucket
    await storage.bucket(bucketName).file(modelFileName).download(options);
    console.log('Model downloaded successfully from bucket.');
  } catch (error) {
    console.error('Failed to download model from bucket:', error);
    throw error;
  }
}

// Load model dari file lokal
let model;
(async () => {
  try {
    // Unduh model dari bucket sebelum memuatnya
    await downloadModelFromBucket();

    // Model akan dimuat setelah diunduh
    model = await tf.loadGraphModel('file://./model/model.json');
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Failed to load model:', error);
  }
})();

const app = express();

// Endpoint untuk memverifikasi server berjalan
app.get('/', (req, res) => {
  res.send('Server is running');
});

// Endpoint untuk menerima gambar dan melakukan prediksi
app.post('/predict', upload, async (req, res) => {
  try {
    // Cek apakah file ada
    if (!req.file) {
      return res.status(400).json({
        status: "fail",
        message: "No image uploaded"
      });
    }

    // Cek apakah ukuran file lebih dari 1MB
    if (req.file.size > 1000000) {
      return res.status(413).json({
        status: "fail",
        message: "Payload content length greater than maximum allowed: 1000000"
      });
    }

    // Proses gambar
    const buffer = req.file.buffer;
    const image = tf.node.decodeImage(buffer, 3).resizeBilinear([224, 224]).expandDims(0);
    const prediction = model.predict(image);
    const result = prediction.dataSync()[0] > 0.5 ? 'Cancer' : 'Non-cancer';

    // ID unik untuk hasil prediksi
    const id = uuidv4();
    const createdAt = new Date().toISOString();
    const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';

    // Simpan hasil prediksi ke Firestore
    await db.collection('predictions').doc(id).set({
      id,
      result,
      suggestion,
      createdAt
    });

    // Kirimkan response dengan hasil prediksi
    res.status(201).json({
      status: 'success',
      message: 'Model is predicted successfully',
      data: { id, result, suggestion, createdAt }
    });

  } catch (error) {
    console.error(error);
    res.status(400).json({ status: 'fail', message: 'Terjadi kesalahan dalam melakukan prediksi' });
  }
});

// Menjalankan server di port 8080
const PORT = process.env.PORT || 8080;
app.listen(PORT, '0.0.0.0', async () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
