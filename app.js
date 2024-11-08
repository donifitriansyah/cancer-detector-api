const express = require('express');
const multer = require('multer');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const { v4: uuidv4 } = require('uuid');

const app = express();
const port = 3000;

const storage = multer.memoryStorage();

const upload = multer({
    storage: storage,
    limits: { fileSize: 1000000 },
}).single('image');

app.post('/predict', (req, res, next) => {
    upload(req, res, function (err) {
        if (err instanceof multer.MulterError) {
            if (err.code === 'LIMIT_FILE_SIZE') {
                return res.status(413).json({
                    status: 'fail',
                    message: 'Payload content length greater than maximum allowed: 1000000',
                });
            }
        } else if (err) {
            return res.status(500).json({
                status: 'fail',
                message: 'Something went wrong during the upload.',
            });
        }
        next();
    });
}, async (req, res) => {
    if (!req.file) {
        return res.status(400).json({
            status: 'fail',
            message: 'No image file uploaded.',
        });
    }

    try {
        const { result, suggestion } = await predictImage(req.file.buffer);

        return res.status(200).json({
            status: 'success',
            message: 'Model is predicted successfully',
            data: {
                id: uuidv4(),
                result,
                suggestion,
                createdAt: new Date().toISOString(),
            },
        });
    } catch (error) {
        console.error('Error in prediction process:', error);
        return res.status(400).json({
            status: 'fail',
            message: 'Terjadi kesalahan dalam melakukan prediksi',
        });
    }
});

let model;
async function loadModel() {
    try {
        model = await tf.loadGraphModel('https://storage.googleapis.com/model-asclepius/model.json');
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading the model:', error);
    }
}

loadModel();

async function predictImage(fileBuffer) {
    try {
        const decodedImage = tf.node.decodeImage(fileBuffer);
        const resizedImage = tf.image.resizeBilinear(decodedImage, [224, 224]);
        const batchedImage = resizedImage.expandDims(0);
        const prediction = model.predict(batchedImage);
        const predictionArray = prediction.arraySync();
        const probability = predictionArray[0][0];
        const result = probability > 0.5 ? 'Cancer' : 'Non-cancer';
        const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';
        return {
            result,
            suggestion,
        };
    } catch (error) {
        console.error('Error during image processing:', error);
        throw new Error('Error during image prediction');
    }
}

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
