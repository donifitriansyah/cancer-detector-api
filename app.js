const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const { v4: uuidv4 } = require('uuid');
const { Firestore } = require('@google-cloud/firestore');

const app = express();
const port = 8080;

// Initialize Firestore
const firestore = new Firestore();

// Set up multer for image upload
const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: { fileSize: 1000000 },
}).single('image');

// Load TensorFlow model and ensure it's ready
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

// Middleware to check if model is loaded before handling requests
app.use((req, res, next) => {
    if (!model) {
        return res.status(503).json({
            status: 'fail',
            message: 'Model is not loaded yet. Please try again later.',
        });
    }
    next();
});

// Define prediction route
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

        const predictionData = {
            id: uuidv4(),
            result,
            suggestion,
            createdAt: new Date().toISOString(),
        };

        // Save prediction data to Firestore
        await firestore.collection('predictions').doc(predictionData.id).set(predictionData);

        return res.status(201).json({
            status: 'success',
            message: 'Model is predicted successfully',
            data: predictionData,
        });
    } catch (error) {
        console.error('Error in prediction process:', error);
        return res.status(400).json({
            status: 'fail',
            message: 'Terjadi kesalahan dalam melakukan prediksi',
        });
    }
});

// Define new endpoint to get prediction history
app.get('/predict/histories', async (req, res) => {
    try {
        const predictionsSnapshot = await firestore.collection('predictions').get();
        const histories = predictionsSnapshot.docs.map(doc => ({
            id: doc.id,
            history: doc.data(),
        }));

        return res.status(200).json({
            status: 'success',
            data: histories,
        });
    } catch (error) {
        console.error('Error fetching prediction history:', error);
        return res.status(500).json({
            status: 'fail',
            message: 'Error fetching prediction history',
        });
    }
});

// Function to predict image class
async function predictImage(fileBuffer) {
    try {
        if (!model) {
            throw new Error('Model is not loaded');
        }
        const decodedImage = tf.node.decodeImage(fileBuffer);
        const resizedImage = tf.image.resizeBilinear(decodedImage, [224, 224]);
        const batchedImage = resizedImage.expandDims(0);
        const prediction = model.predict(batchedImage);
        const predictionArray = prediction.arraySync();
        const probability = predictionArray[0][0];
        const result = probability > 0.5 ? 'Cancer' : 'Non-cancer';
        const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Anda sehat!';
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
