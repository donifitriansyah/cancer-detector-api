const express = require('express');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const tf = require('@tensorflow/tfjs-node');
const db = require('./firestore'); // Import Firestore instance

const app = express();
const port = 3000;

// Multer memory storage for handling file uploads in memory
const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  limits: { fileSize: 1000000 },
}).single('image');

app.post('/predict', (req, res, next) => {
  upload(req, res, function (err) {
    if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({
        status: 'fail',
        message: 'Payload content length greater than maximum allowed: 1000000',
      });
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
      result,
      suggestion,
      createdAt: new Date().toISOString(),
    };

    const predictionId = uuidv4();

    // Save the prediction result in Firestore
    await db.collection('predictions').doc(predictionId).set({
      ...predictionData,
      id: predictionId,
    });

    return res.status(200).json({
      status: 'success',
      message: 'Model is predicted successfully',
      data: {
        id: predictionId,
        result,
        suggestion,
        createdAt: predictionData.createdAt,
      },
    });
  } catch (error) {
    console.error('Error in prediction process:', error);
    return res.status(400).json({
      status: 'fail',
      message: 'There was an error during the prediction process.',
    });
  }
});

// Prediction function remains the same
async function predictImage(fileBuffer) {
  try {
    const decodedImage = tf.node.decodeImage(fileBuffer);
    const resizedImage = tf.image.resizeBilinear(decodedImage, [224, 224]);
    const batchedImage = resizedImage.expandDims(0);

    const prediction = model.predict(batchedImage);
    const predictionArray = prediction.arraySync();
    const probability = predictionArray[0][0];

    const result = probability > 0.5 ? 'Cancer' : 'Non-cancer';
    const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Anda sehat!';
    return { result, suggestion };
  } catch (error) {
    console.error('Error during image processing:', error);
    throw new Error('Error during image prediction');
  }
}

// Endpoint to fetch prediction histories
app.get('/predict/histories', async (req, res) => {
  try {
    const snapshot = await db.collection('predictions').get();

    if (snapshot.empty) {
      return res.status(404).json({
        status: 'fail',
        message: 'No prediction history found.',
      });
    }

    const histories = snapshot.docs.map(doc => ({
      id: doc.id,
      history: doc.data(),
    }));

    return res.status(200).json({
      status: 'success',
      data: histories,
    });
  } catch (error) {
    console.error('Error fetching prediction histories:', error);
    return res.status(500).json({
      status: 'fail',
      message: 'Error fetching prediction histories.',
    });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
