const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs-node');

const app = express();
const port = 3000;

//classes 
const classes = ['cake',
                'sun',
                'flower',
                'moon',
                'butterfly',
                'car',
                'cup',
                'apple',
                'cat',
                'star']

//argMAX in JS
function argmax(array) {
    if (array.length === 0) {
        throw new Error('Array is empty');
    }

    let maxIndex = 0;
    let maxValue = array[0];

    for (let i = 1; i < array.length; i++) {
        if (array[i] > maxValue) {
            maxValue = array[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

// Load the model from S3
async function loadModelFromS3() {
    const MODEL_URL = 'https://ai-doodle-classifier.s3.ap-south-1.amazonaws.com/model.json'
    return await tf.loadLayersModel(MODEL_URL);
}

let loadedModel;

// Middleware for JSON parsing
app.use(express.json());

// Load the model on server start
loadModelFromS3()
  .then(model => {
    loadedModel = model;
    console.log('Model loaded successfully');
  })
  .catch(error => {
    console.error('Error loading the model:', error);
  });

// Define a route for prediction
app.post('/predict', async (req, res) => {
  try {
    if (!loadedModel) {
      return res.status(500).json({ error: 'Model not loaded' });
    }

    const inputArray = req.body.input;

    const inputTensor = tf.tensor(inputArray, [1, 28, 28, 1]);  // Convert to a tensor with the desired shape
    
    // Perform prediction
    const result = await loadedModel.predict(inputTensor);
    const value = await result.data();
    const idx = argmax(value)

    // Return the prediction result
    res.json({ prediction: classes[idx]});
  } 
  catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ error: 'Prediction failed' });
  }
});

app.get('/', (req,res)=>{
    res.send("Yo! wassup??");
})

// Start the server
app.listen(port, '0.0.0.0',() => {
  console.log(`Server is up & running!`);
});