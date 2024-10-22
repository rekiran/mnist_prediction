const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;
let model;

async function loadModel() {
    model = await tf.loadLayersModel('tfjs_model/model.json');
    document.getElementById('predict-button').disabled = false; // Enable the button once the model is loaded
}

loadModel();

canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mousemove', draw);

function draw(event) {
    if (!drawing) return;
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
    ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('result').innerText = '';
}

async function predictDigit() {
    if (!model) {
        alert('Model not loaded yet. Please wait.');
        return;
    }
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const tensor = tf.browser.fromPixels(imageData, 1)
        .resizeNearestNeighbor([28, 28])
        .mean(2)
        .toFloat()
        .expandDims(0)
        .expandDims(-1)
        .div(255.0);
    const prediction = await model.predict(tensor).argMax(-1).data();
    document.getElementById('result').innerText = `Prediction: ${prediction[0]}`;
}
