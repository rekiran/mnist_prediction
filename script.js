<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MNIST Digit Recognizer</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Draw a Digit</h1>
    <canvas id="canvas" width="280" height="280"></canvas>
    <br>
    <button onclick="clearCanvas()">Clear</button>
    <button onclick="predictDigit()">Predict</button>
    <p id="result"></p> <!-- Ensure this line is here -->

    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script src="script.js"></script>
</body>
</html>

