
# Chatbot Project

## Project Overview
This project is designed to build and run a neural network-based chatbot. The chatbot is trained to predict user intents and provide appropriate responses based on input patterns. Additionally, there is a functionality to analyze the mood of a given text using the ChatGPT model via the OpenAI API.

## How It Works
- **Data Preparation**: The chatbot is trained using intent patterns and corresponding responses from the `intents.json` file. The training data is processed using the Natural Language Toolkit (NLTK) to tokenize and lemmatize input patterns.
- **Neural Network**: The chatbot uses a neural network model with multiple dense layers and dropout layers for regularization. The network is trained using the processed data to predict user intents.
- **Mood Analysis**: The `chatgptapi.py` script provides functionality to analyze the mood of a text, classifying it into positive, neutral, or negative categories using the ChatGPT model.

## Setup and Installation
1. Ensure you have Python installed along with necessary libraries like TensorFlow, Keras, NLTK, and OpenAI.
2. Clone or download the project repository.
3. Install required packages using the provided `tensorflow-apple-metal.yml` configuration file.
4. Set up the OpenAI API key in the `chatgptapi.py` script.

## Usage
1. Run the `training.py` script to train the neural network model.
2. Once trained, the chatbot can be run using the `chatbot.py` script. Input your queries and receive responses based on the trained intents.
3. To analyze the mood of a specific text, use the `chatgptapi.py` script.

## Further Development
- Expand the `intents.json` file to include more patterns and responses for a richer chatbot experience.
- Optimize the neural network architecture and training parameters for better accuracy.
- Integrate the chatbot into web or mobile applications for broader usage.

