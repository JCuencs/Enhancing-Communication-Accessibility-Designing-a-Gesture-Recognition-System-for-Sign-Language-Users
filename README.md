# Enhancing Communication Accessibility: Designing a Gesture Recognition System for Sign Language Users

## Researchers

- John Joseph M. Cuenco
- Krisnel A. Cabato

## Overview

This project aims to develop a gesture recognition system to facilitate communication for sign language users. By leveraging machine learning techniques, the system interprets sign language gestures and translates them into text, enhancing accessibility for individuals with hearing impairments adn speech impediments.

## Features

- **Gesture Data Collection**: Scripts to collect and store gesture data.
- **Data Preprocessing**: Tools to preprocess collected data for model training.
- **Model Training**: Implementation of machine learning models for gesture recognition.
- **Accuracy Evaluation**: Evaluation metrics to assess model performance.
- **Real-time Recognition**: Application to recognize gestures in real-time.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/JCuencs/Enhancing-Communication-Accessibility-Designing-a-Gesture-Recognition-System-for-Sign-Language-Users.git
   cd Enhancing-Communication-Accessibility-Designing-a-Gesture-Recognition-System-for-Sign-Language-Users
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Collection**:

   - Run `01-data_collection.py` to collect gesture data. Ensure your webcam is connected and properly configured.

2. **Data Preprocessing**:

   - Execute `02-data_preprocessing.py` to preprocess the collected data.

3. **Model Training**:

   - Train the model by running `03-model_training.py`. Adjust hyperparameters as needed.

4. **Accuracy Evaluation**:

   - Evaluate the model's performance using `04-accuracy_evaluation.py`.

5. **Real-time Recognition**:

   - Start the real-time gesture recognition application with `05-main.py`.

## Project Structure

- `MP_Data/`: Directory containing collected gesture data.
- `trained_model/`: Directory where trained models are saved.
- `01-data_collection.py`: Script for collecting gesture data.
- `02-data_preprocessing.py`: Script for preprocessing data.
- `03-model_training.py`: Script for training the gesture recognition model.
- `04-accuracy_evaluation.py`: Script for evaluating model accuracy.
- `05-main.py`: Main application script for real-time gesture recognition.
- `requirements.txt`: List of required Python packages.

