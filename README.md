# Bengalese Finch Song Classification Using CNNs

## Problem Description:
Bengalese finch song analysis is essential in bioacoustics for studying avian communication.
Spectrograms visually represent audio signals, displaying frequency content over time. When a 
finch sings, specific frequency patterns appear as distinguishable shapes in the
spectrogram. This project aims to develop an image classification model to analyze 20-second
spectrogram images and determine whether they contain zebra finch songs. If sufficient labeled
data is available, the model can also be trained to identify specific syllables (but currently it 
only identifies the existence of songs).
This problem is important because birds are often recorded for hours, as we do not know when
they will sing. This software will help distinguish parts of the audio that contain songs from those
that have noise or other loud sounds, making analysis more efficient

## Important Notes:
1. 20 second audio files are processed as syllables can be clearly seen when the spectrogram is formed. 
2. The trainnig data is actually spectrograms broken down into syllables. For this the spectrogram was first plotted and then broken into syllable fragments, as the noise fragments obtained this way have a colour scheme consistent with the syllable fragments. This is becasue usually sectrogram plotting scripts asigns colours based on the relative intensity at that frequency.
3. The data folder has positive (syllables) and negative (noise) test images.
4. In proper application of this CNN, I will also include a script that converts audio files into spectrograms, uses a sliding window of syllable dimensions on this spectrogram and feeds each image to the network. The script is present in the repository for your reference.

## Input-Output Statement:
 
● Input: spectrogram image of syllable in .png or .jpg format.\
● Output: A Boolean (True/False) indicating if a bengalese finch song is present (True = Song
present, False = No song).

## Data Source:
The dataset will primarily consist of 5288 syllable images provided by the Rajan Lab. (dimensions: 600x200)
The dataset is available on this drive link https://drive.google.com/drive/folders/1ZuLzsaUbUfqQ1Ovi4HSAmKbIJv4GQTAl?usp=sharings

## Model Architecture Choice:
### Primary Model: Custom 10-Layer CNN for Song Classification
Purpose: A lightweight, deeper convolutional neural network designed specifically for classifying bird song spectrograms.

Strengths:
- 10 convolutional layers allow deeper feature extraction while maintaining computational efficiency.
- Batch normalization after each convolution stabilizes and speeds up training. 
- Max pooling after every two layers progressively reduces spatial dimensions, capturing hierarchical spectrogram patterns effectively.
- Simpler architecture compared to ResNet, making it more interpretable and better suited for small to medium-sized datasets.

Adaptation:
- Final fully connected layers tailored for binary classification (syllable vs. background noise).
- Sigmoid activation at the output ensures direct probability estimates for classification