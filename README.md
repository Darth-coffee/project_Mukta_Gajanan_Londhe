Zebra Finch Song Classification Using CNNs

Problem Description:
Zebra finch song analysis is essential in bioacoustics for studying avian communication.
Spectrograms visually represent audio signals, displaying frequency content over time. When a
zebra finch sings, specific frequency patterns appear as distinguishable shapes in the
spectrogram. This project aims to develop an image classification model to analyze 30-second
spectrogram images and determine whether they contain zebra finch songs. If sufficient labeled
data is available, the model will also identify specific syllables.
This problem is important because birds are often recorded for hours, as we do not know when
they will sing. This software will help distinguish parts of the audio that contain songs from those
that have noise or other loud sounds, making analysis more efficient
Input-Output Statement:
● Input: A 30-second spectrogram image in .png or .jpg format.
● Output: A Boolean (True/False) indicating if a zebra finch song is present (True = Song
present, False = No song). If a song is detected and enough labeled data is available,
the model should also return the individual syllables present.
Data Source:
The dataset will primarily consist of 500 labeled spectrograms provided by the Rajan Lab. If
additional data is required, publicly available zebra finch song recordings from sources like the
Xeno-canto database, Cornell Lab of Ornithology’s Macaulay Library, or the BirdVox dataset will
be used. Audio recordings will be converted into spectrogram images using a pre-existing
matlab script.
Model Architecture Choice:
A Convolutional Neural Network (CNN)-based model, such as ResNet-50 or EfficientNet, will be
used for classification and localization.
Primary Model: ResNet-50 for song classification.
● Well-suited for image classification tasks, including spectrogram analysis.
● Captures hierarchical features effectively, ensuring strong classification accuracy.
● Residual connections mitigate vanishing gradients, making training more stable even
with limited data.
● Handles complex patterns in spectrograms without excessive computational costs.
Alternate Model: EfficientNet-B0 for its computational efficiency and strong generalization
capabilities with small datasets.