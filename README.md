# Our Project

This project is Team H's submission for the Darwin Project, supplementing our report which is titled 'Audio-based detection of COVID-19'. The study explores the effectiveness of different feature sets for cough and speech audio and machine learning algorithms for diagnosing COVID-19, using The COVID-19 Sounds Dataset provided by The University of Cambridge. The study doest just aim to outperform existing research, but to explore a gap in existing research: the combination of the both cough and speech audio rather than just looking at the results of these audio types individually.

# File Structure

```
src/
  batch.py                # Batch processing and queue management (This is our main function)

  feature_extract.py      # Audio feature extraction helpers
  classify.py             # Classification helpers

  cnn2.py                 # CNN model implementation
  cnn_lstm.py             # CNN-LSTM model implementation
  ffnn.py                 # Feedforward neural network
  lr.py                   # Logistic regression model
  resnet50.py             # ResNet50 model implementation
  SVM.py                  # Support Vector Machine model

  metadata/               # Metadata files for audio samples
  results/                # Model results and logs
```

# The Dataset

The project uses the **[COVID-19 Sounds dataset](https://www.covid-19-sounds.org/en/)**, provided by the **University of Cambridge**. The dataset includes audio recordings such as coughs, breathing, and speech samples, along with a variety of metadata regarding these recordings

Due to the large file size and dataset restirctions, the raw audio files and metadata are not included in our git repository. 

To use the dataset locally, place the audio files in a directory called `../data` and metadata in a folder in a directory called `../src/metadata`


# Setup

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd Darwin-TeamH-Submission
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Prepare the dataset:**
   - Place the audio data in `../data/` 
   - Place the metadata CSVs are in `src/metadata/`

# Usage

- From the `src` directory, run batch which will load up options:
  ```sh
  python batch.py
  ```

