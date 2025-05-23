# Our Project

This project is Team H's submission for the Darwin Project, supplementing our report which is titled 'Audio-based detection of COVID-19'. The study explores the effectiveness of different feature sets for cough and speech audio and machine learning algorithms for diagnosing COVID-19, using The COVID-19 Sounds Dataset provided by The University of Cambridge. The study doest just aim to outperform existing research, but to explore a gap in existing research: the combination of the both cough and speech audio rather than just looking at the results of these audio types individually.

# File Structure

```
data/                     # Audio data files (not included in the repository)
src/
  batch.py                # Batch processing and queue management (This is our main function)

  feature_extract.py      # Audio feature extraction helpers
  classify.py             # Classification helpers

  cnn2.py                 # CNN model implementation
  cnn_lstm.py             # CNN-LSTM model implementation
  ffnn.py                 # Feedforward neural network
  legacy.py               # Random Forest, Extra Trees, and Gradient Boosting classifiers
  lr.py                   # Logistic regression model
  resnet50.py             # ResNet50 model implementation
  SVM.py                  # Support Vector Machine model

  metadata/               # Metadata files for audio samples (not included in the repository)
  queues/                 # Predefined queues for common configurations
  results/                # Model results and logs
```

# The Dataset

The project uses the **[COVID-19 Sounds dataset](https://www.covid-19-sounds.org/en/)**, provided by the **University of Cambridge**. The dataset includes audio recordings such as coughs, breathing, and speech samples, along with a variety of metadata regarding these recordings

Due to the large file size (~200GB) and dataset restrictions, the raw audio files and metadata are not included in our git repository.

To use the dataset locally, place the audio files in the `data/` directory and metadata in the `src/metadata` directory. See the **Setup** section below for more information.

# Setup

1. **Clone the repository:**

   ```sh
   git clone <repo-url>
   cd Darwin-TeamH-Submission
   ```

2. **Install dependencies:**

   _Note: to utilise PyTorch's GPU acceleration, you will need to manually install the correct version of with CUDA support for your system before running the command below. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for more information. The program will run on CPU if CUDA is not available, but will be significantly slower when training complex models._

   ```sh
   pip install -r requirements.txt
   ```

3. **Prepare the dataset:**

   - Place the audio data in `data/`

     ```
     data/
       covid19_data_0426/
         0A0ULgZntg/
         0Ab8lJ4AoMT7/
         ...
     src/
     ...
     ```

   - Place the three metadata CSVs in `src/metadata/`

     ```
     data/
     src/
       metadata/
         results_raw_20210426_lan_yamnet_android_noloc.csv
         results_raw_20210426_lan_yamnet_ios_noloc.csv
         results_raw_20210426_lan_yamnet_web_noloc.csv
       queues/
       results/
       ...
     ```

# Usage

All models are run through the `batch.py` script, which handles the queue management and batch processing of models.

```bash
python batch.py
```

## First Run

When running the script for the first time, it will begin to generate the feature CSV files for the data.
This has taken upwards of 6 hours during testing on an Intel i9-14900k, but this is a one-time process. Around 6-7GB of storage
space is required for the features files, so ensure you have enough space before running the script.

The script will automatically check for the existence of the features files in the `src/features/` directory,
so the setup can be cancelled and resumed at any time. The caveat is that the checks are only done for the
presence of the files and not their contents, so if a file is incomplete (i.e. the setup was cancelled during
generation) it will need to be deleted manually and regenerated before it is able to be used for training,
and therefore it's advised to ensure the setup is uninterrupted.

File sizes may vary, but as a rough estimate the expected sizes of the feature files are as follows:

- `features_cough_compare.csv`: ~1.1GB
- `features_cough_compare_speech_compare.csv`: ~2.2GB
- `features_cough_compare_speech_gemaps.csv`: ~1.1GB
- `features_cough_gemaps.csv`: ~11MB
- `features_cough_gemaps_speech_compare.csv`: ~1.1GB
- `features_cough_gemaps_speech_gemaps.csv`: ~21MB
- `features_cough_mfcc.csv`: ~350MB
- `features_speech_compare.csv`: ~1.2GB
- `features_speech_gemaps.csv`: ~12MB

## Configuration and Running Models

Upon successful completion of the feature generation - and instantly on subsequent launches - the script will display a menu system to configure the models to be run, view and sort previous results, and other options. The menu system is
interactive, and will display the available options to choose from - which can be selected by entering the
corresponding number. The initial menu is displayed as follows:

```
**************************************
Welcome to the Team H Batch Processing UI
Current runs in queue: 0
**************************************
1) Add run to queue
2) Remove run from queue
3) View queue
4) Process queue
5) Import queue from file
6) Export queue to file
7) View top results
0) Exit

Select an option:
```

The options are as follows:

1. **Add run to queue**: This option allows you to add a new model run to the queue. You will be prompted to select the audio type, model, and feature set for the run.
2. **Remove run from queue**: This option allows you to remove a run from the queue. You will be prompted to select the run you want to remove.
3. **View queue**: This option displays the current queue of runs, including the audio type, model, and feature set for each run.
4. **Process queue**: This option starts processing the runs in the queue. The script will run each model in the queue and display the results as they are completed.
5. **Import queue from file**: This option allows you to import a queue of runs from a text file, and will append the contents of the file to the current queue. There is a range of predefined queues available in the `src/queues/` directory for access to common configurations.
6. **Export queue to file**: This option allows you to export the current queue to a text file, which can be used for later reference or sharing with others.
7. **View top results**: This option allows you to view and sort the top results from previous runs. You can sort the results by AUC, F1 score, sensitivity, specificity, or the timestamp of the run - all in either ascending or descending order. Future plans include a more advanced filtering and viewing system, but at this time the sorting is done by the selected metric only and is limited to the top 20 results per query.

## Outputs

The script will output the results of each run directly to the console upon completion, or an error message if the run fails. The results displayed will include the AUC, F1 score, sensitivity and specificity, as well as the time taken to complete the run. A sample output is as follows:

```
>>> Running 1/2: Cough - Random Forest - MFCC
✅ Done: AUC=0.5815, F1=0.5737, Sens=0.5568, Spec=0.5909 [took 0:00:02.462689]

>>> Running 2/2: Cough - CNN - MFCC
> Using GPU for training
Error during model execution: name 'fake error' is not defined
❌ Run failed [took 0:00:00]

All runs complete. Press Enter to return...
```

The script will also save the results of each run to the `src/results_log.csv` file, as well as exporting a ROC curve and confusion matrix plot for each run to the `src/results/` directory - named appropriately with the model name, audio type, and feature set used (for example: `roc_resnet_cough_mfcc_[timestamp].png`). The timestamp will be the same as the one in the results log, so the two can be matched up easily.
