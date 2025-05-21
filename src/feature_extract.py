import csv
import pandas as pd
import numpy as np
import librosa
import os
from joblib import Parallel, delayed
import opensmile
class FeatureExtractor:
    def __init__(self, use_subset: bool, output_path='features_CoughCOSpeechGE.csv', batch_size=1000, verbose: bool = False, feature_type = "COUGH_MFCC"):
        
        self.verbose = verbose
        self.use_subset = use_subset
        self.output_path = output_path
        self.batch_size = batch_size
        self.sample_rate = 44100
        self.n_mfcc = 20
        self.target_time_steps = 100  
        self.feature_type = feature_type # COUGH_MFCC OR SPEECH_COMPARE


        # Initialize CSV
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        with open(self.output_path, 'w') as f:
            pass

    def extract_audio_files(self, web=False, ios=True, android=True):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_dir = os.path.join(base_dir, 'metadata')
        paths = []

        if web:
            paths.append(os.path.join(metadata_dir, 'results_raw_20210426_lan_yamnet_web_noloc.csv'))
        if ios:
            paths.append(os.path.join(metadata_dir, 'results_raw_20210426_lan_yamnet_ios_noloc.csv'))
        if android:
            paths.append(os.path.join(metadata_dir, 'results_raw_20210426_lan_yamnet_android_noloc.csv'))

        all_rows = []
        for path in paths:
            rows = self.process_file(path)  # Ensure process_file returns a list
            if rows:  # Only extend if we get valid data
                all_rows.extend(rows)

        if all_rows:
            print(f"Processed {len(all_rows)} rows in total.")
        else:
            print("No data processed.")

    def process_file(self, file_path):
        data_batch = []

        with open(file_path, mode='r') as file:
            reader = csv.reader(file, delimiter=';')
            header = next(reader)

            for row in reader:
                result = self.process_row(row)
                if result:
                    data_batch.append(result)

                # Save in batches
                if len(data_batch) >= self.batch_size:
                    self.save_batch(data_batch)
                    data_batch = []

        if data_batch:
            self.save_batch(data_batch)

        return data_batch  # Return the processed batch of rows

    def process_row(self, row):
        # Map COVID Status
        status = row[10].strip().lower()
        positive_set = {'positivelast14', 'last14'}
        negative_set = {'negativenever', 'negativeover14'}
        
        if status in positive_set:
            label = 1
        elif status in negative_set:
            label = 0
        else:
            return None

        folder_id, folder_name = row[1], row[8]
        cough_file = row[13]
        speech_file = row[12]
        base_path = os.path.join("../data/covid19_data_0426", folder_id, folder_name)
    
        if self.feature_type == "COUGHGEspeechGE":
            cough_path = os.path.join(base_path, cough_file).replace("\\", "/")
            speech_path = os.path.join(base_path, speech_file).replace("\\", "/")

            if row[16] != 'c' or row[15] != 'v':
                return None

            if not cough_path.endswith(".wav"):
                cough_path = os.path.splitext(cough_path)[0] + ".wav"
            if not speech_path.endswith(".wav"):
                speech_path = os.path.splitext(speech_path)[0] + ".wav"

            cough_features = self.extract_features(cough_path, mode="GeMAPS")
            speech_features = self.extract_features(speech_path, mode="GeMAPS")
            if cough_features is None or speech_features is None:
                return None

            combined_features = np.concatenate((cough_features, speech_features))
            return [label] + combined_features.tolist()
        
        if self.feature_type == "COUGHGEspeechCO":
            cough_path = os.path.join(base_path, cough_file).replace("\\", "/")
            speech_path = os.path.join(base_path, speech_file).replace("\\", "/")

            if row[16] != 'c' or row[15] != 'v':
                return None

            if not cough_path.endswith(".wav"):
                cough_path = os.path.splitext(cough_path)[0] + ".wav"
            if not speech_path.endswith(".wav"):
                speech_path = os.path.splitext(speech_path)[0] + ".wav"

            cough_features = self.extract_features(cough_path, mode="GeMAPS")
            speech_features = self.extract_features(speech_path, mode="ComParE")
            if cough_features is None or speech_features is None:
                return None

            combined_features = np.concatenate((cough_features, speech_features))
            return [label] + combined_features.tolist()
        
        if self.feature_type == "COUGHCOspeechCO":
            cough_path = os.path.join(base_path, cough_file).replace("\\", "/")
            speech_path = os.path.join(base_path, speech_file).replace("\\", "/")

            if row[16] != 'c' or row[15] != 'v':
                return None

            if not cough_path.endswith(".wav"):
                cough_path = os.path.splitext(cough_path)[0] + ".wav"
            if not speech_path.endswith(".wav"):
                speech_path = os.path.splitext(speech_path)[0] + ".wav"

            cough_features = self.extract_features(cough_path, mode="ComParE")
            speech_features = self.extract_features(speech_path, mode="ComParE")
            if cough_features is None or speech_features is None:
                return None

            combined_features = np.concatenate((cough_features, speech_features))
            return [label] + combined_features.tolist()
        
        if self.feature_type == "COUGHCOspeechGE":
            cough_path = os.path.join(base_path, cough_file).replace("\\", "/")
            speech_path = os.path.join(base_path, speech_file).replace("\\", "/")

            if row[16] != 'c' or row[15] != 'v':
                return None

            if not cough_path.endswith(".wav"):
                cough_path = os.path.splitext(cough_path)[0] + ".wav"
            if not speech_path.endswith(".wav"):
                speech_path = os.path.splitext(speech_path)[0] + ".wav"

            cough_features = self.extract_features(cough_path, mode="ComParE")
            speech_features = self.extract_features(speech_path, mode="GeMAPS")
            if cough_features is None or speech_features is None:
                return None

            combined_features = np.concatenate((cough_features, speech_features))
            return [label] + combined_features.tolist()
      
        if self.feature_type == "COUGH_MFCC":
            path = os.path.join("../data/covid19_data_0426", folder_id, folder_name, cough_file).replace("\\", "/")
            if row[16] != 'c':
                return None
        else: 
            path = os.path.join("../data/covid19_data_0426", folder_id, folder_name, speech_file).replace("\\", "/")
            if row[15] != 'v':
                return None
        if not path.endswith(".wav"):
            path = os.path.splitext(path)[0] + ".wav"
     
        features = self.extract_features(path)
        if features is None:
            return None

        return [label] + features.tolist()


        

    def extract_features(self, audio_path, mode=None):
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        y = librosa.util.normalize(self.trim_silence(y))

        if (self.feature_type == "COUGH_MFCC") or (mode == "MFCC"):
            features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            time_steps = features.shape[1]
            if time_steps < self.target_time_steps:
                pad_width = self.target_time_steps - time_steps
                features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                features = features[:, :self.target_time_steps]

            if features.shape[1] != self.target_time_steps:
                return None
            return features.flatten()

        if (self.feature_type == "SPEECH_COMPARE") or (mode == "ComParE"):
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals
            )
            features = smile.process_signal(y, sr)
            return features.to_numpy().flatten()

        if (self.feature_type == "SPEECH_GeMAPS") or (mode == "GeMAPS"):
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.GeMAPSv01b,
                feature_level=opensmile.FeatureLevel.Functionals
            )
            features = smile.process_signal(y, sr)
            return features.to_numpy().flatten()

        return None



    def trim_silence(self, y):
        return librosa.effects.trim(y, top_db=20)[0]

    def save_batch(self, batch):
        df = pd.DataFrame(batch)
        header = not os.path.exists(self.output_path) or os.path.getsize(self.output_path) == 0
        df.to_csv(self.output_path, mode='a', index=False, header=header)
        if self.verbose:
            print(f"Saved {len(batch)} samples to CSV.")

if __name__ == "__main__":
    extractor = FeatureExtractor(use_subset=False, verbose=True, feature_type="COUGHCOspeechGE") #COUGH_MFCC, SPEECH_COMPARE, SPEECH_GeMAPS COUGHGEspeechGE COUGHGEspeechCO COUGHCOspeechCO COUGHCOspeechGE
    extractor.extract_audio_files(web=False, ios=True, android=True)