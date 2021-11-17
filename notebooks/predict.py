import glob
import os

import soundfile
import tensorflow as tf
import pandas as pd
import tqdm
import librosa

from emotion_recognition_using_speech.emotion_recognition import EmotionRecognizer


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Load the keras model
    candidate_model_filepaths = glob.glob(os.path.join(os.path.dirname(__file__), '../results/*.h5'))
    assert len(candidate_model_filepaths) == 1
    model_filepath = os.path.abspath(candidate_model_filepaths[0])
    keras_model = tf.keras.models.load_model(model_filepath)
    print(keras_model.summary())

    model = EmotionRecognizer(
        model=keras_model,
        emotions=['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps'],
    )

    # Acquire train/test set indexes:
    df_train_index = pd.read_csv(os.path.join(base_dir, 'data/train_ravdess.csv'))
    df_test_index = pd.read_csv(os.path.join(base_dir, 'data/test_ravdess.csv'))
    df_train_index.rename(columns={'emotion': 'gt'})
    df_test_index.rename(columns={'emotion': 'gt'})
    for emotion in model.emotions:
        df_train_index.insert(column=f'pred_{emotion}', loc=len(df_train_index.columns), value=None)
        df_test_index.insert(column=f'pred_{emotion}', loc=len(df_test_index.columns), value=None)
    print(df_train_index.head())
    print(df_test_index.head())
    print(df_train_index.shape)
    print(df_test_index.shape)

    def predict_row(row: pd.Series) -> dict:
        with soundfile.SoundFile(os.path.join(base_dir, row['path'])) as sound_file:
            audio_data = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
        if len(audio_data.shape) == 1 or audio_data.shape[1] == 1:
            pass
        elif audio_data.shape[1] == 2:
            audio_data = librosa.to_mono(audio_data.transpose())
        else:
            raise ValueError("Unexpected audio data shape: {audio_data.shape}")
        audio_data.reshape([1, 1, -1])
        return model.predict_proba(audio_data=audio_data, sample_rate=sample_rate)

    # Perform predictions:
    for idx, row in tqdm.tqdm(df_train_index.iterrows(), total=df_train_index.shape[0]):
        # TODO(TK): Batch this and do it more efficiently
        prediction = predict_row(row)
        for emotion, prob in prediction.items():
            df_train_index.at[idx, f'pred_{emotion}'] = prob

    for idx, row in tqdm.tqdm(df_test_index.iterrows(), total=df_test_index.shape[0]):
        # TODO(TK): Batch this and do it more efficiently
        prediction = predict_row(row)
        for emotion, prob in prediction.items():
            df_test_index.at[idx, f'pred_{emotion}'] = prob

    # Write dataframes with predictions/targets to csv:
    df_train_index.to_csv(os.path.join(base_dir, 'data/train_ravdess_pred.csv'), index=True, header=True)
    df_test_index.to_csv(os.path.join(base_dir, 'data/test_ravdess_pred.csv'), index=True, header=True)


if __name__ == '__main__':
    main()

