from emotion_recognition_using_speech.deep_emotion_recognition import DeepEmotionRecognizer


model = DeepEmotionRecognizer(
    emotions=['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps'],
    tess_ravdess=True,
    tess_ravdess_name='ravdess.csv',
    emodb=False,
    emodb_name=None,  # FIXME(TK): assumes dataset is in ./{train_,test_}{emodb_name}
    custom_db=False,
    n_rnn_layers=2,
    n_dense_layers=2,
    rnn_units=128,
    dense_units=128)
model.train()  # this will save to results/*.h5
