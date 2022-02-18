from flask import Flask, render_template, request, redirect
import pickle
import sys

app = Flask(__name__)

def preprocess(file):
    y, sr = librosa.load(file, duration=30)

    #data0 = []
    #defining the Chroma features in the dataframe
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = chroma.mean()#chroma_stft_mean
    chroma_stft_var = chroma.var()#chroma_stft_var
    chroma_stft_std = chroma.std()#chroma_stft_std

    #defining the rms features in the dataframe
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()#rms_mean
    rms_var = rms.var()#rms_var
    rms_std = rms.std()#rms_std

    #defining the spectral centroid features in the dataframe
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_cent_mean = cent.mean()#spec_cent_mean
    spec_cent_var = cent.var()#spec_cent_var
    spec_cent_std = cent.std()#spec_cent_std

    #defining the spectral bandwidth features in the dataframe
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_ban_mean = spec_bw.mean()#spec_ban_mean
    spec_ban_var = spec_bw.var()#spec_ban_var
    spec_ban_std = spec_bw.std()#spec_ban_std

    #defining the spectral constract features in the dataframe
    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    spec_cons_mean = contrast.mean()#spec_cons_mean
    spec_cons_var = contrast.var()#spec_cons_var
    spec_cons_std = contrast.std()#spec_cons_std

    #defining the spectral flatness features in the dataframe
    flatness = librosa.feature.spectral_flatness(y=y)
    spec_flat_mean = flatness.mean()#spec_flat_mean
    spec_flat_var = flatness.var()#spec_flat_var
    spec_flat_std = flatness.std()#spec_flat_std

    #defining the mel frequency cepstrum coefficient features in the dataframe
    mfcc1 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)#n_mfcc=1
    mfcc1_mean = mfcc1.mean()#mfcc1_mean
    mfcc1_var = mfcc1.var()#mfcc1_var

    mfcc2 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)#n_mfcc=2
    mfcc2_mean = mfcc2.mean()#mfcc2_mean
    mfcc2_var = mfcc2.var()#mfcc2_var

    mfcc3 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)#n_mfcc=3
    mfcc3_mean = mfcc3.mean()#mfcc3_mean
    mfcc3_var = mfcc3.var()#mfcc3_var

    mfcc4 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4)#n_mfcc=4
    mfcc4_mean = mfcc4.mean()#mfcc4_mean
    mfcc4_var = mfcc4.var()#mfcc4_var

    mfcc5 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)#n_mfcc=5
    mfcc5_mean = mfcc5.mean()#mfcc5_mean
    mfcc5_var = mfcc5.var()#mfcc5_var
    
    mfcc6 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=6)#n_mfcc=6
    mfcc6_mean = mfcc6.mean()#mfcc6_mean
    mfcc6_var = mfcc6.var()#mfcc6_var

    mfcc7 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=7)#n_mfcc=7
    mfcc7_mean = mfcc7.mean()#mfcc7_mean
    mfcc7_var = mfcc7.var()#mfcc7_var

    mfcc8 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8)#n_mfcc=8
    mfcc8_mean = mfcc8.mean()#mfcc8_mean
    mfcc8_var = mfcc8.var()#mfcc8_var

    mfcc9 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=9)#n_mfcc=9
    mfcc9_mean = mfcc9.mean()#mfcc9_mean
    mfcc9_var = mfcc9.var()#mfcc9_var

    mfcc10 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)#n_mfcc=10
    mfcc10_mean = mfcc10.mean()#mfcc10_mean
    mfcc10_var = mfcc10.var()#mfcc10_var

    mfcc20 = librosa.feature.mfcc(y=y, sr=sr)#n_mfcc=20
    mfcc20_mean = mfcc20.mean()#mfcc20_mean
    mfcc20_var = mfcc20.var()#mfcc20_var

    #defining the spectral roll-off features in the dataframe
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spec_roll_mean = rolloff.mean()#spec_roll_mean
    spec_roll_var = rolloff.var()#spec_roll_var
    spec_roll_std = rolloff.std()#spec_roll_std
    
    #defining the zero crossing rate features in the dataframe
    zero_crosing_rate = librosa.feature.zero_crossing_rate(y)
    zero_cros_mean = zero_crosing_rate.mean()#zero_cros_mean
    zero_cros_var = zero_crosing_rate.var()#zero_cros_var
    zero_cros_std = zero_crosing_rate.std()#zero_cros_std

    #defining the Tempo feature in the dataframe
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    Tempo = tempo.mean()#Tempo

    #defining the Beat feature in the dataframe
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat = librosa.frames_to_time(beats, sr=sr)#changing the beats to timestamps
    Beat_mean = beat.mean()#Beat_var
    Beat_var = beat.var()#Beat_var
    Beat_std = beat.std()#Beat_std

    data0 = [[chroma_stft_mean, chroma_stft_var, chroma_stft_std, rms_mean, rms_var, rms_std, spec_cent_mean, spec_cent_var, spec_cent_std, 
                 spec_ban_mean, spec_ban_var, spec_ban_std, spec_cons_mean, spec_cons_var, spec_cons_std, 
                 spec_flat_mean, spec_flat_var, spec_flat_std, spec_roll_mean, spec_roll_var, spec_roll_std, 
                 zero_cros_mean, zero_cros_var, zero_cros_std, Tempo, Beat_mean, Beat_var, Beat_std, mfcc1_mean, mfcc1_var, mfcc2_mean, mfcc2_var, 
                 mfcc3_mean, mfcc3_var, mfcc4_mean, mfcc4_var, mfcc5_mean, mfcc5_var, mfcc6_mean, mfcc6_var, mfcc7_mean, mfcc7_var, mfcc8_mean, mfcc8_var, 
                 mfcc9_mean, mfcc9_var, mfcc10_mean, mfcc10_var, mfcc20_mean, mfcc20_var]]

    return data0

def predict(data0, sc, model, le_label):
    data0 = sc.transform(data0)
    prediction = model.predict(data0)
    pred = np.argmax(prediction, axis=1)
    pred = int(pred)
    result = le_label.inverse_transform([pred])
    print(result)

# Load classifier model
l_model = keras.models.load_model("music_model.h5")

# Load label encoder and standard scaler model
with open("capstone.pkl", "rb") as f:
    loaded_model = pickle.load(f)

lle_label = loaded_model["label"]  # label encoder
Standard_Scalar = loaded_model["Standard_Scalar"]  # Standard scalar

@app.route('/')
def home():
    """ Render homepage """
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    sample_new = preprocess("John_Legend_-_All_of_Me.mp3")
    data_new = np.array(sample_new)
    final_result = predict(data_new, Standard_Scalar, l_model, lle_label)
    return render_template('result.html', prediction='The music genre is {}'.format(final_result))

if __name__ == "__main__":
    try:
        port = int(sys.argv[1]) # incase a command line port argument is specified use it as default port
    except:
        port = 5200 # if not use this
    print(sys.argv)
    app.run(port=port, debug=True)
    