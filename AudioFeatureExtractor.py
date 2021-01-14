class AudioFeatureExtractor:
    
    def __init__(
            self,
            sampling_rate=22050,
            frame_length=2048,
            hop_ratio=4,audio_duration=3):
        
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.hop_ratio = hop_ratio
        self.hop_length = int(self.frame_length / self.hop_ratio)
        self.audio_duration= audio_duration
        self.audio_length = self.audio_duration*self.sampling_rate

    def padding(self,file_path,df):
        data, _ = librosa.load(file_path, sr=self.sampling_rate)
        input_length = self.audio_length
        data = librosa.effects.preemphasis(data)
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        return data,int(df[df.path==file_path].emotion)-1

    def extract_melspectrogram(self, audio=None, S=None):
        
        if S is not None:
            S = np.abs(librosa.stft(
    audio, n_fft=self.frame_length, hop_length=self.hop_length, win_length=self.frame_length, window='hann')) ** 2
        melspectrogram = librosa.feature.melspectrogram(
            y=audio,
            S=S,
            sr=self.sampling_rate,
            n_fft=self.frame_length,
            hop_length=self.hop_length)
        melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
        return melspectrogram

    def get_3d_spec(self,Sxx_in):
        
        base_mean, delta_mean, delta2_mean = (0, 0, 0)
        base_std, delta_std, delta2_std = (1, 1, 1)
        h, w = Sxx_in.shape
        right1 = np.concatenate([Sxx_in[:, 0].reshape((h, -1)), Sxx_in], axis=1)[:, :-1]
        delta = (Sxx_in - right1)[:, 1:]
        delta_pad = delta[:, 0].reshape((h, -1))
        delta = np.concatenate([delta_pad, delta], axis=1)
        right2 = np.concatenate([delta[:, 0].reshape((h, -1)), delta], axis=1)[:, :-1]
        delta2 = (delta - right2)[:, 1:]
        delta2_pad = delta2[:, 0].reshape((h, -1))
        delta2 = np.concatenate([delta2_pad, delta2], axis=1)
        base = (Sxx_in - base_mean) / base_std
        delta = (delta - delta_mean) / delta_std
        delta2 = (delta2 - delta2_mean) / delta2_std
        stacked = [arr.reshape((h, w, 1)) for arr in (base, delta, delta2)]
        return np.concatenate(stacked, axis=2)
   