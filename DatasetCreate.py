import AudioFeatureExtractor as af

def create_df(path):
    dir_list = os.listdir(path)
    count = 0
    df = pd.DataFrame(columns = ['path', 'actor', 'gender','intensity', 'statement', 'repetition', 'emotion'])   
    for i in dir_list:
        filelist = os.listdir(path+'/'+i)
        for f in filelist:
            fname = f.split('.')[0].split('-')
            Path = path+'/'+i+'/'+f
            actor = int(fname[-1])
            emotion = int(fname[2])
            if actor%2==0:
                gender = "Female"
            else: 
                gender = "Male"
            if fname[3] == '01':
                intensity = 0
            else:
                intensity = 1
        
            if fname[4] == '01':
                statement = 0
            else:
                statement = 1
        
            if fname[5] == '01':
                repeat = 0
            else:
                repeat = 1 
            df.loc[count] = [Path, actor, gender, intensity, statement, repeat, emotion]
            count = count+1
    return df        
def parse_audio_files(df):  
    features, labels = np.empty([0,128,259,3]), np.empty(0)  
    for file_path in tqdm(df.path):  
            try:
                X, label = af.padding(str(file_path),df)
                mfccs = af.extract_melspectrogram(X)
                mfccs = af.get_3d_spec(mfccs)
            except Exception as e:
                print("[Error] there was an error in feature extraction. %s" % (e))
                continue
 
            features = np.vstack([features, [mfccs]]) 
            labels = np.append(labels, label)
    return np.array(features), np.array(labels, dtype=np.int)

def save_feat(save_dir,features,labels):
    np.save(save_dir+'/'+'feat.npy', features)
    np.save(save_dir+'/'+'label.npy', labels)        