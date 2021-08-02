import os
import glob
import torch
import torchaudio
from tqdm import tqdm
from torch.utils.data import Dataset
from torchaudio.compliance.kaldi import fbank
        
class LibriDataset(Dataset):
    
    phonecode = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 
                 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 
                 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 
                 'OW', 'OY', 'P', 'R', 'S', 'SH', 'SIL', 'SP', 'SPN', 'T', 'TH', 
                 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
    
    def __init__(self, dataset_path):
        self.utt_files = sorted(glob.glob(os.path.join(dataset_path, '**/*.wav'), recursive=True))
        self.features, self.phonetexts = [], []
        for utt_file in self.utt_files:
            ftr_file = utt_file.replace('wav', 'pth')
            if os.path.exists(ftr_file):
                feature = torch.load(ftr_file)
            else:
                waveform, self.sample_rate = torchaudio.load(utt_file)
                feature = fbank(waveform, num_mel_bins=40, use_energy=True)
                feature_delta1 = torchaudio.functional.compute_deltas(feature)
                feature_delta2 = torchaudio.functional.compute_deltas(feature_delta1)
                feature = torch.cat([feature, feature_delta1, feature_delta2], dim=-1)
                torch.save(feature, ftr_file)
            self.features.append(feature)
            with open(utt_file.replace('wav', 'phn'), 'r') as f:
                phones = [line.strip('\n').split(' ')[-1] for line in f.readlines()]
                self.phonetexts.append(phones)
        print(f'Dataset {dataset_path} # Length: {len(self.features)}')
    
    
    def __len__(self):
        return len(self.features)
    
    
    def __getitem__(self, index):
        return self.features[index], self.phonetexts[index]
