import os
import glob
import time
import torch
import torchaudio
import argparse
from tqdm import tqdm
from torchtext.data import Field
from model import DeepRNN
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchaudio.compliance.kaldi import fbank

# python main.py --data-dir=./data
# description: read .wav files in data_dir and generate .PHN files under the same path, 
# in which each line denotes start sampling point, end sampling point and phone in the form of 's e p'

class LibriDataset(Dataset):
    
    phonecode = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 
                 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 
                 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 
                 'OW', 'OY', 'P', 'R', 'S', 'SH', 'SIL', 'SP', 'SPN', 'T', 'TH', 
                 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
    
    def __init__(self, dataset_path):
        self.utt_files = sorted(glob.glob(os.path.join(dataset_path, '**/*.wav'), recursive=True))
        self.phn_files = [f.replace('wav', 'PHN') for f in self.utt_files]
        self.features = []
        for utt_file in self.utt_files:
            ftr_file = utt_file.replace('wav', 'pth')
            if os.path.exists(ftr_file):
                feature = torch.load(ftr_file)
            else:
                waveform, _ = torchaudio.load(utt_file)
                feature = fbank(waveform, num_mel_bins=40, use_energy=True)
                feature_delta1 = torchaudio.functional.compute_deltas(feature)
                feature_delta2 = torchaudio.functional.compute_deltas(feature_delta1)
                feature = torch.cat([feature, feature_delta1, feature_delta2], dim=-1)
                torch.save(feature, ftr_file)
            self.features.append(feature)
        print(f'Dataset {dataset_path} # Length: {len(self.features)}')
    
    
    def __len__(self):
        return len(self.features)
    
    
    def __getitem__(self, index):
        return self.features[index], self.phn_files[index]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--ckpt-dir', type=str, default='./ckpts')
    parser.add_argument('--feature-size', type=int, default=123)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--sample-rate', type=int, default=16000)
    return parser.parse_args()

args = get_args()
phonedict = Field(lower=True, include_lengths=True, batch_first=True, unk_token=None, pad_token='<blank>')
phonedict.build_vocab([[p] for p in LibriDataset.phonecode])
args.vocab_size = len(phonedict.vocab)
args.blank_index = phonedict.vocab.stoi['<blank>']

def collate(batch):
    features = [item[0] for item in batch]
    phn_files = [item[1] for item in batch]
    feature_len = [len(feature) for feature in features]
    features = pad_sequence(features, batch_first=True)
    return features.float(), torch.LongTensor(feature_len).int(), phn_files

def greedy_decode(probabilities, seq_lens):
    batch_max_len = probabilities.shape[1]
    batch_outputs = []
    for seq, seq_len in zip(probabilities, seq_lens):
        actual_size = int(torch.round(seq_len * batch_max_len))
        _, predictions = torch.max(seq.narrow(0, 0, actual_size), dim=1)
        out = predictions.tolist()
        batch_outputs.append(out)
    return batch_outputs


def write_phn_file(path, info):
    with open(path, 'w') as f:
        f.writelines([' '.join(line) + '\n' for line in info])

def main():
    print('load dataset...')
    dataset = LibriDataset(args.data_dir)
    dataloader = DataLoader(dataset, collate_fn=collate)
    print('load finised.')
    
    net = DeepRNN(args)
    best_model_path = os.path.join(args.ckpt_dir, f'best_model.pth')
    net.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    net.eval()
    factor = int(0.01 * args.sample_rate)
    total_time = time.time()
    with torch.no_grad():
        for features, feature_lens, phn_files in tqdm(dataloader):
            logits, _ = net(features)
            logits = torch.nn.functional.log_softmax(logits, dim=-1)
            hyps = greedy_decode(logits, feature_lens/feature_lens.max())
            for idx, hyp in enumerate(hyps):
                pos_phn = [(i,v) for i, v in enumerate(hyp) if i == 0 or (v != hyp[i - 1] and v != net.blank_index)]
                # pos_phn = list(filter(lambda x: x[1] != net.blank_index, pos_phn))
                s_e_p = [(str(i * factor), str(pos_phn[j+1][0] * factor), phonedict.vocab.itos[v]) for j, (i, v) in enumerate(pos_phn[:-1])]
                s_e_p.append((str(pos_phn[-1][0] * factor), str(len(hyp) * factor), phonedict.vocab.itos[pos_phn[-1][1]]))
                write_phn_file(phn_files[idx], s_e_p)
    total_time = time.time() - total_time
    print(f'done in {total_time/len(dataloader.dataset)}s/utt')
    
    
if __name__ == '__main__':
    main()