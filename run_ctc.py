import os
import time
import random
import torch
import argparse
import collections
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataset import LibriDataset
from torch.utils.data import DataLoader
from torchtext.data import Field
from model import DeepRNN
from torchsummaryX import summary
from torch.nn.utils.rnn import pad_sequence
from speechbrain.decoders.ctc import filter_ctc_output, ctc_greedy_decode
from speechbrain.utils.edit_distance import accumulatable_wer_stats
from torch.utils.tensorboard import SummaryWriter

# CUDA_VISIBLE_DEVICES=1 nohup python -u run_ctc.py --cuda >> nohup_ctc_libri.log 2>&1 &
# tensorboard --logdir=./log --port 2333
# nohup python -u run_ctc.py --test >> nohup_ctc_libri.log 2>&1 &

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--ckpt-dir', type=str, default='./ckpts')
    parser.add_argument('--log-dir', type=str, default='./log')
    parser.add_argument('--feature-size', type=int, default=123)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-epoch', type=int, default=30)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--restore', action='store_true')
    return parser.parse_args()

args = get_args()
random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)

phonedict = Field(lower=True, include_lengths=True, batch_first=True, unk_token=None, pad_token='<blank>')
phonedict.build_vocab([[p] for p in LibriDataset.phonecode])
args.vocab_size = len(phonedict.vocab)
args.blank_index = phonedict.vocab.stoi['<blank>']
print(args.vocab_size, args.blank_index)

def collate(batch):
    features = [item[0] for item in batch]
    feature_len = [len(feature) for feature in features]
    features = pad_sequence(features, batch_first=True)
    targets = [item[-1] for item in batch]
    targets, target_len = phonedict.process(targets)
    return features.float(), targets.int(), torch.LongTensor(feature_len).int(), target_len.int()

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch, dataloader, net, optimizer):
    net.train()
    total_loss = 0.0
    for batch, (features, targets, feature_lens, target_lens) in enumerate(dataloader):
        if args.cuda:
            features, targets, feature_lens, target_lens = features.cuda(), targets.cuda(), feature_lens.cuda(), target_lens.cuda()
        logits, _ = net(features)
        logits = F.log_softmax(logits, dim=-1)
        loss = F.ctc_loss(logits.transpose(0,1).contiguous(), targets, feature_lens, target_lens, net.blank_index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss = loss.cpu().item()
        total_loss += batch_loss * len(features)
        if batch % 100 == 0:
            print(f'[TRAIN Batch {batch:3d}] Loss {batch_loss:8.4f}')
    total_loss = total_loss / len(dataloader.dataset)
    print(f'[TRAIN Epoch {epoch:3d}] Loss {total_loss:8.4f} LR {args.lr:.10f}')
    return total_loss

def validate(epoch, dataloader, net):
    net.eval()
    total_loss = 0.0
    stats = collections.Counter()
    with torch.no_grad():
        for features, targets, feature_lens, target_lens in dataloader:
            if args.cuda:
                features, targets, feature_lens, target_lens = features.cuda(), targets.cuda(), feature_lens.cuda(), target_lens.cuda()
            logits, _ = net(features)
            logits = F.log_softmax(logits, dim=-1)
            loss = F.ctc_loss(logits.transpose(0,1).contiguous(), targets, feature_lens, target_lens, net.blank_index)
            total_loss += loss.cpu().item() * len(features)
            predicts = ctc_greedy_decode(logits, feature_lens/feature_lens.max(), net.blank_index)
            refs = [filter_ctc_output(item, net.blank_index) for item in targets.cpu().tolist()]
            hyps = [filter_ctc_output(item, net.blank_index) for item in predicts]
            stats = accumulatable_wer_stats(refs, hyps, stats)
    total_loss = total_loss / len(dataloader.dataset)
    per = stats['WER']
    print(f'[VALID Epoch {epoch:3d}] Loss {total_loss:8.4f} PER {per:8.4f}')
    return total_loss, per

def test(dataloader, net):
    print('Test started.')
    net.eval()
    stats = collections.Counter()
    with torch.no_grad():
        for features, targets, feature_lens, _ in dataloader:
            logits, _ = net(features)
            logits = F.log_softmax(logits, dim=-1)
            predicts = ctc_greedy_decode(logits, feature_lens/feature_lens.max(), net.blank_index)
            refs = [filter_ctc_output(item, net.blank_index) for item in targets.tolist()]
            hyps = [filter_ctc_output(item, net.blank_index) for item in predicts]
            print(refs)
            print(hyps)
            print('-'*200)
            stats = accumulatable_wer_stats(refs, hyps, stats)
    print('Test PER: ', stats['WER'])
    

def weight_init(model):
    for name, param in model.named_parameters():
        if 'rnn.weight' in name:
            nn.init.orthogonal_(param)
        elif 'rnn.bias' in name:
            nn.init.zeros_(param)
        else:
            nn.init.uniform_(param, -0.1, 0.1)

def main():
    net = DeepRNN(args)
    net.apply(weight_init)
    best_model_path = os.path.join(args.ckpt_dir, f'best_model.pth')
    if not args.test: # train & validate
        print('Load dataset...')
        dataset_train = LibriDataset(os.path.join(args.data_dir, 'train'))
        dataset_val = LibriDataset(os.path.join(args.data_dir, 'dev'))
        dataloader_train = DataLoader(dataset_train, collate_fn=collate, batch_size=args.batch_size, shuffle=True)
        dataloader_val = DataLoader(dataset_val, collate_fn=collate, batch_size=args.batch_size, shuffle=True)
        print('Load finished.')
        
        if args.restore and os.path.exists(best_model_path): net.load_state_dict(torch.load(best_model_path))
        if args.cuda: net = net.cuda()
        summary(net, torch.randn((args.batch_size,128,args.feature_size),dtype=torch.float).cuda())
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        writer = SummaryWriter(args.log_dir)
        best_val_loss = float('inf')
        patience = 0
        
        time_cost = time.time()
        for epoch in range(1, args.n_epoch + 1):
            train_loss = train(epoch, dataloader_train, net, optimizer)
            valid_loss, per = validate(epoch, dataloader_val, net)
            writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, epoch)
            writer.add_scalar('per', per, epoch)
                        
            if valid_loss < best_val_loss:
                patience = 0
                best_val_loss = valid_loss
                torch.save(net.state_dict(), best_model_path)
            else:
                patience += 1
                print(f'[EarlyStopping {patience}/{args.patience}]')
                if patience == args.patience:
                    break
                net.load_state_dict(torch.load(best_model_path))
                if args.cuda: net.cuda()
                args.lr /= 2
                adjust_learning_rate(optimizer, args.lr)
        time_cost = int(time.time() - time_cost)
        print(f'Finished in {time_cost//60:3d}m{time_cost%60:2d}s')
        
    else: # test
        dataset_test = LibriDataset(os.path.join(args.data_dir, 'test'))
        dataloader_test = DataLoader(dataset_test, collate_fn=collate)
        net.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
        test(dataloader_test, net)
    
    
    
if __name__ == '__main__':
    main()