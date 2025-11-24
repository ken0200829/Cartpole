import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
import os
import wandb
from models import VAM, make_rnn_input_vector, DIR_MAP, lba_log_likelihood

torch.manual_seed(42)
np.random.seed(42)
DATA_PATH = '/Users/utsumikensuke/Research/RNN_test/data/train'
TEST_DATA_PATH = '/Users/utsumikensuke/Research/Cartpole/test_data'

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"デバイス: {device}")

class VAMDataset(Dataset):
    """
    VAM用のデータセット。
    RNN入力に加えて、損失計算用の「現在のRT」と「現在の選択」を返す。
    """
    def __init__(self, data_dir):
        self.sequences = []
        self.labels = [] # Choice (0-3)
        self.rts = []    # RT (seconds)
        self.targets = [] # Correct Answer (0-3)

        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        print(f"{len(csv_files)}個のCSVファイルを検出しました")

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)

            for nth_play_val in sorted(df["nth_play"].unique()):
                df_ep = df[df["nth_play"] == nth_play_val].copy()
                df_ep = df_ep.sort_values("trial").reset_index(drop=True)

                if len(df_ep) == 0: continue

                # 系列データ作成
                seq_data = self._build_sequence_data(df_ep)
                if seq_data is not None:
                    self.sequences.append(seq_data['input'])
                    self.labels.append(seq_data['choice'])
                    self.rts.append(seq_data['rt'])
                    self.targets.append(seq_data['target'])

        print(f"最終的な系列数: {len(self.sequences)}個")

    def _build_sequence_data(self, df_ep):
        seq_inputs = []
        seq_choices = []
        seq_rts = []
        seq_targets = []
        
        prev_reward = 0.0
        prev_action_idx = -1

        for _, row in df_ep.iterrows():
            # --- 入力作成 ---
            flanker_char = str(row["flanker_direction"])
            target_char = str(row["target_direction"])
            
            input_vec = make_rnn_input_vector(flanker_char, target_char, prev_reward, prev_action_idx)
            seq_inputs.append(input_vec)
            
            # --- ターゲット情報 (現在のステップ) ---
            resp_char = str(row["response_direction"])
            resp_idx = DIR_MAP.get(resp_char, -1)
            
            # RT (ms -> sec)
            current_rt_ms = row.get("response_time", 0.0)
            if pd.isna(current_rt_ms): current_rt_ms = 0.0
            current_rt_sec = float(current_rt_ms) / 1000.0
            
            # 正解ラベル
            t_idx = DIR_MAP.get(target_char, 0)
            
            # リストに追加
            # LBAの学習には有効な選択とRTが必要。
            # 選択なし(resp_idx=-1)の場合は、とりあえず0を入れておき、後でマスクするか、
            # ここでは簡易的に0(R)として扱う。
            seq_choices.append(max(0, resp_idx))
            seq_rts.append(current_rt_sec)
            seq_targets.append(t_idx)
            
            # --- 次ステップ用更新 ---
            prev_action_idx = resp_idx
            if resp_idx == t_idx:
                prev_reward = 1.0
            else:
                prev_reward = 0.0

        if len(seq_inputs) == 0:
            return None
            
        return {
            'input': np.array(seq_inputs, dtype=np.float32),
            'choice': np.array(seq_choices, dtype=np.int64),
            'rt': np.array(seq_rts, dtype=np.float32),
            'target': np.array(seq_targets, dtype=np.int64)
        }

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.sequences[idx]).float(),
            torch.from_numpy(self.labels[idx]).long(),
            torch.from_numpy(self.rts[idx]).float(),
            torch.from_numpy(self.targets[idx]).long()
        )

def pad_collate_fn(batch):
    # batch: list of (seq, choice, rt, target)
    seqs, choices, rts, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded_seqs = torch.zeros(len(seqs), max_len, seqs[0].shape[1])
    padded_choices = torch.zeros(len(seqs), max_len, dtype=torch.long)
    padded_rts = torch.zeros(len(seqs), max_len, dtype=torch.float)
    padded_targets = torch.zeros(len(seqs), max_len, dtype=torch.long)
    
    # マスク（パディング部分は0、データ部分は1）
    mask = torch.zeros(len(seqs), max_len, dtype=torch.float)

    for i, (seq, choice, rt, target) in enumerate(zip(seqs, choices, rts, targets)):
        l = len(seq)
        padded_seqs[i, :l] = seq
        padded_choices[i, :l] = choice
        padded_rts[i, :l] = rt
        padded_targets[i, :l] = target
        mask[i, :l] = 1.0

    return padded_seqs, padded_choices, padded_rts, padded_targets, mask

def train_vam(model, train_loader, val_loader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    if config['use_wandb']:
        wandb.watch(model, log="all", log_freq=10)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for sequences, choices, rts, targets, mask in train_loader:
            sequences = sequences.to(device)
            choices = choices.to(device)
            rts = rts.to(device)
            mask = mask.to(device)
            
            # Forward pass
            # drifts: [Batch, Seq, 4]
            drifts, (A, b, t0), _ = model(sequences)
            # print(f'drift rate:{drifts}, A:{A}, b:{b}, t0:{t0}')
            
            # Flatten for loss calculation
            # LBA loss function expects [N, ...]
            drifts_flat = drifts.view(-1, 4)
            choices_flat = choices.view(-1)
            rts_flat = rts.view(-1)
            mask_flat = mask.view(-1)
            
            # --- 修正: パディング部分(RT=0)を安全な値に置換してNaNを防ぐ ---
            # Loss計算時にmaskされるので値自体は学習に影響しないが、
            # 計算グラフ上でNaNが発生すると勾配全体が壊れるため必須。
            safe_rts = rts_flat.clone()
            safe_rts[mask_flat < 0.5] = 1.0  # パディング部分を1.0秒にする
            
            # Calculate Log Likelihood (safe_rtsを使用)
            # log_lik: [N]
            log_lik = lba_log_likelihood(safe_rts, choices_flat, drifts_flat, A, b, t0)
            
            # Apply mask (ignore padding)
            # Loss = -mean(log_likelihood)
            # Only consider valid steps
            loss = -torch.sum(log_lik * mask_flat) / torch.sum(mask_flat)
            
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
        avg_train_loss = train_loss / train_steps
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for sequences, choices, rts, targets, mask in val_loader:
                sequences = sequences.to(device)
                choices = choices.to(device)
                rts = rts.to(device)
                mask = mask.to(device)
                
                drifts, (A, b, t0), _ = model(sequences)
                
                drifts_flat = drifts.view(-1, 4)
                choices_flat = choices.view(-1)
                rts_flat = rts.view(-1)
                mask_flat = mask.view(-1)
                
                # --- 修正: Validationでも同様に安全な値を使用 ---
                safe_rts = rts_flat.clone()
                safe_rts[mask_flat < 0.5] = 1.0
                
                log_lik = lba_log_likelihood(safe_rts, choices_flat, drifts_flat, A, b, t0)
                loss = -torch.sum(log_lik * mask_flat) / torch.sum(mask_flat)
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        
        # LBA Params logging
        A_val, b_val, t0_val = model.get_lba_params()
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  LBA Params: A={A_val.item():.3f}, b={b_val.item():.3f}, t0={t0_val.item():.3f}")
        
        if config['use_wandb']:
            wandb.log({
                'epoch': epoch+1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'param_A': A_val.item(),
                'param_b': b_val.item(),
                'param_t0': t0_val.item()
            })
            
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'model_weights/best_vam_model.pth')
            print("  Model saved.")

def main():
    config = {
        'batch_size': 64,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'use_wandb': True,
    }
    
    if config['use_wandb']:
        wandb.init(project="vam-rnn-lba", config=config)
    
    os.makedirs('model_weights', exist_ok=True)
    
    print("データセットを作成中...")
    dataset = VAMDataset(TEST_DATA_PATH)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=pad_collate_fn)
    
    model = VAM(
        input_size=22,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=4,
        dropout=config['dropout']
    ).to(device)
    
    print("\nVAMの訓練を開始します...")
    train_vam(model, train_loader, val_loader, config)
    
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    main()