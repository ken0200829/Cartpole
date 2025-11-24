import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob
import os
import wandb
# models.py と同じディレクトリにあると仮定
from models import FlankerRNN, make_rnn_input_vector, DIR_MAP, get_stimulus_index

torch.manual_seed(42)
np.random.seed(42)

# パスは環境に合わせて変更してください
DATA_PATH = '/Users/utsumikensuke/Research/RNN_test/data/train'

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"使用デバイス: CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"使用デバイス: Mac GPU (MPS)")
else:
    device = torch.device('cpu')
    print(f"使用デバイス: CPU")
print(f"デバイス: {device}")

class FlankerDataset(Dataset):
    """
    ドキュメントに基づき、人間の応答分布(Soft Target)を計算して返すように拡張されたDataset
    """
    def __init__(self, data_dir, split="train"):
        self.sequences = []
        self.labels = []
        self.soft_targets = [] # 追加: 人間の応答分布

        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        print(f"{len(csv_files)}個のCSVファイルを検出しました")

        # --- Step 1: 人間の応答分布(P_human)の集計 ---
        # 刺激インデックス(0-15)ごとに、各行動(0-3)が選ばれた回数をカウント
        # shape: (16, 4) -> (Stimulus Pattern, Response Class)
        response_counts = np.zeros((16, 4), dtype=np.float32)
        
        # 全ファイルを一度読み込んで分布を作成（メモリ効率のためデータ読み込みと同時に行うのが理想ですが、
        # ここでは集計ロジックを明確にするため、簡略化して全データを一度パスします）
        print("人間の応答分布を集計中...")
        all_dfs = []
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
            
            for _, row in df.iterrows():
                f_char = str(row["flanker_direction"])
                t_char = str(row["target_direction"])
                r_char = str(row["response_direction"])
                
                stim_idx = get_stimulus_index(f_char, t_char)
                resp_idx = DIR_MAP.get(r_char, -1)
                
                if resp_idx != -1:
                    response_counts[stim_idx, resp_idx] += 1.0

        # カウントを確率分布に変換 (Soft Target) [cite: 37]
        # ゼロ除算回避のためepsilonを加える
        row_sums = response_counts.sum(axis=1, keepdims=True) + 1e-9
        self.human_distribution_map = response_counts / row_sums
        
        print("応答分布の集計完了。")

        # --- Step 2: 系列データの作成 ---
        total_sequences = 0
        for df in all_dfs:
            # nth_play ごとに分割
            for nth_play_val in sorted(df["nth_play"].unique()):
                df_ep = df[df["nth_play"] == nth_play_val].copy()
                df_ep = df_ep.sort_values("trial").reset_index(drop=True)

                if len(df_ep) == 0:
                    continue

                # 入力を作成
                sequence = self._build_sequence(df_ep)
                label = self._build_label(df_ep)
                # ソフトターゲットを作成
                soft_target = self._build_soft_target(df_ep)

                if sequence is not None and label is not None:
                    self.sequences.append(sequence)
                    self.labels.append(label)
                    self.soft_targets.append(soft_target)
                    total_sequences += 1

        print(f"最終的な系列数: {total_sequences}個")

    def _build_sequence(self, df_ep):
        seq = []
        prev_reward = 0.0
        prev_action_idx = -1

        for _, row in df_ep.iterrows():
            flanker_char = str(row["flanker_direction"])
            target_char = str(row["target_direction"])
            
            input_vec = make_rnn_input_vector(flanker_char, target_char, prev_reward, prev_action_idx)
            seq.append(input_vec)
            
            resp_char = str(row["response_direction"])
            resp_idx = DIR_MAP.get(resp_char, -1)
            prev_action_idx = resp_idx
            
            t_idx = DIR_MAP.get(target_char, 0)
            if resp_idx == t_idx:
                prev_reward = 1.0
            else:
                prev_reward = 0.0

        if len(seq) == 0: return None
        return np.array(seq, dtype=np.float32)

    def _build_label(self, df_ep):
        labels = []
        for _, row in df_ep.iterrows():
            response_idx = DIR_MAP.get(str(row["response_direction"]), -1)
            if response_idx != -1:
                labels.append(response_idx)
            else:
                labels.append(-1)
        return np.array(labels, dtype=np.int64)

    def _build_soft_target(self, df_ep):
        """各ステップの刺激に対応する人間の応答分布を取得"""
        soft_targets = []
        for _, row in df_ep.iterrows():
            f_char = str(row["flanker_direction"])
            t_char = str(row["target_direction"])
            stim_idx = get_stimulus_index(f_char, t_char)
            
            # 事前に集計した分布を取得 [cite: 37]
            dist = self.human_distribution_map[stim_idx]
            soft_targets.append(dist)
            
        return np.array(soft_targets, dtype=np.float32) # (SeqLen, 4)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx]).float()
        label = torch.from_numpy(self.labels[idx]).long()
        soft_target = torch.from_numpy(self.soft_targets[idx]).float()
        
        # Target (正解) インデックスの復元
        stim_idx = torch.argmax(seq[:, :16], dim=1)
        target_indices = stim_idx % 4 
        
        return seq, label, target_indices, soft_target

def load_data(data_dir, batch_size=32, split_ratio=0.8):
    dataset = FlankerDataset(data_dir, split="all")
    total_len = len(dataset)
    train_len = int(total_len * split_ratio)
    val_len = total_len - train_len

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    return train_loader, val_loader

def pad_collate_fn(batch):
    """
    可変長の系列をパディングしてバッチ化する (Soft Target対応版)
    """
    seqs, labels, targets, soft_targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded_seqs = torch.zeros(len(seqs), max_len, seqs[0].shape[1])
    padded_labels = torch.full((len(seqs), max_len), -1, dtype=torch.long)
    padded_targets = torch.full((len(seqs), max_len), -1, dtype=torch.long)
    # Soft Target のパディング (値は0で埋めるが、maskで除外されるので影響なし)
    padded_soft_targets = torch.zeros(len(seqs), max_len, 4, dtype=torch.float32)

    for i, (seq, label, target, s_target) in enumerate(zip(seqs, labels, targets, soft_targets)):
        seq_len = len(seq)
        padded_seqs[i, :seq_len] = seq
        padded_labels[i, :seq_len] = label 
        padded_targets[i, :seq_len] = target
        padded_soft_targets[i, :seq_len] = s_target

    return padded_seqs, padded_labels, padded_targets, padded_soft_targets

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, patience=10, use_wandb=True, temperature=1.0, kl_weight=1.0):
    """
    KLダイバージェンス損失を導入した学習関数
    Args:
        kl_weight (float): lambdaパラメータ。KL損失の重み [cite: 27]。
    """
    # 1. ハードターゲット用損失 (CrossEntropy)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-1)
    
    # 2. KLダイバージェンス損失 (Forward KL: P_human || P_model) [cite: 30]
    # batchmean指定により、バッチサイズで正規化された和を計算
    criterion_kl = nn.KLDivLoss(reduction='none') 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    if use_wandb:
        wandb.watch(model, criterion_ce, log="all", log_freq=10)
    
    nogo_class_index = 0
    best_train_loss = float('inf')
    epochs_no_improve = 0
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_task_acc': [],
        'val_loss': [], 'val_acc': [], 'val_task_acc': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_kl_loss = 0.0 # ログ用
        train_ce_loss = 0.0 # ログ用
        train_correct = 0
        train_task_correct = 0
        train_total = 0
        
        for sequences, labels, targets, soft_targets in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device) 
            targets = targets.to(device)
            soft_targets = soft_targets.to(device) # (Batch, Seq, 4)
            
            outputs_seq, _, _ = model(sequences) # (Batch, Seq, 4)
            
            # Flatten
            outputs_flat = outputs_seq.reshape(-1, 4)
            labels_flat = labels.reshape(-1)
            targets_flat = targets.reshape(-1)
            soft_targets_flat = soft_targets.reshape(-1, 4)
            
            # マスク作成 (パディング除外)
            mask = (labels_flat != -1)
            
            # --- Loss計算  ---
            # 1. Cross Entropy Loss (Task Optimization)
            loss_ce = criterion_ce(outputs_flat, labels_flat)
            
            # 2. KL Divergence Loss (Behavioral Fidelity)
            # PytorchのKLDivLossは inputにlog-prob, targetにprobを期待する
            # log(P_model)
            log_probs = F.log_softmax(outputs_flat / temperature, dim=1)
            
            # KL(P_human || P_model) = sum(P_human * (log(P_human) - log(P_model)))
            # PyTorch KLDivLoss(input, target) = target * (log(target) - input)
            # ここではソフトターゲット(P_human)をターゲットとする
            kl_elementwise = criterion_kl(log_probs, soft_targets_flat)
            
            # パディング部分をマスクして平均を取る
            loss_kl = (kl_elementwise.sum(dim=1) * mask).sum() / mask.sum()
            
            # ハイブリッド損失: L_total = L_ce + lambda * L_kl
            loss = loss_ce + kl_weight * loss_kl
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_ce_loss += loss_ce.item()
            train_kl_loss += loss_kl.item()
            
            # 精度計算
            probs = torch.softmax(outputs_flat / temperature, dim=-1)
            predicted = torch.multinomial(probs, num_samples=1).squeeze()
            
            valid_labels = labels_flat[mask]
            valid_predicted = predicted[mask]
            valid_targets = targets_flat[mask]
            
            train_total += valid_labels.size(0) 
            train_correct += (valid_predicted == valid_labels).sum().item()
            train_task_correct += (valid_predicted == valid_targets).sum().item()
            
        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_task_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels, targets, soft_targets in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                targets = targets.to(device)
                soft_targets = soft_targets.to(device)

                outputs_seq, _, _ = model(sequences)
                outputs_flat = outputs_seq.reshape(-1, 4)
                labels_flat = labels.reshape(-1)
                targets_flat = targets.reshape(-1)
                soft_targets_flat = soft_targets.reshape(-1, 4)
                
                mask = (labels_flat != -1)
                
                loss_ce = criterion_ce(outputs_flat, labels_flat)
                
                log_probs = F.log_softmax(outputs_flat / temperature, dim=1)
                kl_elementwise = criterion_kl(log_probs, soft_targets_flat)
                loss_kl = (kl_elementwise.sum(dim=1) * mask).sum() / mask.sum()
                
                loss = loss_ce + kl_weight * loss_kl
                val_loss += loss.item()

                probs = torch.softmax(outputs_flat / temperature, dim=-1)
                predicted = torch.multinomial(probs, num_samples=1).squeeze()
                
                valid_labels = labels_flat[mask]
                valid_predicted = predicted[mask]
                valid_targets = targets_flat[mask]
                
                val_total += valid_labels.size(0)
                val_correct += (valid_predicted == valid_labels).sum().item()
                val_task_correct += (valid_predicted == valid_targets).sum().item()

        # 平均計算
        num_batches = len(train_loader)
        train_loss /= num_batches
        train_ce_loss /= num_batches
        train_kl_loss /= num_batches
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0.0
        train_task_acc = 100 * train_task_correct / train_total if train_total > 0 else 0.0
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
        val_task_acc = 100 * val_task_correct / val_total if val_total > 0 else 0.0
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_task_acc'].append(train_task_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_task_acc'].append(val_task_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} (CE:{train_ce_loss:.3f}, KL:{train_kl_loss:.3f}) | Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if use_wandb:
            wandb.log({
                'epoch': epoch+1, 
                'train_loss': train_loss, 
                'train_ce_loss': train_ce_loss,
                'train_kl_loss': train_kl_loss,
                'val_loss': val_loss, 
                'val_acc': val_acc
            })

        scheduler.step(train_loss)
        
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), 'model_weights/best_flanker_rnn_model_kl.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            # 早期終了ロジック
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break
                
    return history

def main():
    config = {
        'sequence_length': 10,
        'batch_size': 64,
        'hidden_size': 16,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'num_epochs': 5000,
        'patience': 100,
        'temperature': 1.0,
        'kl_weight': 2.0, # ドキュメントの lambda [cite: 27]
    }
    
    wandb.init(project="flanker-task-rnn_kl", config=config)
    
    data_dir = DATA_PATH
    print("データセットを作成中 (KLダイバージェンス用ソフトターゲットを含む)...")
    dataset = FlankerDataset(data_dir)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    
    model = FlankerRNN(
        input_size=21,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=4,
        dropout=config['dropout']
    ).to(device)
    
    print("\nモデルの訓練を開始します (Hybrid Loss: CE + lambda * KL)...")
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=config['num_epochs'], 
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        use_wandb=True,
        temperature=config['temperature'],
        kl_weight=config['kl_weight']
    )
    
    wandb.finish()

if __name__ == "__main__":
    main()