import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import glob
import os
import wandb
import matplotlib.pyplot as plt
from models import FlankerRNN, make_rnn_input_vector, DIR_MAP # <--- 追加

torch.manual_seed(42)
np.random.seed(42)
DATA_PATH = '/Users/utsumikensuke/Research/RNN_test/data/train'

# 方向のマッピング (Flanker_envと統一)
# DIR_MAP = {'R': 0, 'L': 1, 'U': 2, 'D': 3}

# デバイスの設定（Mac GPU対応）
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
    各CSVファイルを nth_play ごとに分割して、系列データを作成する
    """
    def __init__(self, data_dir, split="train"):
        self.sequences = []
        self.labels = []

        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        print(f"{len(csv_files)}個のCSVファイルを検出しました")

        total_sequences = 0

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)

            # nth_play ごとに分割
            for nth_play_val in sorted(df["nth_play"].unique()):
                df_ep = df[df["nth_play"] == nth_play_val].copy()
                df_ep = df_ep.sort_values("trial").reset_index(drop=True)

                if len(df_ep) == 0:
                    continue

                # 入力を作成 (21次元)
                sequence = self._build_sequence(df_ep)
                label = self._build_label(df_ep)

                if sequence is not None and label is not None:
                    self.sequences.append(sequence)
                    self.labels.append(label)
                    total_sequences += 1

        print(f"最終的な系列数: {total_sequences}個")

    def _build_sequence(self, df_ep):
        seq = []
        
        # 初期状態
        prev_reward = 0.0
        prev_action_idx = -1

        for _, row in df_ep.iterrows():
            flanker_char = str(row["flanker_direction"])
            target_char = str(row["target_direction"])
            
            # ★ 共通関数を使用
            input_vec = make_rnn_input_vector(flanker_char, target_char, prev_reward, prev_action_idx)
            seq.append(input_vec)
            
            # --- 次のステップのために更新 ---
            resp_char = str(row["response_direction"])
            resp_idx = DIR_MAP.get(resp_char, -1)
            
            prev_action_idx = resp_idx
            
            # 報酬更新
            t_idx = DIR_MAP.get(target_char, 0)
            if resp_idx == t_idx:
                prev_reward = 1.0
            else:
                prev_reward = 0.0

        if len(seq) == 0:
            return None
        return np.array(seq, dtype=np.float32)

    def _build_label(self, df_ep):
        """
        response_direction と target_direction から正誤ラベルを作成
        """
        labels = []
        for _, row in df_ep.iterrows():
            # DIR_MAPを使ってインデックス化
            response_idx = DIR_MAP.get(str(row["response_direction"]), -1)
            target_idx = DIR_MAP.get(str(row["target_direction"]), -1)
            
            # ここでのラベルは「次の行動予測」ではなく「正解かどうか」ではなく、
            # RNNの出力ターゲットは「回答者の行動(0-3)」であるべき
            # 元のコードの _build_label は 0/1 (正誤) を返しているように見えるが、
            # train_model内の criterion(last_logits, labels) は CrossEntropyLoss なので
            # labels はクラスインデックス (0-3) である必要がある。
            
            # 元コードの修正: ラベルは response_direction のインデックスにする
            if response_idx != -1:
                labels.append(response_idx)
            else:
                # 無効な回答の場合はとりあえず0埋め、あるいはスキップ等の処理が必要だが
                # ここでは0(U)として扱う（データ依存）
                labels.append(-1)

        return np.array(labels, dtype=np.int64)  # (T,)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx]).float()
        label = torch.from_numpy(self.labels[idx]).long()
        # train_modelで targets (正解ラベル) も使っているのでそれも返す必要がある
        # しかし Dataset の __getitem__ は (seq, label) しか返していない
        # load_data -> pad_collate_fn で targets をどう扱っているか確認が必要
        
        # 元のコードの train_model ループ: for sequences, labels, targets in train_loader:
        # つまり DataLoader は 3つの値を返している。
        # pad_collate_fn を見ると: seqs, labels = zip(*batch) となっている。
        # Dataset自体が targets を返していないと動かない。
        
        # ここでは簡易的に label (response) を返す。
        # もし train_model で task accuracy (vs target) を計算したいなら、
        # Dataset は (seq, response_label, target_label) を返す必要がある。
        
        # 今回の修正範囲は入力次元なので、Datasetの戻り値を拡張して整合性を取る。
        
        # target_label の再構築 (sequenceから逆算するのは大変なので保持しておくのがベターだが...)
        # 簡易実装: sequence内の刺激情報から target を復元する
        # sequence[t, :16] が one-hot。
        stim_idx = torch.argmax(seq[:, :16], dim=1)
        target_indices = stim_idx % 4 # idx = f*4 + t なので %4 で target
        
        return seq, label, target_indices


def load_data(data_dir, batch_size=32, split_ratio=0.8):
    """
    データセットを読み込み、train/val に分割してDataLoaderを作成する
    """
    dataset = FlankerDataset(data_dir, split="all")
    total_len = len(dataset)

    train_len = int(total_len * split_ratio)
    val_len = total_len - train_len

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn
    )

    return train_loader, val_loader


# def pad_collate_fn(batch):
#     """
#     可変長の系列をパディングしてバッチ化する
#     """
#     # Datasetが (seq, label, target) を返すようになったので修正
#     seqs, labels, targets = zip(*batch)
#     lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
#     max_len = int(lengths.max().item())

#     padded_seqs = torch.zeros(len(seqs), max_len, seqs[0].shape[1])
#     padded_labels = torch.zeros(len(seqs), max_len, dtype=torch.long)
#     padded_targets = torch.zeros(len(seqs), max_len, dtype=torch.long)

#     for i, (seq, label, target) in enumerate(zip(seqs, labels, targets)):
#         seq_len = len(seq)
#         padded_seqs[i, :seq_len] = seq
#         padded_labels[i, :seq_len] = label # torch.from_numpy(label).long() はDatasetでやってる
#         padded_targets[i, :seq_len] = target

#     return padded_seqs, padded_labels, padded_targets

def pad_collate_fn(batch):
    """
    可変長の系列をパディングしてバッチ化する
    """
    seqs, labels, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded_seqs = torch.zeros(len(seqs), max_len, seqs[0].shape[1])
    
    # 【修正】ラベルの初期値を -1 に設定（0だとクラス0と被るため）
    padded_labels = torch.full((len(seqs), max_len), -1, dtype=torch.long)
    
    # ターゲットも同様に無視したい場合は -1 にする
    padded_targets = torch.full((len(seqs), max_len), -1, dtype=torch.long)

    for i, (seq, label, target) in enumerate(zip(seqs, labels, targets)):
        seq_len = len(seq)
        padded_seqs[i, :seq_len] = seq
        padded_labels[i, :seq_len] = label 
        padded_targets[i, :seq_len] = target

    return padded_seqs, padded_labels, padded_targets


# def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, patience=10, use_wandb=True, temperature=1.0):
#     criterion = nn.CrossEntropyLoss()#Logitを入力とし，内部でSoftmaxを通してからロスを計算してくれる．
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
#     if use_wandb:
#         wandb.watch(model, criterion, log="all", log_freq=10)
    
#     nogo_class_index = 0 # U
    
#     best_train_loss = float('inf')
#     best_val_acc = 0.0
#     best_val_loss = float('inf')
#     epochs_no_improve = 0
#     early_stop = False
    
#     history = {
#         'train_loss': [], 'train_acc': [], 'train_task_acc': [], 'train_recall': [],
#         'val_loss': [], 'val_acc': [], 'val_task_acc': [], 'val_recall': []
#     }
    
#     for epoch in range(num_epochs):
#         if early_stop:
#             print(f"\n早期停止: {patience}エポック連続で訓練ロスが改善しませんでした")
#             break
            
#         model.train()
#         train_loss = 0.0
#         train_correct = 0
#         train_task_correct = 0
#         train_total = 0
#         train_tp = 0
#         train_fn = 0
        
#         # pad_collate_fn は (seqs, labels, targets, lengths) を返すように修正した方が良いが
#         # 既存の pad_collate_fn は (seqs, labels, targets) を返している (Dataset変更後)
#         # いや、pad_collate_fn も修正した。
        
#         for sequences, labels, targets in train_loader:
#             sequences = sequences.to(device)
#             labels = labels.to(device) # (Batch, Seq)
#             targets = targets.to(device)
            
#             # マスク作成 (パディング部分を無視するため)
#             # labels が 0 (U) の場合とパディングの 0 が区別できない問題があるが、
#             # ここでは簡易的に「最後のステップ」のみで学習する形にする
#             # (RNNのMany-to-One構成)
            
#             # 最後の有効なステップのインデックスを取得したいが lengths がない。
#             # 仕方ないので、パディングされていないと仮定して -1 を使うか、
#             # 全ステップで学習するか。
#             # ここでは「全ステップ学習 (Many-to-Many)」に変更する。
            
#             outputs_seq, _, _ = model(sequences) # (Batch, Seq, Class)
            
#             # Flatten
#             outputs_flat = outputs_seq.reshape(-1, 4)
#             labels_flat = labels.reshape(-1)
            
#             # パディング部分(0)も学習してしまうが、とりあえず動くようにする
#             loss = criterion(outputs_flat, labels_flat)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
            
#             # 精度計算 (全ステップ)
#             probs = torch.softmax(outputs_flat / temperature, dim=-1)
#             predicted = torch.multinomial(probs, num_samples=1).squeeze()
            
#             train_total += labels_flat.size(0)
#             train_correct += (predicted == labels_flat).sum().item()
#             train_task_correct += (predicted == targets.reshape(-1)).sum().item()
            
#             train_tp += ((predicted == nogo_class_index) & (labels_flat == nogo_class_index)).sum().item()
#             train_fn += ((predicted != nogo_class_index) & (labels_flat == nogo_class_index)).sum().item()
            
#         # 検証フェーズ (同様に全ステップ評価)
#         model.eval()
#         val_loss = 0.0
#         val_correct = 0
#         val_task_correct = 0
#         val_total = 0
#         val_tp = 0
#         val_fn = 0
        
#         val_probs_all = []
#         val_labels_all = []
#         val_predicted_all = []

#         with torch.no_grad():
#             for sequences, labels, targets in val_loader:
#                 sequences = sequences.to(device)
#                 labels = labels.to(device)
#                 targets = targets.to(device)

#                 outputs_seq, _, _ = model(sequences)
#                 outputs_flat = outputs_seq.reshape(-1, 4)
#                 labels_flat = labels.reshape(-1)
                
#                 loss = criterion(outputs_flat, labels_flat)
#                 val_loss += loss.item()

#                 probs = torch.softmax(outputs_flat / temperature, dim=-1)
#                 predicted = torch.multinomial(probs, num_samples=1).squeeze()

#                 val_total += labels_flat.size(0)
#                 val_correct += (predicted == labels_flat).sum().item()
#                 val_task_correct += (predicted == targets.reshape(-1)).sum().item()

#                 val_tp += ((predicted == nogo_class_index) & (labels_flat == nogo_class_index)).sum().item()
#                 val_fn += ((predicted != nogo_class_index) & (labelsFlat == nogo_class_index)).sum().item()
                
#                 # ログ用 (バッチ次元を残すため reshape 前のものを使う)
#                 # ここで保存する配列の形状は (Batch, SeqLen, Class) や (Batch, SeqLen)
#                 # バッチごとに SeqLen が異なるため、単純な concatenate はできない
#                 val_probs_all.append(torch.softmax(outputs_seq, dim=-1).cpu().numpy())
#                 val_labels_all.append(labels.cpu().numpy()) # labels_flatではなくlabelsを使う
#                 val_predicted_all.append(predicted.view(labels.shape).cpu().numpy()) # shapeを戻す

#         # バッチを連結して1つの配列にする処理の修正
#         if len(val_probs_all) > 0:
#             # 全バッチの中での最大系列長を見つける
#             max_seq_len = max(arr.shape[1] for arr in val_probs_all)
            
#             # パディング関数
#             def pad_batch(arr_list, target_len, pad_val=0):
#                 padded_list = []
#                 for arr in arr_list:
#                     # arr shape: (Batch, Seq, ...)
#                     batch_size, seq_len = arr.shape[:2]
#                     if seq_len < target_len:
#                         pad_width = [(0, 0), (0, target_len - seq_len)] + [(0, 0)] * (arr.ndim - 2)
#                         padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=pad_val)
#                         padded_list.append(padded_arr)
#                     else:
#                         padded_list.append(arr)
#                 return np.concatenate(padded_list, axis=0)

#             # パディングしてから結合
#             val_probs_all = pad_batch(val_probs_all, max_seq_len, pad_val=0.0)
#             val_labels_all = pad_batch(val_labels_all, max_seq_len, pad_val=-1) # ラベルのパディングは-1など区別できる値推奨
#             val_predicted_all = pad_batch(val_predicted_all, max_seq_len, pad_val=-1)
        
#         train_loss /= len(train_loader)
#         train_acc = 100 * train_correct / train_total
#         train_task_acc = 100 * train_task_correct / train_total
#         val_loss /= len(val_loader)
#         val_acc = 100 * val_correct / val_total
#         val_task_acc = 100 * val_task_correct / val_total
        
#         train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
#         val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        
#         history['train_loss'].append(train_loss)
#         history['train_acc'].append(train_acc)
#         history['train_task_acc'].append(train_task_acc)
#         history['train_recall'].append(train_recall)
#         history['val_loss'].append(val_loss)
#         history['val_acc'].append(val_acc)
#         history['val_task_acc'].append(val_task_acc)
#         history['val_recall'].append(val_recall)
        
#         print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
#         if use_wandb:
#             wandb.log({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})

#         scheduler.step(train_loss)
        
#         if train_loss < best_train_loss:
#             best_train_loss = train_loss
#             torch.save(model.state_dict(), 'model_weights/best_flanker_rnn_model.pth')
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 print("Early stopping")
#                 break
                
#     return history


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, patience=10, use_wandb=True, temperature=1.0):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # 変更: weight_decay=1e-4 (または 1e-5) を追加
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    if use_wandb:
        wandb.watch(model, criterion, log="all", log_freq=10)
    
    # ここでの nogo_class_index=0 は DIR_MAP['R'] ですが、
    # 実際はFlanker_envでの「nogo」の定義に基づき調整してください。
    nogo_class_index = 0 
    
    best_train_loss = float('inf')
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    # early_stop = True  <-- 削除: これがTrueだと即座に終了してしまうため
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_task_acc': [], 'train_recall': [],
        'val_loss': [], 'val_acc': [], 'val_task_acc': [], 'val_recall': []
    }
    
    for epoch in range(num_epochs):
        # 削除: ループ先頭でのチェックを削除
        # if early_stop:
        #     print(f"\n早期停止: {patience}エポック連続で訓練ロスが改善しませんでした")
        #     break
            
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_task_correct = 0
        train_total = 0
        train_tp = 0
        train_fn = 0
        
        for sequences, labels, targets in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device) # (Batch, Seq)
            targets = targets.to(device)
            
            outputs_seq, _, _ = model(sequences) # (Batch, Seq, Class)
            
            # Flatten して全ステップで学習
            outputs_flat = outputs_seq.reshape(-1, 4)
            labels_flat = labels.reshape(-1)
            targets_flat = targets.reshape(-1) # ターゲットもフラット化
            
            # 【修正点2】パディング部分(-1)は criterion により無視される
            loss = criterion(outputs_flat, labels_flat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 精度計算 (パディングを無視するためマスク処理が必要)
            probs = torch.softmax(outputs_flat / temperature, dim=-1)
            predicted = torch.multinomial(probs, num_samples=1).squeeze()
            
            # 【修正点3】パディング以外の有効な要素を抽出するためのマスク
            mask = (labels_flat != -1)
            
            # マスクされた有効なラベルと予測値
            valid_labels = labels_flat[mask]
            valid_predicted = predicted[mask]
            valid_targets = targets_flat[mask] # ターゲットもマスク
            
            # 【修正点4】分母を有効なデータ数のみにする
            train_total += valid_labels.size(0) 
            train_correct += (valid_predicted == valid_labels).sum().item()
            train_task_correct += (valid_predicted == valid_targets).sum().item()
            
            # 【修正点5】Recall計算も有効なデータのみで行う
            train_tp += ((valid_predicted == nogo_class_index) & (valid_labels == nogo_class_index)).sum().item()
            train_fn += ((valid_predicted != nogo_class_index) & (valid_labels == nogo_class_index)).sum().item()
            
        # 検証フェーズ (同様にパディング無視のマスク処理)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_task_correct = 0
        val_total = 0
        val_tp = 0
        val_fn = 0
        
        val_probs_all = []
        val_labels_all = []
        val_predicted_all = []

        with torch.no_grad():
            for sequences, labels, targets in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                targets = targets.to(device)

                outputs_seq, _, _ = model(sequences)
                outputs_flat = outputs_seq.reshape(-1, 4)
                labels_flat = labels.reshape(-1)
                targets_flat = targets.reshape(-1)
                
                # パディングは ignore_index=-1 で無視される
                loss = criterion(outputs_flat, labels_flat)
                val_loss += loss.item()

                probs = torch.softmax(outputs_flat / temperature, dim=-1)
                predicted = torch.multinomial(probs, num_samples=1).squeeze()
                
                # 【修正点6】マスクを作成
                mask = (labels_flat != -1)
                
                valid_labels = labels_flat[mask]
                valid_predicted = predicted[mask]
                valid_targets = targets_flat[mask]
                
                # 【修正点7】分母を有効なデータ数のみにする
                val_total += valid_labels.size(0)
                val_correct += (valid_predicted == valid_labels).sum().item()
                val_task_correct += (valid_predicted == valid_targets).sum().item()

                val_tp += ((valid_predicted == nogo_class_index) & (valid_labels == nogo_class_index)).sum().item()
                val_fn += ((valid_predicted != nogo_class_index) & (valid_labels == nogo_class_index)).sum().item()
                
                # ロギング用のデータ収集は、パディングを除外する必要があるため、少し複雑です。
                # ここでは簡易のため元のロジックを維持し、パディングを -1 で埋めます。
                val_probs_all.append(torch.softmax(outputs_seq, dim=-1).cpu().numpy())
                val_labels_all.append(labels.cpu().numpy()) 
                val_predicted_all.append(predicted.view(labels.shape).cpu().numpy()) 
        
        # バッチを連結して1つの配列にする処理 (ここは元コードのまま)
        if len(val_probs_all) > 0:
            max_seq_len = max(arr.shape[1] for arr in val_probs_all)
            
            def pad_batch(arr_list, target_len, pad_val=0):
                padded_list = []
                for arr in arr_list:
                    batch_size, seq_len = arr.shape[:2]
                    if seq_len < target_len:
                        pad_width = [(0, 0), (0, target_len - seq_len)] + [(0, 0)] * (arr.ndim - 2)
                        # 【修正点8】ラベルと予測値はパディングを -1 で埋める
                        if arr.ndim == 2: # ラベルや予測値の場合
                            padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=-1) 
                        else: # 確率の場合
                            padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=pad_val)
                        padded_list.append(padded_arr)
                    else:
                        padded_list.append(arr)
                return np.concatenate(padded_list, axis=0)

            val_probs_all = pad_batch(val_probs_all, max_seq_len, pad_val=0.0)
            val_labels_all = pad_batch(val_labels_all, max_seq_len, pad_val=-1) # ラベルは -1
            val_predicted_all = pad_batch(val_predicted_all, max_seq_len, pad_val=-1) # 予測値も -1
        
        train_loss /= len(train_loader)
        # 【修正点9】分母が 0 にならないようにチェック
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0.0
        train_task_acc = 100 * train_task_correct / train_total if train_total > 0 else 0.0
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
        val_task_acc = 100 * val_task_correct / val_total if val_total > 0 else 0.0
        
        train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_task_acc'].append(train_task_acc)
        history['train_recall'].append(train_recall)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_task_acc'].append(val_task_acc)
        history['val_recall'].append(val_recall)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if use_wandb:
            wandb.log({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})

        scheduler.step(train_loss)
        
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), 'model_weights/best_flanker_rnn_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break
                
    return history


def main():
    # ハイパーパラメータの設定
    config = {
        'sequence_length': 10,
        'batch_size': 64,
        'hidden_size': 16,
        'num_layers': 1,
        'num_classes': 4,
        'learning_rate': 0.001,
        'num_epochs': 5000,
        # 変更: 0.0 -> 0.3 または 0.5 に上げる
        'dropout': 0.5, 
        'use_wandb': True,
        "patience": 100
    }
    
    wandb.init(project="flanker-task-rnn_v2", config=config)
    
    data_dir = DATA_PATH
    print("データセットを作成中...")
    # sequence_length は Dataset 内では使われていないが、一応渡すなら修正が必要
    # ここでは Dataset の引数から削除されているので渡さない
    dataset = FlankerDataset(data_dir)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    
    # モデル作成: input_size=21
    model = FlankerRNN(
        input_size=21,  # 変更: 16+1+4
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=4,
        dropout=config['dropout']
    ).to(device)
    
    print("\nモデルの訓練を開始します...")
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=config['num_epochs'], 
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        use_wandb=True,
        temperature=config['temperature']
    )
    
    wandb.finish()

if __name__ == "__main__":
    main()