import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定数をここに移動
DIR_MAP = {'R': 0, 'L': 1, 'U': 2, 'D': 3}

def get_stimulus_index(flanker, target):
    """FlankerとTargetの文字から0-15のインデックスを返す"""
    f_idx = DIR_MAP.get(flanker, 0)
    t_idx = DIR_MAP.get(target, 0)
    return f_idx * 4 + t_idx

def make_rnn_input_vector(flanker, target, prev_reward, prev_action_idx):
    """
    RNNへの入力ベクトル(21次元)を作成する共通関数
    Args:
        flanker (str): Flankerの方向 ('R', 'L', 'U', 'D')
        target (str): Targetの方向
        prev_reward (float): 前回の報酬 (0.0 or 1.0)
        prev_action_idx (int): 前回の行動インデックス (0-3), なければ -1
    Returns:
        np.array: shape (21,) のnumpy配列
    """
    # 1. 刺激 (16次元 one-hot)
    stim_idx = get_stimulus_index(flanker, target)
    stim_onehot = np.zeros(16, dtype=np.float32)
    stim_onehot[stim_idx] = 1.0
    
    # 2. 前回報酬 (1次元)
    prev_reward_vec = np.array([prev_reward], dtype=np.float32)
    
    # 3. 前回行動 (4次元 one-hot)
    prev_action_vec = np.zeros(4, dtype=np.float32)
    if prev_action_idx is not None and 0 <= prev_action_idx < 4:
        prev_action_vec[prev_action_idx] = 1.0

    # 結合 [刺激(16), 前回報酬(1), 前回行動(4)] = 21次元
    return np.concatenate([stim_onehot, prev_reward_vec, prev_action_vec])

class FlankerRNN(nn.Module):
    # input_size デフォルトを 21 に変更
    def __init__(self, input_size=21, hidden_size=64, num_layers=2, num_classes=4, dropout=0.3):
        super(FlankerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, num_classes)
        )

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        else:
            h0 = hidden_state
        x = self.input_dropout(x)
        out, h_n = self.gru(x, h0)
        logits_seq = self.fc(self.output_dropout(out))
        last_logits = logits_seq[:, -1, :]
        return logits_seq, last_logits, h_n

# --- LBA Logic in PyTorch ---

def _round_cdfs(x, minval=0.001, maxval=0.999):
    return torch.clamp(x, min=minval, max=maxval)

def _fix_negprob(x, minval=1e-40):
    return torch.clamp(x, min=minval)

def lba_log_likelihood(rt, choice, v, A, b, t0, s=1.0):
    """
    PyTorch implementation of LBA log-likelihood.
    """
    # Ensure inputs are correct shape
    if rt.dim() == 2: rt = rt.squeeze(-1)
    if choice.dim() == 2: choice = choice.squeeze(-1)
    
    # t_rel の最小値を保証
    t_rel = torch.clamp(rt - t0, min=0.01) 
    
    normal_dist = torch.distributions.Normal(0, 1)
    
    # Gather drift rate for the chosen option
    v_c = v.gather(1, choice.unsqueeze(1)).squeeze(1)
    
    # --- Calculate PDF of the winning accumulator f_c(t) ---
    denom = t_rel * s
    denom = torch.clamp(denom, min=1e-5)
    
    w1 = (b - t_rel * v_c) / denom
    w2 = A / denom
    
    # w1, w2 が極端な値にならないようにクリップ
    w1 = torch.clamp(w1, min=-50.0, max=50.0)
    w2 = torch.clamp(w2, min=-50.0, max=50.0)

    # ★修正: NaNを0.0に置換 (cdfにNaNを渡さないための最終防衛ライン)
    w1 = torch.nan_to_num(w1, nan=0.0)
    w2 = torch.nan_to_num(w2, nan=0.0)

    cdf_w1_w2 = normal_dist.cdf(w1 - w2)
    cdf_w1 = normal_dist.cdf(w1)
    pdf_w1_w2 = torch.exp(normal_dist.log_prob(w1 - w2))
    pdf_w1 = torch.exp(normal_dist.log_prob(w1))
    
    # Round CDFs
    cdf_w1_w2 = _round_cdfs(cdf_w1_w2)
    cdf_w1 = _round_cdfs(cdf_w1)
    
    A_safe = torch.clamp(A, min=1e-4)
    
    # PDF term calculation
    pdf_term = (1.0 / A_safe) * (
        -v_c * cdf_w1_w2 +
        s * pdf_w1_w2 +
        v_c * cdf_w1 -
        s * pdf_w1
    )
    
    # pdf_term 自体が NaN になっていないかチェックし、安全な値に置換
    pdf_term = torch.nan_to_num(pdf_term, nan=1e-40)
    log_pdf = torch.log(_fix_negprob(pdf_term))
    
    # --- Calculate CDF of non-winning accumulators (1 - F_k(t)) ---
    
    t_rel_exp = t_rel.unsqueeze(1) 
    
    if isinstance(b, float): b_exp = b
    elif b.ndim == 0: b_exp = b.unsqueeze(0)
    else: b_exp = b.unsqueeze(1)

    if isinstance(A, float): A_exp = A
    elif A.ndim == 0: A_exp = A.unsqueeze(0)
    else: A_exp = A.unsqueeze(1)
        
    A_exp_safe = torch.clamp(A_exp, min=1e-4)
    
    denom_all = t_rel_exp * s
    denom_all = torch.clamp(denom_all, min=1e-5)
    
    w1_all = (b_exp - t_rel_exp * v) / denom_all
    w2_all = A_exp / denom_all 
    
    # こちらもクリップ
    w1_all = torch.clamp(w1_all, min=-50.0, max=50.0)
    w2_all = torch.clamp(w2_all, min=-50.0, max=50.0)
    
    # ★修正: NaNを0.0に置換
    w1_all = torch.nan_to_num(w1_all, nan=0.0)
    w2_all = torch.nan_to_num(w2_all, nan=0.0)
    
    cdf_w1_w2_all = _round_cdfs(normal_dist.cdf(w1_all - w2_all))
    cdf_w1_all = _round_cdfs(normal_dist.cdf(w1_all))
    pdf_w1_w2_all = torch.exp(normal_dist.log_prob(w1_all - w2_all))
    pdf_w1_all = torch.exp(normal_dist.log_prob(w1_all))
    
    term2 = (1.0 / A_exp_safe) * (b_exp - A_exp - t_rel_exp * v) * cdf_w1_w2_all
    term3 = -(1.0 / A_exp_safe) * (b_exp - t_rel_exp * v) * cdf_w1_all
    
    w2_all_safe = torch.clamp(w2_all, min=1e-4)
    term4 = (1.0 / w2_all_safe) * pdf_w1_w2_all
    term5 = -(1.0 / w2_all_safe) * pdf_w1_all
    
    F_k = 1.0 + term2 + term3 + term4 + term5
    
    # F_k が NaN の場合の対策
    F_k = torch.nan_to_num(F_k, nan=0.0)
    F_k = torch.clamp(F_k, min=0.0, max=0.9999)
    
    log_survivor = torch.log(1.0 - F_k)
    
    sum_log_survivor = torch.sum(log_survivor, dim=1)
    log_survivor_c = log_survivor.gather(1, choice.unsqueeze(1)).squeeze(1)
    
    log_likelihood = log_pdf + (sum_log_survivor - log_survivor_c)
    
    # 最終的な尤度が NaN なら、勾配を壊さないように大きなペナルティに置換
    if torch.isnan(log_likelihood).any():
        log_likelihood = torch.nan_to_num(log_likelihood, nan=-1e9)
        
    return log_likelihood

class VAM(nn.Module):
    def __init__(self, input_size=21, hidden_size=64, num_layers=2, num_classes=4, dropout=0.3):
        super(VAM, self).__init__()
        self.rnn = FlankerRNN(input_size, hidden_size, num_layers, num_classes, dropout)
        
        # LBA Parameters (Learnable)
        self.raw_A = nn.Parameter(torch.tensor(0.5)) 
        self.raw_b_gap = nn.Parameter(torch.tensor(0.2)) 
        # 修正: 初期値を小さくする (Softplus(-2.0) approx 0.12s)
        self.raw_t0 = nn.Parameter(torch.tensor(-2.0)) 
        
    def forward(self, x, hidden_state=None):
        # RNN Forward
        logits_seq, last_logits, h_n = self.rnn(x, hidden_state)
        
        # Convert logits to drift rates (v)
        drifts_seq = F.softplus(logits_seq)
        
        # ドリフト率の上限を制限
        drifts_seq = torch.clamp(drifts_seq, max=20.0)
        
        # ★修正: RNN出力がNaNの場合の対策
        drifts_seq = torch.nan_to_num(drifts_seq, nan=0.0)
        
        # Get LBA parameters
        A = F.softplus(self.raw_A)
        b_gap = F.softplus(self.raw_b_gap)
        b = A + b_gap
        t0 = F.softplus(self.raw_t0)
        
        # Clamp t0
        t0 = torch.clamp(t0, min=0.05)
        
        return drifts_seq, (A, b, t0), h_n

    def get_lba_params(self):
        
        A = F.softplus(self.raw_A)
        b_gap = F.softplus(self.raw_b_gap)
        b = A + b_gap
        t0 = F.softplus(self.raw_t0)
        return A, b, t0

def simulate_lba(v, A, b, t0, s=1.0):
    """
    LBAモデルからRTと選択をシミュレーションする関数
    Args:
        v: (Batch, NumChoices) ドリフト率 (平均)
        A: (Batch,) or Scalar 最大開始点
        b: (Batch,) or Scalar 閾値
        t0: (Batch,) or Scalar 非決定時間
        s: Scalar ドリフト率の標準偏差 (通常1.0)
    Returns:
        rt: (Batch,) 秒単位
        choice: (Batch,) インデックス
    """
    batch_size, num_choices = v.shape
    device = v.device
    
    # パラメータの形状調整
    if isinstance(A, torch.Tensor): A = A.view(-1, 1)
    if isinstance(b, torch.Tensor): b = b.view(-1, 1)
    
    # 1. 開始点 k をサンプリング ~ Uniform(0, A)
    k = torch.rand(batch_size, num_choices, device=device) * A
    
    # 2. ドリフト率 d をサンプリング ~ Normal(v, s)
    # 各試行、各選択肢ごとに独立にサンプリング
    d = torch.normal(mean=v, std=s)
    
    # 3. 閾値到達時間を計算
    # 距離 = b - k
    dist = b - k
    
    # 時間 = 距離 / 速度
    # 速度 d が 0以下の場合は到達しない（無限大）。
    # 計算安定性のため、負のドリフトは非常に小さな正の値にクリップするか、無限大として扱う。
    # ここでは無限大として扱うために、d <= 0 の場所の時間を非常に大きくする。
    t_accumulate = dist / d
    t_accumulate[d <= 0] = float('inf')
    
    # 4. 勝者を決定 (最小時間)
    min_t, choice_idx = torch.min(t_accumulate, dim=1)
    
    # 全ての蓄積器が終わらない場合(全てd<=0)の対策
    # 実データでは稀だが、シミュレーションではあり得る。
    # ここでは便宜上、最大RTなどで埋めるか、再サンプリングが必要だが、
    # 簡易的に t0 + 大きな値 とする。
    infinite_mask = torch.isinf(min_t)
    if infinite_mask.any():
        min_t[infinite_mask] = 10.0 # 10秒など
    
    # 5. 非決定時間を加算
    if isinstance(t0, torch.Tensor): t0 = t0.view(-1)
    rt = t0 + min_t
    
    return rt, choice_idx

if __name__ == "__main__":
    # --- VAMクラスとLBAシミュレーションの動作テスト ---
    print("--- Testing VAM and simulate_lba ---")
    
    # 1. モデルのインスタンス化
    # テスト用に小さなモデルを作成
    batch_size = 2
    seq_len = 5
    input_dim = 22
    model = VAM(input_size=input_dim, hidden_size=16, num_layers=1)
    print("Model instantiated.")
    
    # 2. ダミー入力の作成 (Batch, Seq, Feature)
    # ランダムな値で入力をシミュレート
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    print(f"Input shape: {dummy_input.shape}")
    
    # 3. Forwardパスの実行
    # モデルからドリフト率とLBAパラメータを取得
    drifts_seq, (A, b, t0), _ = model(dummy_input)
    
    print(f"Drifts output shape: {drifts_seq.shape}") # (Batch, Seq, 4)
    print(f"LBA Params -> A: {A.item():.4f}, b: {b.item():.4f}, t0: {t0.item():.4f}")
    
    # 4. LBAシミュレーションの実行
    # 最後のタイムステップのドリフト率を使って、行動とRTを生成してみる
    last_drifts = drifts_seq[:, -1, :] # (Batch, 4)
    print(f"\nDrifts for simulation (last step): \n{last_drifts}")
    
    # シミュレーション関数を呼び出し
    rt, choice = simulate_lba(last_drifts, A, b, t0)
    
    print(f"\nSimulated RTs (seconds): {rt}")
    print(f"Simulated Choices (indices): {choice}")
    
    # 形状チェック
    assert rt.shape == (batch_size,), f"RT shape mismatch: {rt.shape}"
    assert choice.shape == (batch_size,), f"Choice shape mismatch: {choice.shape}"
    
    print("\n--- Test Passed Successfully ---")


