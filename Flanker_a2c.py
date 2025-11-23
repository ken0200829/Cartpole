from collections import deque
from pathlib import Path
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class a2cAgent(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, action_dim: int = 16):
        """
        A2Cエージェント (Actor-Critic)
        
        Args:
            input_dim (int): 環境からの観測状態の次元数（current trial numberとnumber of stims assignedのみ．17次元．学習者の内部状態は受け取らない）
            hidden_dim (int): 隠れ層のユニット数
            action_dim (int): 行動の次元数 (デフォルト16: 4種類のFlanker x 4種類のTarget)
        """
        super(a2cAgent, self).__init__()
        
        # 共通の特徴抽出層
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor Head: 各行動(刺激)を選択する確率(logits)を出力
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic Head: 現在の状態価値(V)を出力
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        順伝播
        Args:
            x (Tensor): 観測状態 [batch_size, input_dim]
        Returns:
            logits: 行動の非正規化確率
            value: 状態価値
        """
        features = self.feature_layer(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(self, x, action_mask=None):
        """
        行動をサンプリングする
        
        Args:
            x (Tensor): 観測状態
            action_mask (Tensor, optional): 有効な行動のマスク (1=有効, 0=無効)。
                                          制約により選べない刺激がある場合に使用。
        Returns:
            action: 選択された行動インデックス
            log_prob: 選択された行動の対数確率
            entropy: 確率分布のエントロピー (探索の指標)
            value: 状態価値
        """
        logits, value = self.forward(x)
        
        if action_mask is not None:
            # 無効な行動のlogitを非常に小さな値にして、選択確率を0にする
            # maskが0の部分を -1e9 に置き換える
            logits = torch.where(action_mask.bool(), logits, torch.tensor(-1e9, device=x.device))

        # カテゴリカル分布を作成
        dist = Categorical(logits=logits)
        
        # 行動をサンプリング
        action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value
