import torch
import torch.nn as nn
import numpy as np
from scipy.stats import binom

class LearnerRNN(nn.Module):
    """
    学習者モデル (RNN) の定義。
    Kerasモデルの構造に合わせて定義する必要があります。
    ここでは一般的なGRUモデルを想定しています。
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(LearnerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hidden):
        # x: (batch, seq_len, input_size)
        # hidden: (1, batch, hidden_size) -> GRU expects (num_layers, batch, hidden_size)
        
        # Kerasのstateful=Trueのような挙動を再現するため、隠れ状態を受け取って返す
        out, next_hidden = self.gru(x, hidden)
        
        # out: (batch, seq_len, hidden_size)
        # 最後のステップの出力を使う
        last_out = out[:, -1, :]
        probs = self.softmax(self.fc(last_out))
        
        return probs, next_hidden

class LearnverEnv:
    def __init__(self, model, n_actions, n_states, n_batches,
                 LOF=None,
                 LOF_weight=None,
                 device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval() # 推論モード
        
        self.n_batches = n_batches
        # モデルから隠れ層サイズを取得 (Kerasの get_layer('GRU').units に相当)
        self.n_cells = model.hidden_size 
        
        self.n_actions = n_actions
        self.n_states = n_states
        
        # 観測空間と行動空間の定義 (Gym互換)
        self.observation_space = type("observation_space", (object,), {"shape": (n_states + 1,)}) # state + trial
        self.action_space = type("action_space", (object,), {"n": self.n_actions})
        
        # 初期状態
        self.init_state = torch.zeros((1, n_batches, self.n_cells), dtype=torch.float32, device=self.device)
        
        self.null_state = np.zeros((n_batches, n_states), dtype=np.float32)
        self.null_action = np.zeros((n_batches, n_actions), dtype=np.float32)
        self.null_reward = np.zeros((n_batches, 1), dtype=np.float32)
        
        self.LOF = LOF
        self.LOF_weight = LOF_weight
        
        self.reset()

    @staticmethod
    def adv_reward(state, learner_action):
        # state: (batch, n_states), learner_action: (batch, n_actions)
        # 単純な内積のような計算
        return (state * (1 - learner_action)).sum(axis=1)

    @staticmethod
    def adv_done(curr_trial):
        return curr_trial >= 350

    def reset(self):
        self.norm_factor = 100.0
        self.curr_trial = np.zeros((self.n_batches,), dtype=np.float32)
        self.learner_rnn_state = self.init_state.clone()
        
        self.total_action = np.zeros((self.n_batches, self.n_actions), dtype=np.float32)
        self.total_reward = np.zeros((self.n_batches, 1), dtype=np.float32)
        self.total_state = np.zeros((self.n_batches, self.n_states), dtype=np.float32)
        
        self.pred_pol = None
        self.last_action = np.zeros((self.n_batches, self.n_actions), dtype=np.float32)
        self.last_reward = np.zeros((self.n_batches, 1), dtype=np.float32)
        
        self.reseted = True
        self.Q = []
        
        # Gym互換の戻り値 (obs, info)
        return self.get_adv_state(), {}

    def close(self):
        pass

    def adv_action_to_state(self, adv_action):
        # adv_action: (batch,) のインデックス
        learner_state = np.zeros((self.n_batches, self.n_states), dtype=np.float32)
        # one-hot encoding
        learner_state[np.arange(self.n_batches), adv_action] = 1.0
        return self.constrained_state(learner_state)

    def constrained_state(self, learner_state):
        # 特定の条件で状態を強制するロジック
        # total_state[:, 1] は特定の状態の累積回数と仮定
        ind = self.total_state[:, 1] >= 35
        if ind.sum() > 0:
            learner_state[ind] = np.array([1, 0], dtype=np.float32)

        ind = self.total_state[:, 1] + 350 - self.curr_trial < 36
        if ind.sum() > 0:
            learner_state[ind] = np.array([0, 1], dtype=np.float32)

        return learner_state

    def step_action_reward(self, action, reward):
        self.step(self.null_state, action, reward)

    def step_state(self, adv_action):
        state = self.adv_action_to_state(adv_action)
        self.step(state, self.null_action, self.null_reward)
        self.curr_trial += 1
        return state

    def step_adv(self, adv_action):
        # adv_action: (batch,) int or (batch, 1)
        if isinstance(adv_action, torch.Tensor):
            adv_action = adv_action.cpu().numpy()
        if adv_action.ndim > 1:
            adv_action = adv_action.flatten()

        state = None
        learner_action = None
        learner_reward = None
        
        if self.reseted:
            adv_reward = np.zeros(self.n_batches, dtype=np.float32)
            seudo_rew = 0
            adv_done = np.zeros(self.n_batches, dtype=bool) # False array
            adv_state = self.get_adv_state()
            info = {}
        else:
            state = self.step_state(adv_action)
            learner_action, pols = self.get_action()
            
            # learner_reward計算: 行動と状態の一致度など
            learner_reward = np.array([(learner_action * state).sum(1)], dtype=np.float32).T
            
            self.step_action_reward(learner_action, learner_reward)
            
            adv_reward = self.adv_reward(state, learner_action)
            seudo_rew = adv_reward.copy()
            
            # 終了判定
            done_flag = self.adv_done(self.curr_trial) # (batch,) bool
            adv_done = done_flag
            
            adv_state = self.get_adv_state()
            
            self.Q.insert(0, adv_action.copy())
            if len(self.Q) > 20:
                self.Q.pop()
            
            info = {
                'state': state, 
                'learner_action': learner_action, 
                'learner_reward': learner_reward,
                'seudo_rew': seudo_rew,
                # 正答率計算用に追加
                'learner_correct': (learner_action.argmax(axis=1) == state.argmax(axis=1)).astype(float)
            }

        # LOF (Local Outlier Factor) reward bonus
        if self.LOF_weight is not None and len(self.Q) == 20:
            # binom.pmf は numpy array を受け取れる
            q_sum = np.array(self.Q).sum(axis=0)
            bonus = self.LOF_weight * binom.pmf(q_sum, 20, 0.1)
            adv_reward += bonus

        self.reseted = False
        
        # Gym API: obs, reward, terminated, truncated, info
        terminated = adv_done
        truncated = np.zeros_like(terminated) # 今回は区別しない
        
        return adv_state, adv_reward, terminated, truncated, info

    def get_action(self):
        # self.pred_pol は (batch, n_actions) の確率分布
        if self.pred_pol is None:
            # 初期状態など
            return np.zeros((self.n_batches, self.n_actions)), np.ones((self.n_batches, self.n_actions))/self.n_actions
            
        pols = self.pred_pol
        # サンプリング: torch.multinomial を使用
        pols_tensor = torch.tensor(pols, device=self.device)
        action_indices = torch.multinomial(pols_tensor, 1).cpu().numpy().flatten()
        
        # one-hot に変換
        actions = np.zeros((self.n_batches, self.n_actions), dtype=np.float32)
        actions[np.arange(self.n_batches), action_indices] = 1.0
        
        return actions, pols

    def get_adv_state(self):
        # 敵対者への観測: [累積状態頻度, 現在の試行回数] など
        # 正規化して返す
        
        # pred_pol がない場合は均等確率
        # (元のコードでは pol を返していないが、get_adv_state内で使っている？)
        # 元コード: pol = self.pred_pol if ... else ...
        # しかし return に pol が含まれていない。
        # ここでは元コードの return に合わせる。
        
        return np.concatenate([
            self.total_state / self.norm_factor,
            self.curr_trial[:, np.newaxis] / self.norm_factor,
        ], axis=1)

    def step(self, cur_state, last_action, last_reward):
        self.total_reward += last_reward
        self.total_action += last_action
        self.total_state += cur_state

        # 学習者への入力作成
        # last_reward: (batch, 1)
        # last_action: (batch, n_actions)
        # cur_state: (batch, n_states)
        learner_input = np.concatenate([last_reward, last_action, cur_state], axis=1)
        
        # PyTorchモデルへの入力変換
        # input shape: (batch, seq_len=1, input_size)
        inp_tensor = torch.tensor(learner_input, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        with torch.no_grad():
            probs, next_hidden = self.model(inp_tensor, self.learner_rnn_state)
        
        self.pred_pol = probs.cpu().numpy() # (batch, n_actions)
        self.learner_rnn_state = next_hidden # (1, batch, hidden_size)

# テスト用コード
if __name__ == '__main__':
    # ダミーモデル
    input_size = 1 + 2 + 2 # reward + action + state
    hidden_size = 10
    output_size = 2
    n_batches = 4
    
    model = LearnerRNN(input_size, hidden_size, output_size)
    env = LearnverEnv(model, n_actions=2, n_states=2, n_batches=n_batches)
    
    obs, _ = env.reset()
    print("Initial obs:", obs.shape)
    
    done = False
    while not done:
        # ランダムな敵対者行動 (0 or 1)
        adv_action = np.random.randint(0, 2, size=n_batches)
        obs, reward, terminated, truncated, info = env.step_adv(adv_action)
        
        if terminated[0]:
            done = True
            print("Episode finished.")
            print("Final reward:", reward)
