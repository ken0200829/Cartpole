import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import wandb  # 追加

from Flanker_env import FlankerEnv
from Flanker_a2c import a2cAgent

# ハイパーパラメータ
LEARNING_RATE = 1e-4
GAMMA = 0.99
ENTROPY_BETA = 0.01
MAX_EPISODES = 20000
LOG_INTERVAL = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "fig_flanker"

os.makedirs(SAVE_DIR, exist_ok=True)

def train():
    # wandbの初期化
    wandb.init(
        project="flanker-adversarial-a2c_kl",
        config={
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "entropy_beta": ENTROPY_BETA,
            "max_episodes": MAX_EPISODES,
            "device": str(DEVICE)
        }
    )

    # 環境とエージェントの初期化
    env = FlankerEnv()
    
    # 観測空間の次元数 (17)
    input_dim = env.observation_space.shape[0]
    # 行動空間の次元数 (16)
    action_dim = env.action_space.n
    
    agent = a2cAgent(input_dim, hidden_dim=128, action_dim=action_dim).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    
    # wandbにモデルの勾配などを監視させる（オプション）
    wandb.watch(agent, log="all", log_freq=100)
    
    # ログ用
    returns_history = []
    accuracy_history = []
    loss_history = []
    recent_returns = deque(maxlen=LOG_INTERVAL)
    recent_accuracy = deque(maxlen=LOG_INTERVAL)
    
    print(f"Start training on {DEVICE}...")
    
    for episode in range(1, MAX_EPISODES + 1):
        obs, info = env.reset()
        done = False
        
        log_probs = []
        values = []
        rewards = []
        entropies = []
        masks = []
        
        episode_reward = 0
        learner_correct_count = 0
        step_count = 0
        
        # 1エピソード実行
        while not done:
            # Tensor化
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            
            # マスク取得 (infoから)
            action_mask = info.get("action_mask")
            if action_mask is not None:
                mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            else:
                mask_tensor = None
            
            # 行動選択
            action, log_prob, entropy, value = agent.get_action(obs_tensor, action_mask=mask_tensor)
            
            # 環境ステップ
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            
            done = terminated or truncated
            
            # 履歴保存
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
            masks.append(1 - int(done)) # 終了ステップは0
            
            episode_reward += reward
            if info.get("learner_correct", False):
                learner_correct_count += 1
            step_count += 1
            
            obs = next_obs
            
        # --- エピソード終了後の更新 (A2C) ---
        
        # 割引報酬和 (Returns) の計算
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        values = torch.cat(values).squeeze()
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        
        # Advantage = Returns - Values
        advantage = returns - values
        
        # Loss計算
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        entropy_loss = -entropies.mean()
        
        loss = actor_loss + 0.5 * critic_loss + ENTROPY_BETA * entropy_loss
        
        # 更新
        optimizer.zero_grad()
        loss.backward()
        # 勾配クリッピング (安定化のため)
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()
        
        # ログ集計
        learner_acc = learner_correct_count / step_count if step_count > 0 else 0
        
        returns_history.append(episode_reward)
        accuracy_history.append(learner_acc)
        loss_history.append(loss.item())
        recent_returns.append(episode_reward)
        recent_accuracy.append(learner_acc)
        
        # wandbへのログ送信
        wandb.log({
            "episode": episode,
            "return": episode_reward,
            "learner_accuracy": learner_acc,
            "total_loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy_loss": entropy_loss.item()
        })
        
        if episode % LOG_INTERVAL == 0:
            avg_ret = np.mean(recent_returns)
            avg_acc = np.mean(recent_accuracy)
            print(f"Episode {episode:04d} | Avg Return: {avg_ret:.2f} | Learner Acc: {avg_acc:.2%}")
            
    # --- 学習終了後のプロット ---
    plot_results(returns_history, accuracy_history, loss_history)
    
    # モデル保存
    torch.save(agent.state_dict(), os.path.join(SAVE_DIR, "flanker_a2c_agent.pth"))
    print("Training finished and model saved.")
    
    # wandb終了
    wandb.finish()

def plot_results(returns, accuracies, losses):
    # 移動平均の計算関数
    def moving_average(data, window_size=50):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))  # 3行に変更
    
    # 報酬プロット
    ax1.plot(returns, alpha=0.3, color='blue', label='Raw')
    if len(returns) >= 50:
        ax1.plot(moving_average(returns), color='blue', label='Moving Avg (50)')
    ax1.set_title("Adversary Reward (Higher is better)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    
    # 正答率プロット
    ax2.plot(accuracies, alpha=0.3, color='orange', label='Raw')
    if len(accuracies) >= 50:
        ax2.plot(moving_average(accuracies), color='orange', label='Moving Avg (50)')
    ax2.set_title("Learner Accuracy (Lower is better for Adversary)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.0)
    ax2.legend()

    # ロスプロット (追加)
    ax3.plot(losses, alpha=0.3, color='red', label='Raw')
    if len(losses) >= 50:
        ax3.plot(moving_average(losses), color='red', label='Moving Avg (50)')
    ax3.set_title("Training Loss")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Loss")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_results.png"))
    plt.close()

if __name__ == "__main__":
    train()