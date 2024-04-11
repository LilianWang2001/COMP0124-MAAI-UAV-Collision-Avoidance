import numpy as np
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ========================== Loss Curve ========================== #
def loss(file_path):
    import numpy as np
    import re
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    avg_reward = []
    
    with open(file_path, 'rb') as f:
        event_acc = EventAccumulator(f.name)
        event_acc.Reload()
        tags = event_acc.Tags()
        
        print(tags)

        # Load the loss
        tag = 'policy_0/loss'
        loss = event_acc.Scalars(tag)
        loss_value = []
        for i in loss:
            print(i.step, i.value)
            loss_value.append(i.value)
        
        
    return loss_value

file_path_loss_un = 'D:/MAAI/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/run40/logs/policy_0/loss/policy_0/loss/events.out.tfevents.1712502532.DESKTOP-U9OC6U6'
file_path_loss_ex = 'D:/MAAI/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/run41/logs/policy_0/loss/policy_0/loss/events.out.tfevents.1712571460.DESKTOP-U9OC6U6'
loss_value_ex = loss(file_path_loss_ex)
loss_value_un = loss(file_path_loss_un)

iter = range(len(loss_value_un))

# fig = plt.figure(figsize=(10, 4))
# Plot the loss curve of greedy strategy
# ax1 = fig.add_subplot(1, 2, 1)
# ax1.plot(iter, loss_value_un, '-', color='firebrick', label='Greedy Strategy (exploitation)')
# ax1.set_title('Loss Curve (Greedy)')
# ax1.set_xlabel('episode*1000')
# ax1.set_ylabel('loss')

# ax2 = fig.add_subplot(1, 2, 2)
# Plot the loss curve of epislon-greedy strategy
# ax2.plot(iter, loss_value_ex, '-', color='dodgerblue', label='Epsilon-greedy Strategy (exploration)')
# ax2.set_title('Loss Curve (Epsilon-greedy)')
# ax2.set_xlabel('episode*1000')
# ax2.set_ylabel('loss')
# plt.show()


# ======================================== Average Reward ========================================= #
def average_reward(file_path):
    import numpy as np
    import re
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    avg_reward = []
    
    with open(file_path, 'rb') as f:
        event_acc = EventAccumulator(f.name)
        # Load the log file
        event_acc.Reload()
        tags = event_acc.Tags()
        print(tags)
        
        # Load the average episode reward
        tag2 = 'average_episode_rewards'
        reward = event_acc.Scalars(tag2)
        for r in reward:
            avg_reward.append(r.value)     
        
        
    return avg_reward



file_path3 = 'D:/MAAI/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/run40/logs/average_episode_rewards/average_episode_rewards/events.out.tfevents.1712502532.DESKTOP-U9OC6U6'
avg_reward_unexplore = average_reward(file_path3)
print("unexplore:",avg_reward_unexplore)
iter_unexplore = range(len(avg_reward_unexplore))

file_path4 = 'D:/MAAI/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/run41/logs/average_episode_rewards/average_episode_rewards/events.out.tfevents.1712571460.DESKTOP-U9OC6U6'
avg_reward2 = average_reward(file_path4)
print("explore:",avg_reward2)
iter_explore2 = range(len(avg_reward2))

plt.plot(iter_explore2, avg_reward2, '-', color='dodgerblue', label='Epsilon-greedy Strategy (exploration)') 
plt.plot(iter_unexplore, avg_reward_unexplore, '-', color='firebrick', label='Greedy Strategy (exploitation)')   

plt.title('Average Reward Curve')
plt.xlabel('episode*1000')
plt.ylabel('average episode reward')  
plt.legend()

plt.show()
