# Simple DQN

> minimal implementation of DQN in Ray/rllib

> This is modified from rllib_contrib's simpleQ.

Tested with ray 2.32.0

```python
from rllib_simple_dqn import SimpleQConfig

ray.init()

config = (
    SimpleQConfig()
    .environment("xxx")
    .framework("torch")           # only support torch
    .env_runners(
        num_env_runners = 20,
        num_envs_per_env_runner = 1,
        num_cpus_per_env_runner = 1,
        rollout_fragment_length = 300,
        batch_mode = "truncate_episodes",
        sample_timeout_s = 6000,
    )
    .resources(num_gpus = 1)
    .training(
        gamma = 0,
        train_batch_size = 64,    # sample batch size
        lr = 1e-3,
        collect_size = 6000,      # interact n step with env (env_step) in a train_step
        train_times_per_step = 1  # sample and train n times per env_step in one train_step
                                  # for example train_times_per_step = 1 and collect_size = 6000
                                  # each train_step will sample a batch from buffer
                                  # and update policy 6000 times (same behaviour with tianshou)
    )
    .evaluation(
        evaluation_duration = 3,
        evaluation_duration_unit = "episodes",
        evaluation_sample_timeout_s = 6000,
        evaluation_num_env_runners = 3,
        evaluation_parallel_to_training = False,
        evaluation_config = {
            "env": "xxx",
            "explore": False
        }
    )
)

config.exploration_config.update({
    "initial_epsilon": 0.5,
    "final_epsilon": 0.1,
    "epsilon_timesteps": 120000,
})

algo = config.build()

for _ in range(200):
    algo.train()
    algo.evaluate()
```