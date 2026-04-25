# jsrl mean_reward计算时未用到jsrl info，所以没有值，可以跑其他实验查看mean_reward使用什么变量计算
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from jsrl import get_jsrl_algorithm
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import cql_torch
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import EnvCompatibility
from stable_baselines3.common.monitor import Monitor


def main():
    env = Monitor(gym.make("GraphEnv-v0"))
    # guide_policy = DQN.load("/home/dell/workspace/lmf/llm_jsrl/model_junyi/dqn_model/dqn50").policy
    # guide_policy = cql_torch.CQLPolicyForJSRL("/home/dell/workspace/lmf/llm_jsrl/cql_model/assist09/cql_policy_assist09.pth")
    guide_policy = cql_torch.CQLPolicyForJSRL("/home/dell/workspace/lmf/llm_jsrl/cql_model/junyi/cql_policy_new.pth")
    # test_obs = np.random.randn(5)
    # print("测试动作:", cql_policy.predict(test_obs))
    max_horizon = 70
    model = get_jsrl_algorithm(DQN)(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            guide_policy=guide_policy,
            max_horizon=max_horizon,
            strategy="random"
        ),
        verbose=1,
        buffer_size=100_000,
        stats_window_size=100,
        # tensorboard_log="/home/dell/workspace/lmf/llm_jsrl/logs/junyi_70_dkt_new"  
    )
    model.learn(
        total_timesteps=10000,
        log_interval=2,    
        progress_bar=True,
        callback=EvalCallback( 
            env,
            n_eval_episodes=100,
            deterministic=True,
            # best_model_save_path="model/myenv_jsrl_random_DQN_noguide"
        ),
    )
    
    #评估
    print("evaluation")
    model.policy.jsrl_evaluation = False
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
    print(mean_reward, std_reward)
    action_list = []
    #可视化
    # print(model.get_env())
    env = model.get_env()
    obs = env.reset()
    # print("nihao", obs, obs[np.newaxis, :], obs.shape, obs[np.newaxis, :].shape)
    for i in range(50):
        action, _ = model.predict(obs, deterministic=True)
        # print(model.predict(obs, deterministic=True))
        # print("action", action)
        obs, rewards, done, _ = env.step(action)
        # env.render()
        # print(action)
        action_list.append(action[0])
        if done:
            print("The end state is:", obs[0][-5:])
            print(action_list,len(set(action_list)))
            print("the eposide ends")
            action_list = []
            obs = env.reset()

    # while 1:
    #     action, _ = model.predict(obs, deterministic=True)
    #     # print(model.predict(obs, deterministic=True))
    #     # print("action", action)
    #     obs, rewards, done, _ = env.step(action)
    #     # env.render()
    #     # print(action)
    #     action_list.append(action[0])
    #     if done:
    #         print("The end state is:", obs[0][-5:])
    #         print(action_list,len(set(action_list)))
    #         print("the eposide ends")
    #         if len(set(action_list)) == 50:
    #             break
    #         action_list = []
    #         obs = env.reset()

if __name__ == "__main__":
    main()



# # jsrl mean_reward计算时未用到jsrl info，所以没有值，可以跑其他实验查看mean_reward使用什么变量计算
# import gymnasium as gym
# from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import EvalCallback
# from jsrl import get_jsrl_algorithm
# from stable_baselines3.common.evaluation import evaluate_policy
# import numpy as np
# import cql_torch
# from stable_baselines3.common.vec_env import DummyVecEnv
# from gymnasium.wrappers import EnvCompatibility
# from stable_baselines3.common.monitor import Monitor
# import os


# def main():
#     env = Monitor(gym.make("GraphEnv-v0"))
#     eval_env = Monitor(gym.make("GraphEnv-v0"))
#     guide_policy = cql_torch.CQLPolicyForJSRL("/home/dell/workspace/lmf/llm_jsrl/cql_model/assist09/cql_policy_assist09.pth")
#     # test_obs = np.random.randn(5)
#     # print("测试动作:", cql_policy.predict(test_obs))
#     max_horizon = 50
#     model = get_jsrl_algorithm(DQN)(
#         "MlpPolicy",
#         env,
#         policy_kwargs=dict(
#             guide_policy=guide_policy,
#             max_horizon=max_horizon,
#             strategy="random"
#         ),
#         verbose=1,
#         buffer_size=100_000,
#         # stats_window_size=100,
#         # tensorboard_log="/home/dell/workspace/lmf/llm_jsrl/logs/junyi_70_dkt_new"  
#     )

#     log_dir = r"/home/dell/workspace/lmf/llm_jsrl/jsrl/log_assist09_jsrl"
#     os.makedirs(log_dir, exist_ok=True)

#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path=log_dir,
#         log_path=log_dir,          # ✅ 必须有，才会生成 evaluations.npz
#         eval_freq=500,
#         n_eval_episodes=10,
#         deterministic=True,
#         render=False
#     )

#     model.learn(
#         total_timesteps=7000,
#         log_interval=2,    
#         progress_bar=True,
#         callback=eval_callback
#     )
    
#     #评估
#     print("evaluation")
#     model.policy.jsrl_evaluation = False
#     mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
#     print(mean_reward, std_reward)
#     action_list = []
#     #可视化
#     # print(model.get_env())
#     env = model.get_env()
#     obs = env.reset()
#     # print("nihao", obs, obs[np.newaxis, :], obs.shape, obs[np.newaxis, :].shape)
#     for i in range(50):
#         action, _ = model.predict(obs, deterministic=True)
#         # print(model.predict(obs, deterministic=True))
#         # print("action", action)
#         obs, rewards, done, _ = env.step(action)
#         # env.render()
#         # print(action)
#         action_list.append(action[0])
#         if done:
#             print("The end state is:", obs[0][-5:])
#             print(action_list,len(set(action_list)))
#             print("the eposide ends")
#             action_list = []
#             obs = env.reset()

#     # while 1:
#     #     action, _ = model.predict(obs, deterministic=True)
#     #     # print(model.predict(obs, deterministic=True))
#     #     # print("action", action)
#     #     obs, rewards, done, _ = env.step(action)
#     #     # env.render()
#     #     # print(action)
#     #     action_list.append(action[0])
#     #     if done:
#     #         print("The end state is:", obs[0][-5:])
#     #         print(action_list,len(set(action_list)))
#     #         print("the eposide ends")
#     #         if len(set(action_list)) == 50:
#     #             break
#     #         action_list = []
#     #         obs = env.reset()

# if __name__ == "__main__":
#     main()
