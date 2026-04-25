from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm, Logger
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
import llm_stage
import llm_com
import pandas as pd
import random


class JSRLAfterEvalCallback(BaseCallback):
    def __init__(self, policy, logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.logger = logger
        self.best_moving_mean_reward = -np.inf
        self.tolerated_moving_mean_reward = -np.inf
        self.mean_rewards = np.full(policy.window_size, -np.inf, dtype=np.float32)

    def _on_step(self) -> bool:
        self.policy.jsrl_evaluation = False
        self.logger.record("jsrl/horizon", self.policy.horizon)

        if self.policy.strategy == "random":
            return True

        self.mean_rewards = np.roll(self.mean_rewards, 1)
        self.mean_rewards[0] = self.parent.last_mean_reward
        moving_mean_reward = np.mean(self.mean_rewards)

        self.logger.record("jsrl/moving_mean_reward", moving_mean_reward)
        self.logger.record("jsrl/best_moving_mean_reward", self.best_moving_mean_reward)
        self.logger.record("jsrl/tolerated_moving_mean_reward", self.tolerated_moving_mean_reward)
        self.logger.dump(self.num_timesteps)

        if self.mean_rewards[-1] == -np.inf or self.policy.horizon <= 0:
            return True
        elif self.best_moving_mean_reward == -np.inf:
            self.best_moving_mean_reward = moving_mean_reward
        elif moving_mean_reward >= self.tolerated_moving_mean_reward:
            self.policy.update_horizon()

        if moving_mean_reward >= self.best_moving_mean_reward:
            self.tolerated_moving_mean_reward = moving_mean_reward - self.policy.tolerance * np.abs(moving_mean_reward)
            self.best_moving_mean_reward = max(self.best_moving_mean_reward, moving_mean_reward)

        return True


class JSRLEvalCallback(EvalCallback):
    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.logger = JSRLLogger(self.logger)

    def _on_step(self) -> bool:
        self.model.policy.jsrl_evaluation = True
        return super()._on_step()


class JSRLLogger():
    def __init__(self, logger: Logger):
        self._logger = logger

    def record(self, key: str, value: Any, exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: save to log this key
        :param value: save to log this value
        :param exclude: outputs to be excluded
        """
        key = key.replace("eval/", "jsrl/")
        self._logger.record(key, value, exclude)

    def dump(self, step: int = 0) -> None:
        """
        Write all of the diagnostics from the current iteration
        """
        self._logger.dump(step)


def get_jsrl_policy(ExplorationPolicy: BasePolicy):
    class JSRLPolicy(ExplorationPolicy):
        def __init__(
            self,
            *args,
            guide_policy: BasePolicy = None,
            max_horizon: int = 0,
            horizons: List[int] = [0],
            tolerance: float = 0.0,
            strategy: str = "curriculum",
            window_size: int = 1,
            eval_freq: int = 1000,
            n_eval_episodes: int = 20,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.guide_policy = guide_policy
            self.tolerance = tolerance
            assert strategy in ["curriculum", "random"], f"strategy: '{strategy}' must be 'curriculum' or 'random'"
            self.strategy = strategy
            self.horizon_step = 0
            self.max_horizon = max_horizon
            self.horizons = horizons
            assert window_size > 0, f"window_size: {window_size} must be greater than 0"
            self.window_size = window_size
            self.eval_freq = eval_freq
            if self.strategy == "curriculum":
                self.n_eval_episodes = n_eval_episodes
            else:
                self.n_eval_episodes = 0
            self.jsrl_evaluation = False
            # self.recommender = llm_stage.RobustRecommender("/home/dell/workspace/lmf/llm_jsrl/data_junyi/envpath_data.csv")
            self.recommender = llm_stage.RobustRecommender(data_path="/home/dell/workspace/lmf/llm_jsrl/data_assist09/envpath_data_new.csv", prerequisites_path="/home/dell/workspace/lmf/llm_jsrl/data_assist09/prerequisites.csv")
            # self.recommender_com = llm_com.RobustRecommender("/home/dell/workspace/lmf/llm_jsrl/data_junyi/envpath_data.csv")
            self.action_history = []
            self.array_skill = [0, 1, 2, 3, 4]

        @property
        def horizon(self):
            return self.horizons[self.horizon_step]

        def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            timesteps: np.ndarray,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
        ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
            """
            Get the policy action from an observation (and optional hidden state).
            Includes sugar-coating to handle different observations (e.g. normalizing images).

            :param observation: the input observation
            :param timesteps: the number of timesteps since the beginning of the episode
            :param state: The last hidden states (can be None, used in recurrent policies)
            :param episode_start: The last masks (can be None, used in recurrent policies)
                this correspond to beginning of episodes,
                where the hidden states of the RNN must be reset.
            :param deterministic: Whether or not to return deterministic actions.
            :return: the model's action and the next hidden state
                (used in recurrent policies)
            """
            horizon = self.horizon
            if not self.training and not self.jsrl_evaluation:
                horizon = 0
            # 动作历史重置判断
            if timesteps == 49:
                self.action_history = []
            timesteps_lte_horizon = timesteps <= horizon
            timesteps_gt_horizon = timesteps > horizon
            if isinstance(observation, dict):
                observation_lte_horizon = {k: v[timesteps_lte_horizon] for k, v in observation.items()}
                observation_gt_horizon = {k: v[timesteps_gt_horizon] for k, v in observation.items()}
            elif isinstance(observation, np.ndarray):
                observation_lte_horizon = observation[timesteps_lte_horizon]
                observation_gt_horizon = observation[timesteps_gt_horizon]
            if state is not None:
                state_lte_horizon = state[timesteps_lte_horizon]
                state_gt_horizon = state[timesteps_gt_horizon]
            else:
                state_lte_horizon = None
                state_gt_horizon = None
            if episode_start is not None:
                episode_start_lte_horizon = episode_start[timesteps_lte_horizon]
                episode_start_gt_horizon = episode_start[timesteps_gt_horizon]
            else:
                episode_start_lte_horizon = None
                episode_start_gt_horizon = None

            action = np.zeros((len(timesteps), *self.action_space.shape), dtype=self.action_space.dtype)
            if state is not None:
                state = np.zeros((len(timesteps), *state.shape[1:]), dtype=state_lte_horizon.dtype)

            # 选取状态中的掌握程度
            if len(observation_lte_horizon) > 0 and len(observation_lte_horizon[0]) > 5:
                observation_lte_horizon_mastery = observation_lte_horizon[0][-5:]
                observation_lte_horizon_mastery = np.array([observation_lte_horizon_mastery])
            else:
                observation_lte_horizon_mastery = observation_lte_horizon
            
            if timesteps_lte_horizon.any():
                action_lte_horizon, state_lte_horizon = self.guide_policy.predict(
                    observation_lte_horizon_mastery, state_lte_horizon, episode_start_lte_horizon, deterministic
                )
                # start
                # print("zheshishenmedongxi", timesteps, self.horizons)
                # print("nihao", self.guide_policy, observation_lte_horizon)
                if timesteps[0] < 0.3 * self.horizons[0]:
                    # llm推荐模块
                    concept = self.recommender.recommend(observation_lte_horizon_mastery[0])
                    concept_id = concept["recommendations"][0]["topic_id"]
                    concept_name = concept["recommendations"][0]["topic"]
                    question = self.recommender.recommend_ques(observation_lte_horizon_mastery[0], concept_id, concept_name, self.action_history)
                    question_id = question["recommended_question"]["question_id"]
                    # print(type([question_id]), type(action_lte_horizon))
                    
                    # # llm_com推荐模块
                    # question = self.recommender_com.recommend(observation_lte_horizon_mastery[0], self.action_history)
                    # question_id = question["recommendations"][0]["id"]
                    
                    # 选择最小值模块
                    min_concept1, min_concept2 = self.find_min_second_min_indices(observation_lte_horizon_mastery[0])
                    min_question1 = self.generate_random_numbers(min_concept1)
                    min_question2 = self.generate_random_numbers(min_concept2)
                    
                    # 随机选择模块
                    actions_guide = [action_lte_horizon[0], min_question1, min_question2, question_id]
                    # actions_guide = [question_id, min_question1, min_question2]
                    action_lte_horizon = np.array([random.choice(actions_guide)])
                else:
                    # # llm推荐模块
                    # concept = self.recommender.recommend(observation_lte_horizon_mastery[0])
                    # concept_id = concept["recommendations"][0]["topic_id"]
                    # concept_name = concept["recommendations"][0]["topic"]
                    # question = self.recommender.recommend_ques(observation_lte_horizon_mastery[0], concept_id, concept_name, self.action_history)
                    # question_id = question["recommended_question"]["question_id"]
                    # # print(type([question_id]), type(action_lte_horizon))
                    
                    # # llm_com推荐模块
                    # question = self.recommender_com.recommend(observation_lte_horizon_mastery[0], self.action_history)
                    # question_id = question["recommendations"][0]["id"]
                    
                    # 选择最小值模块
                    min_concept1, min_concept2 = self.find_min_second_min_indices(observation_lte_horizon_mastery[0])
                    min_question1 = self.generate_random_numbers(min_concept1)
                    min_question2 = self.generate_random_numbers(min_concept2)
                    
                    # 随机选择模块
                    actions_guide = [action_lte_horizon[0], min_question1, min_question2]
                    # actions_guide = [question_id, min_question1, min_question2]
                    action_lte_horizon = np.array([random.choice(actions_guide)])
                
                # # 选择最小值模块
                # min_concept1, min_concept2 = self.find_min_second_min_indices(observation_lte_horizon_mastery[0])
                # min_question1 = self.generate_random_numbers(min_concept1)
                # min_question2 = self.generate_random_numbers(min_concept2)
                
                # # 随机选择模块
                # actions_guide = [action_lte_horizon[0], min_question1, min_question2]
                # # actions_guide = [question_id, min_question1, min_question2]
                # action_lte_horizon = np.array([random.choice(actions_guide)])
                
                # action[timesteps_lte_horizon] = action_guide
                # end
                action[timesteps_lte_horizon] = action_lte_horizon
                if state is not None:
                    state[timesteps_lte_horizon] = state_lte_horizon

            if timesteps_gt_horizon.any():
                action_gt_horizon, state_gt_horizon = super().predict(
                    observation_gt_horizon, state_gt_horizon, episode_start_gt_horizon, deterministic
                )
                action[timesteps_gt_horizon] = action_gt_horizon
                if state is not None:
                    state[timesteps_gt_horizon] = state_gt_horizon
            # print("predict", action)
            self.action_history.append(action[0])
            # print(len(self.action_history), timesteps, self.action_history)
            # print(observation, observation_lte_horizon, observation_gt_horizon)
            return action, state

        def update_horizon(self) -> None:
            """
            Update the horizon based on the current strategy.
            """
            if self.strategy == "curriculum":
                self.horizon_step += 1
                self.horizon_step = min(self.horizon_step, len(self.horizons) - 1)
            elif self.strategy == "random":
                self.horizons = [np.random.choice(self.max_horizon)]
        
        # # 寻找llm_stage推荐的题目对应的索引        
        # def find_index_by_name(slef, csv_path, target_name):
        #     """
        #     根据name在csv文件中查找对应题目的索引（行号）
        #     :param csv_path: csv文件的路径
        #     :param target_name: 要查找的题目name
        #     :return: 找到的索引，若未找到返回 -1
        #     """
        #     # 读取csv文件到DataFrame
        #     df = pd.read_csv(csv_path)
        #     # 获取name列等于target_name的行索引，返回的是一个Series，取第一个元素（若存在）
        #     index = df[df['name'] == target_name].index.tolist()
        #     return index[0] if index else -1
        
        # 找到状态中最小的两个元素索引
        def find_min_second_min_indices(self, lst):
            if len(lst) < 2:
                raise ValueError("List must contain at least two elements")

            min_val = float('inf')
            min_index = -1
            second_min_val = float('inf')
            second_min_index = -1
            min_indices = []  # 初始化存储最小值索引的列表

            for i, value in enumerate(lst):
                if value < min_val:
                    min_val = value
                    min_index = i
                    min_indices = [i]  # 重置最小值索引列表
                elif value == min_val:
                    min_indices.append(i)  # 添加到最小值索引列表
                elif value < second_min_val:
                    second_min_val = value
                    second_min_index = i

            # 如果第二小的值没有被设置，说明所有元素都相同，随机选择一个索引
            if second_min_index == -1:
                second_min_index = random.choice(min_indices)
            elif len(min_indices) > 1 and second_min_index not in min_indices:
                # 如果有多个最小值，且第二小的值不在最小值索引列表中，随机选择第二小的索引
                second_min_index = random.choice([i for i in min_indices if i != min_index])

            return min_index, second_min_index
        
        def generate_random_numbers(self, base):
            # 读取CSV文件
            # df = pd.read_csv('/home/dell/workspace/lmf/llm_jsrl/data_junyi/envpath_data.csv')
            df = pd.read_csv('/home/dell/workspace/lmf/llm_jsrl/data_assist09/envpath_data_new.csv')

            # 给定的数字
            # target_value = base
            target_value = self.array_skill[base]

            # 筛选出列元素值为target_value的所有行的索引
            # target_indices = df.index[df['topic_id'] == target_value].tolist()
            target_indices = df.index[df['skill_id'] == target_value].tolist()
            
            # 从匹配的行中随机抽取一个索引
            selected_index = random.choice(target_indices)
                
            return selected_index

    return JSRLPolicy


def get_jsrl_algorithm(Algorithm: BaseAlgorithm):
    class JSRLAlgorithm(Algorithm):
        def __init__(self, policy, *args, **kwargs):
            if isinstance(policy, str):
                policy = self._get_policy_from_name(policy)
            else:
                policy = policy
            policy = get_jsrl_policy(policy)
            kwargs["learning_starts"] = 0
            super().__init__(policy, *args, **kwargs)
            self._timesteps = np.zeros((self.env.num_envs), dtype=np.int32)
            # self.recommender = llm_stage.RobustRecommender("/home/dell/workspace/lmf/llm_jsrl/data_junyi/envpath_data.csv")
            # self.recommender = llm_stage.RobustRecommender(data_path="/home/dell/workspace/lmf/llm_jsrl/data_assist09/envpath_data_new.csv", prerequisites_path="/home/dell/workspace/lmf/llm_jsrl/data_assist09/prerequisites.csv")
            # self.recommender_com = llm_com.RobustRecommender("/home/dell/workspace/lmf/llm_jsrl/data_junyi/envpath_data.csv")
            self.action_history = []
            self.array_skill = [0, 1, 2, 3, 4]

        def _init_callback(
            self,
            callback: MaybeCallback,
            progress_bar: bool = False,
        ) -> BaseCallback:
            """
            :param callback: Callback(s) called at every step with state of the algorithm.
            :param progress_bar: Display a progress bar using tqdm and rich.
            :return: A hybrid callback calling `callback` and performing evaluation.
            """
            callback = super()._init_callback(callback, progress_bar)
            eval_callback = JSRLEvalCallback(
                self.env,
                callback_after_eval=JSRLAfterEvalCallback(
                    self.policy,
                    self.logger,
                    verbose=self.verbose,
                ),
                eval_freq=self.policy.eval_freq,
                n_eval_episodes=self.policy.n_eval_episodes,
                verbose=self.verbose,
            )
            callback = CallbackList(
                [
                    callback,
                    eval_callback,
                ]
            )
            callback.init_callback(self)
            return callback
        
        # 找到状态中最小的两个元素索引
        def find_min_second_min_indices(self, lst):
            if len(lst) < 2:
                raise ValueError("List must contain at least two elements")

            min_val = float('inf')
            min_index = -1
            second_min_val = float('inf')
            second_min_index = -1
            min_indices = []  # 初始化存储最小值索引的列表

            for i, value in enumerate(lst):
                if value < min_val:
                    min_val = value
                    min_index = i
                    min_indices = [i]  # 重置最小值索引列表
                elif value == min_val:
                    min_indices.append(i)  # 添加到最小值索引列表
                elif value < second_min_val:
                    second_min_val = value
                    second_min_index = i

            # 如果第二小的值没有被设置，说明所有元素都相同，随机选择一个索引
            if second_min_index == -1:
                second_min_index = random.choice(min_indices)
            elif len(min_indices) > 1 and second_min_index not in min_indices:
                # 如果有多个最小值，且第二小的值不在最小值索引列表中，随机选择第二小的索引
                second_min_index = random.choice([i for i in min_indices if i != min_index])

            return min_index, second_min_index
        
        def generate_random_numbers(self, base):
            # 读取CSV文件
            # df = pd.read_csv('/home/dell/workspace/lmf/llm_jsrl/data_junyi/envpath_data.csv')
            df = pd.read_csv('/home/dell/workspace/lmf/llm_jsrl/data_assist09/envpath_data_new.csv')

            # 给定的数字
            # target_value = base
            target_value = self.array_skill[base]

            # 筛选出列元素值为target_value的所有行的索引
            target_indices = df.index[df['skill_id'] == target_value].tolist()
            # target_indices = df.index[df['topic_id'] == target_value].tolist()
            
            # 从匹配的行中随机抽取一个索引
            selected_index = random.choice(target_indices)
                
            return selected_index

        def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
        ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
            """
            Get the policy action from an observation (and optional hidden state).
            Includes sugar-coating to handle different observations (e.g. normalizing images).

            :param observation: the input observation
            :param state: The last hidden states (can be None, used in recurrent policies)
            :param episode_start: The last masks (can be None, used in recurrent policies)
                this correspond to beginning of episodes,
                where the hidden states of the RNN must be reset.
            :param deterministic: Whether or not to return deterministic actions.
            :return: the model's action and the next hidden state
                (used in recurrent policies)
            """
            action, state = self.policy.predict(observation, self._timesteps, state, episode_start, deterministic)
            
            # if self._timesteps[0] < self.policy.horizons[0]*0.3:
            #     # 选取状态中的掌握程度
            #     if len(observation) > 0 and len(observation[0]) > 5:
            #         observation_mastery = observation[0][-5:]
            #         observation_mastery = np.array([observation_mastery])
            #         # print(observation_mastery)
            #     else:
            #         observation_mastery = observation
                    
            #     if self.env.buf_dones.any():
            #         self.action_history= []  
            #     # # llm推荐模块
            #     concept = self.recommender.recommend(observation_mastery[0])
            #     concept_id = concept["recommendations"][0]["topic_id"]
            #     concept_name = concept["recommendations"][0]["topic"]
            #     question = self.recommender.recommend_ques(observation_mastery[0], concept_id, concept_name, self.action_history)
            #     question_id = question["recommended_question"]["question_id"]
            #     #     # print(type([question_id]), type(action_lte_horizon))
                
            #     # # # llm_com推荐模块
            #     # # question = self.recommender_com.recommend(observation[0], self.action_history)
            #     # # question_id = int(question["recommendations"][0]["id"])
                    
            #     # 选择最小值模块
            #     min_concept1, min_concept2 = self.find_min_second_min_indices(observation_mastery[0])
            #     min_question1 = self.generate_random_numbers(min_concept1)
            #     min_question2 = self.generate_random_numbers(min_concept2)
                    
            #     # 随机选择模块
            #     actions_guide = [question_id, min_question1, min_question2, action[0]]
            #     action_guide = np.array([random.choice(actions_guide)])
            #     action = action_guide
            
            # else:
            
            #     # 选取状态中的掌握程度
            #     if len(observation) > 0 and len(observation[0]) > 5:
            #         observation_mastery = observation[0][-5:]
            #         observation_mastery = np.array([observation_mastery])
            #         # print(observation_mastery)
            #     else:
            #         observation_mastery = observation
                    
            #     if self.env.buf_dones.any():
            #         self.action_history= []  
            #     # # llm推荐模块
            #     # concept = self.recommender.recommend(observation_mastery[0])
            #     # concept_id = concept["recommendations"][0]["topic_id"]
            #     # concept_name = concept["recommendations"][0]["topic"]
            #     # question = self.recommender.recommend_ques(observation_mastery[0], concept_id, concept_name, self.action_history)
            #     # question_id = question["recommended_question"]["question_id"]
            #     #     # print(type([question_id]), type(action_lte_horizon))
                
            #     # # # llm_com推荐模块
            #     # # question = self.recommender_com.recommend(observation[0], self.action_history)
            #     # # question_id = int(question["recommendations"][0]["id"])
                    
            #     # 选择最小值模块
            #     min_concept1, min_concept2 = self.find_min_second_min_indices(observation_mastery[0])
            #     min_question1 = self.generate_random_numbers(min_concept1)
            #     min_question2 = self.generate_random_numbers(min_concept2)
                    
            #     # 随机选择模块
            #     actions_guide = [min_question1, min_question2, action[0]]
            #     action_guide = np.array([random.choice(actions_guide)])
            #     # print(type(action_guide))
            #     # # 去重
            #     # if action_guide[0] not in self.action_history:
            #     #     action = action_guide
            #     # else:
            #     #     unselected_actions = [a for a in range(100) if a not in self.action_history]
            #     #     action = np.array([random.choice(unselected_actions)])
            #     #     # print(type(action), action)
            #     # print("bushigemen", action_guide)
            #     action = action_guide
            
            # # # 根据权重选择模块
            # # # 定义各来源的权重（可根据实际需求调整）
            # # WEIGHTS = {
            # #     "llm": 0.2,        # 大模型权重
            # #     "edu_rule": 0.6,   # 每个教育规则动作的权重
            # #     "cql": 0.2        # CQL权重
            # # }

            # # # 随机选择模块（带权重）
            # # actions_guide = [min_question1, min_question2, action[0]]
            # # action_sources = ["edu_rule", "edu_rule", "cql"]  # 各动作对应的来源
            # # # action_sources = ["edu_rule", "edu_rule", "cql"]  # 各动作对应的来源

            # # # 为每个动作分配对应的权重
            # # weights = [WEIGHTS[source] for source in action_sources]

            # # # 归一化权重（使总和为1）
            # # normalized_weights = np.array(weights) / np.sum(weights)

            # # # 根据权重随机选择
            # # action_guide = np.array([np.random.choice(actions_guide, p=normalized_weights)])
            # # action = action_guide
            
              
            # self.action_history.append(action[0])
            # # print(len(self.action_history), action)

            self._timesteps += 1
            self._timesteps[self.env.buf_dones] = 0
            if self.policy.strategy == "random" and self.env.buf_dones.any():
                self.policy.update_horizon()
            # print("predic", self._timesteps, self.env.buf_dones)
            # print(f"""
            #     JSRL内部检查:
            #     - buf_dones来源: {type(self.env)}.{'buf_dones' if hasattr(self.env, 'buf_dones') else '无'}
            #     - buf_dones值: {getattr(self.env, 'buf_dones', '未找到')}
            #     - timesteps前: {self._timesteps}
            #     """)
            
            return action, state

    return JSRLAlgorithm
