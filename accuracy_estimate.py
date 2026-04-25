import numpy as np
import pandas as pd
import pickle
from argparse import ArgumentParser

class OneDimensionalIRT:
    def __init__(self, num_skills):
        self.num_skills = num_skills
        self.abilities = np.zeros(shape=num_skills)  # 初始化学生能力
        self.c = np.log(4) - np.log(3)  # 猜测系数，对于四选一的选择题
        
        self.array_skill = [0, 1, 2, 3, 4]
        print(self.array_skill)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x + self.c))

    def response_update(self, item_difficulty, skill_idx, response):
        prob_correct = self.sigmoid(self.abilities[skill_idx] - item_difficulty)
        learning_rate = 0.01
        self.abilities[skill_idx] += learning_rate * (response - prob_correct) * (1 - prob_correct) * prob_correct

    def answer_test(self, item_difficulties, item_skills):
        responses = np.zeros(len(item_difficulties))
        for i in range(len(item_difficulties)):
            item_difficulty = item_difficulties[i]
            skill_idx0 = int(item_skills[i])

            # assist数据集
            skill_idx = self.array_skill.index(skill_idx0)
            
            prob_correct = self.sigmoid(self.abilities[skill_idx] - item_difficulty)
            responses[i] = 1 if prob_correct > 0.5 else 0
        return responses 

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str)
    return parser.parse_args()

# 加载模型并进行预测
def load_and_predict(item_difficulties, item_skills, model_path):
    with open(model_path, 'rb') as f:
        irt_model = pickle.load(f)
    responses = irt_model.answer_test(item_difficulties, item_skills)
    print(list(irt_model.abilities))
    # new_state = np.clip(irt_model.abilities, 0, 1)
    
    # # 计算最小值和最大值
    # arr_min = np.min(irt_model.abilities)
    # arr_max = np.max(irt_model.abilities)

    # # 归一化处理
    # normalized_arr = (irt_model.abilities - arr_min) / (arr_max - arr_min)
    
    # print(normalized_arr)
    # print(new_state)
    return responses

# 加载模型并进行预测
def load_and_predict_single(item_difficulties, item_skills, model_path):
    with open('/home/dell/workspace/lmf/llm_jsrl/model_all(graph_llm_edu_cql)/junyi60/9/model.pkl', 'rb') as f:
        irt_model = pickle.load(f)
    responses = irt_model.answer_test(item_difficulties, item_skills)
    print(list(irt_model.abilities))
    # new_state = np.clip(irt_model.abilities, 0, 1)
    
    # # 计算最小值和最大值
    # arr_min = np.min(irt_model.abilities)
    # arr_max = np.max(irt_model.abilities)

    # # 归一化处理
    # normalized_arr = (irt_model.abilities - arr_min) / (arr_max - arr_min)
    
    # print(normalized_arr)
    # print(new_state)
    return responses

# 示例用法
if __name__ == "__main__":
    # num_skills = 5
    # item_difficulties = np.random.uniform(low=0, high=1, size=10)
    # item_skills = np.random.randint(num_skills, size=10)

    # 读取CSV文件
    data = pd.read_csv('/home/dell/workspace/lmf/llm_jsrl/data_junyi/accuracy_data.csv')

    # 假设我们想要读取名为 'your_column_name' 的列
    diff_name = 'normalized_difficulty'
    skill_name = 'topic_id'

    # 读取指定列并转换为列表
    item_difficulties = data[diff_name].tolist()
    item_skills = data[skill_name].tolist()
    
    # # 训练模型并保存
    # train_and_save_model(num_skills, item_difficulties, item_skills)
    
    args = parse_args()
    # 加载模型并进行预测
    # predicted_responses = load_and_predict(item_difficulties, item_skills, args.model_path)
    predicted_responses = load_and_predict_single(item_difficulties, item_skills, args)
    print("预测的学生答题结果:", predicted_responses)
    
    # 计算正确率
    accuracy = np.sum(predicted_responses == 1) / len(predicted_responses)

    print("正确率是:", accuracy)