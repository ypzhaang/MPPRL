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
        # print(self.array_skill)
        
        # 加载测试数据
        # self.data = pd.read_csv('/home/dell/workspace/lmf/llm_jsrl/data_junyi/accuracy_data.csv')
        self.data = pd.read_csv('/home/dell/workspace/lmf/llm_jsrl/data_assist09/accuracy_data_new.csv')
        
        # diff_name = 'normalized_difficulty'
        # skill_name = 'topic_id'
        diff_name = 'difficulty_score'
        skill_name = 'skill_id'
        
        self.item_difficulties = self.data[diff_name].tolist()
        self.item_skills = self.data[skill_name].tolist()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x + self.c))

    def response_update(self, item_difficulty, skill_idx, response):
        prob_correct = self.sigmoid(self.abilities[skill_idx] - item_difficulty)
        learning_rate = 0.01
        self.abilities[skill_idx] += learning_rate * (response - prob_correct) * (1 - prob_correct) * prob_correct

    def answer_test(self, abilities):
        responses = np.zeros(len(self.item_difficulties))
        for i in range(len(self.item_difficulties)):
            item_difficulty = self.item_difficulties[i]
            skill_idx0 = int(self.item_skills[i])

            # assist数据集
            skill_idx = self.array_skill.index(skill_idx0)
            
            prob_correct = self.sigmoid(abilities[skill_idx] - item_difficulty)
            responses[i] = 1 if prob_correct > 0.5 else 0
        return responses 
    
    def correctness(self, response):
        accuracy = np.sum(response == 1) / len(response)
        
        return accuracy

def parse_args(self):
    parser = ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str)
    return parser.parse_args()

# 加载模型并进行预测
def load_and_predict(self, item_difficulties, item_skills, model_path):
    with open(model_path, 'rb') as f:
        irt_model = pickle.load(f)
    responses = irt_model.answer_test(item_difficulties, item_skills)
    print(list(irt_model.abilities))
    
    return responses

# 加载模型并进行预测
def load_and_predict_single(self, item_difficulties, item_skills, model_path):
    with open('/home/dell/workspace/lmf/llm_jsrl/model_all(graph_llm_edu_cql)/5/model.pkl', 'rb') as f:
        irt_model = pickle.load(f)
    responses = irt_model.answer_test(item_difficulties, item_skills)
    print(list(irt_model.abilities))
    
    return responses

# 示例用法
if __name__ == "__main__":
    # num_skills = 5
    # item_difficulties = np.random.uniform(low=0, high=1, size=10)
    # item_skills = np.random.randint(num_skills, size=10)

    # 读取CSV文件
    # data = pd.read_csv('/home/dell/workspace/lmf/llm_jsrl/data_junyi/accuracy_data.csv')
    data = pd.read_csv('/home/dell/workspace/lmf/llm_jsrl/data_assist09/accuracy_data_new.csv')

    # 假设我们想要读取名为 'your_column_name' 的列
    # diff_name = 'normalized_difficulty'
    # skill_name = 'topic_id'
    diff_name = 'difficulty_score'
    skill_name = 'skill_id'

    # 读取指定列并转换为列表
    item_difficulties = data[diff_name].tolist()
    item_skills = data[skill_name].tolist()
    
    # # 训练模型并保存
    # train_and_save_model(num_skills, item_difficulties, item_skills)
    
    args = parse_args()
    # 加载模型并进行预测
    predicted_responses = load_and_predict(item_difficulties, item_skills, args.model_path)
    # predicted_responses = load_and_predict_single(item_difficulties, item_skills, args)
    print("预测的学生答题结果:", predicted_responses)
    
    # 计算正确率
    accuracy = np.sum(predicted_responses == 1) / len(predicted_responses)

    print("正确率是:", accuracy)