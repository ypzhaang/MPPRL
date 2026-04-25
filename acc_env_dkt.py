# test_model.py
import pandas as pd
import numpy as np
import torch
from learner_sim_dkt import DKTSimulator

class DKTmodel:
    def __init__(self):
        self.test_data_path = "/home/dell/workspace/lmf/llm_jsrl/data_junyi/accuracy_data.csv"
        self.test_df = pd.read_csv(self.test_data_path)

    def predict(self, knowledge_state, skill_id, difficulty):
        """预测答对概率"""
        # skill_idx = self.skill_to_idx.get(skill_id)
        if skill_id is None:
            return 0.5
        
        mastery = knowledge_state[skill_id]
        
        c = np.log(4) - np.log(3)
        # 使用IRT公式计算概率
        prob = 1 / (1 + np.exp(-(mastery - difficulty) + c))
        prob = 0.1 + 0.8 * prob  # 调整到合理范围
        return max(0.1, min(0.9, prob))

    def test_model_onilne(self, knowledge_state):
        """
        测试模型准确率 - 基于预测概率>0.5判断为答对
        Args:
            model_name: 模型名称
            test_data_path: 测试数据文件路径（只需要skill_id和difficulty）
            save_dir: 模型保存目录
        """
        
        # 4. 初始化测试
        print("\n开始测试...")
        total_predictions = len(self.test_df)
        
        # 存储详细结果
        results = []
        
        # 5. 对每条测试记录进行预测
        for i, (index, row) in enumerate(self.test_df.iterrows()):
            if i % 100 == 0:  # 每100条显示一次进度
                print(f"处理进度: {i}/{total_predictions}")
            
            # skill_id = row['skill_id']
            skill_id = row['topic_id']
            difficulty = row['normalized_difficulty']
            # difficulty = row['difficulty_score']
            
            # 预测答对概率
            try:
                pred_prob = self.predict(knowledge_state, skill_id, difficulty)
                
                # 概率>0.5则认为学生能答对这道题
                pred_correct = 1 if pred_prob > 0.55 else 0
                # print(pred_prob)
                
                # 记录结果
                results.append({
                    'index': index,
                    'skill_id': skill_id,
                    'difficulty': difficulty,
                    'pred_prob': pred_prob,
                    'pred_correct': pred_correct  # 1表示预测答对，0表示预测答错
                })
                
            except Exception as e:
                print(f"预测第{index}条记录时出错: {e}")
                continue
        
        # 6. 统计结果
        results_df = pd.DataFrame(results)
        total_questions = len(results_df)
        
        # 计算预测答对的比例（这就是我们的"准确率"）
        predicted_correct_rate = results_df['pred_correct'].mean()
        avg_pred_prob = results_df['pred_prob'].mean()
        
        # 7. 输出结果
        # print("\n" + "=" * 50)
        # print("IRT模拟测试结果")
        # print("=" * 50)
        # print(f"总测试题目数: {total_questions}")
        # print(f"预测答对题目数: {results_df['pred_correct'].sum()}")
        # print(f"预测答对比例: {predicted_correct_rate:.4f} ({predicted_correct_rate*100:.2f}%)")
        # print(f"平均预测概率: {avg_pred_prob:.4f}")
        
        return predicted_correct_rate
