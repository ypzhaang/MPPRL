# import pandas as pd
# import random
# import ollama
# import json
# import re
# from typing import List, Dict, Union

# class RobustRecommender:
#     def __init__(self, data_path: str):
#         """初始化加载数据"""
#         self.questions = self._load_data(data_path)
#         self.concepts = self._load_concepts()
#         # self.client = ollama.Client(host='http://0.0.0.0:11435')
    
#     def _load_data(self, path: str) -> List[Dict]:
#         """读取CSV数据并预处理"""
#         df = pd.read_csv(path)
#         return [
#             {
#                 "name": row["name"],
#                 "topic": row["topic"],
#                 "area": row["area"],
#                 "difficulty": float(row["normalized_difficulty"]),
#                 "topic_id": int(row["topic_id"])
#             }
#             for _, row in df.iterrows()
#         ]
        
#     def _load_concepts(self) -> List[Dict]:
#         """从问题数据中提取不重复的知识点信息"""
#         unique_concepts = set()
#         concepts_list = []
        
#         for question in self.questions:
#             # 验证 question 是否包含所需字段
#             if not all(key in question for key in ["topic", "topic_id", "area"]):
#                 continue  # 跳过不完整的数据
            
#             concept_key = (question["topic"], question["topic_id"], question["area"])
            
#             if concept_key not in unique_concepts:
#                 unique_concepts.add(concept_key)
#                 concepts_list.append({
#                     "topic": question["topic"],
#                     "topic_id": question["topic_id"],
#                     "area": question["area"]
#                 })
        
#         # 验证结果是否为字典列表
#         if concepts_list and not isinstance(concepts_list[0], dict):
#             raise ValueError("知识点数据格式错误，期望字典列表")
        
#         return concepts_list
    
#     def recommend(
#         self,
#         mastery_levels: List[float],
#         num_recommendations: int = 1,
#     ) -> Union[Dict, List[Dict]]:
#         """
#         根据当前的知识点掌握状态，选出要进行学习的知识点
#         """
        
#         # 1. 构建Prompt
#         prompt = f"""
#         ### 任务：通过科学推荐实现所有知识点掌握度最大化 
#         ### 核心逻辑：帮助学生高效提升所有知识点的掌握程度，实现最大化学习收益。
#         ### 掌握度与知识点的对应规则：  
#         mastery_levels 的索引 0-4 ，严格对应 topic_id 0-4  
#         即 mastery_levels[0] → topic_id=0 ，mastery_levels[1] → topic_id=1 ，以此类推  
#         ### 学生知识点掌握度（topic_id 0-4）: {mastery_levels}
        
#         ### 候选知识点（topic_id 0-4 分别对应掌握度索引 0-4） ：
#         {self._format_concept(self.concepts)}
        
#         ### 领域知识参考：{self.concepts[0]['area']}（如 arithmetic ）代表知识领域，同领域知识点存在学习先后、基础支撑等关联  
#         ### 决策要求
#         基于以下信息，请严格选出1个最有助于提升整体掌握度的知识点：
#         1. 当前各知识点的掌握度数值
#         2. 知识点之间的逻辑关联（如先决知识、领域归属）
#         3. 提升该知识点对整体知识体系的贡献度
        
#         ### 输出要求（严格JSON格式）:
#         {{
#             "recommendations": [
#                 {{
#                     "topic": "知识点名称",
#                     "topic_id": 对应ID,
#                     "reason": "需同时说明：①当前掌握度分析 ②对整体目标的贡献，请使用中文回答"
#                 }}
#             ]
#         }}
#         """
        
#         # print(prompt)
        
#         # 4. 调用大模型
#         response = ollama.chat(
#             model='qwen2.5:7b',
#             messages=[{'role': 'user', 'content': prompt}],
#             options={'temperature': 0.3}  # 降低随机性
#         )
        
#         # 5. 健壮解析
#         return self._safe_parse_concept(response['message']['content'], mastery_levels)
    
#     def recommend_ques(
#         self,
#         mastery_levels: List[float],
#         topic_id: int,
#         topic_name: str,
#         question_history
#     ) -> Union[Dict, List[Dict]]:
#         """根据推荐的 topic_id 获取对应的题目"""
#         # 从 self.questions 中筛选出 topic_id 对应的题目
#         related_questions = [q for q in self.questions if q["topic_id"] == topic_id]
        
#         # 根据历史推荐序列对问题进行过滤
#         path = "/home/dell/workspace/lmf/llm_jsrl/data_junyi/envpath_data.csv"
#         df = pd.read_csv(path)
#         filtered_questions = [
#             q for q in related_questions 
#             if df[df['name'] == q['name']].index.tolist()[0] not in question_history
#         ]
#         # print(self._format_questions(filtered_questions), question_history)
        
#         # 构建引导词
#         prompt_ques = f"""
#         ### 任务：基于已选知识点和学生掌握情况，从全部候选题目中推荐 1 道最适配的题目，从而最大化学生对知识点的掌握程度，确保学生能力与题目难度适配
#         ### 核心逻辑：
#         1. 已选定知识点 topic_id 为 {topic_id} ({topic_name})，对应知识点掌握程度为 {mastery_levels[topic_id]}
#         2. 需从以下该知识点对应的全部题目中选择,候选题目（共{len(related_questions)}道相关题，name为题目名称，question_id为题目在原数据集的序号（包含难度和题目文本信息）:
#         {self._format_questions(filtered_questions)}
        
#         #### 决策要求(必须严格执行)
#         1. 强制过滤历史题目：历史推荐序列为{question_history}，必须先遍历所有题目，排除question_id在上述序列中的题目，再进入匹配学生能力环节。如题目的question_id为70，在历史推荐序列[55, 20, 41, 70]中，不予以推荐
#         2. 匹配学生能力：进行题目推荐时，从过滤后的题目中，考虑学生当前的学习能力，选出适合当前学生能力的题目        

#         ### 输出要求（严格 JSON 格式）：
#         {{
#             "recommended_question": {{
#                 "name": "题目名称(name)",
#                 "difficulty": 题目难度数值,
#                 "topic_id": {topic_id},
#                 "question_id":题目在原数据集的序号question_id,
#                 "reason": "解释该题目如何有效提升掌握度（如巩固基础、挑战高阶思维等），请使用中文回答"  
#             }}
#         }} 
#         """
        
#         # print(prompt_ques)
        
#         # 4. 调用大模型
#         response = ollama.chat(
#             model='qwen2.5:7b',
#             messages=[{'role': 'user', 'content': prompt_ques}],
#             options={'temperature': 0.3}  # 降低随机性
#         )

#         # 5. 健壮解析
#         return self._safe_parse_question(response['message']['content'], topic_id, mastery_levels)
    
        
    
#     def _format_questions(self, questions: List[Dict]) -> str:
#         path = "/home/dell/workspace/lmf/llm_jsrl/data_junyi/envpath_data.csv"
#         df = pd.read_csv(path)
#         """格式化题目展示"""
#         return "\n".join(
#             f"{i+1}. name:{q['name']}, question_id:{df[df['name'] == q['name']].index.tolist()[0]} (topic_id:{q['topic_id']}, difficulty:{q['difficulty']:.2f})"
#             for i, q in enumerate(questions)
#         )
    
#     def _format_concept(self, concepts: List[Dict]) -> str:
#         """格式化知识点：突出领域、topic_id、知识点名称"""
#         return "\n".join([
#             f"topic_id {c['topic_id']}: {c['topic']} （领域：{c['area']} ）" 
#             for c in self.concepts
#         ])
    
#     def _safe_parse_concept(self, text: str, mastery_levels: List[float]) -> Dict:
#         """知识点推荐解析：失败时 fallback 到真实最低掌握度知识点"""
#         try:
#             json_str = re.search(r'\{.*\}', text, re.DOTALL).group()
#             result = json.loads(json_str)
#             # # 额外校验：若模型乱推荐，强制用真实最低掌握度知识点兜底
#             # min_mastery_idx = mastery_levels.index(min(mastery_levels))
#             # if result["recommendation"]["topic_id"] != min_mastery_idx:
#             #     # 兜底：选真实最低掌握度知识点
#             #     return self._fallback_to_min_mastery_concept(min_mastery_idx)
#             return result
#         except Exception as e:
#             # 解析全失败，直接选真实最低掌握度知识点
#             mastery_levels = mastery_levels.tolist() 
#             min_mastery_idx = mastery_levels.index(min(mastery_levels))
#             return self._fallback_to_min_mastery_concept(min_mastery_idx)
        
#     def _fallback_to_min_mastery_concept(self, topic_id: int) -> Dict:
#         """构造真实最低掌握度知识点的推荐结果"""
#         concept = next(c for c in self.concepts if c["topic_id"] == topic_id)
#         return {{
#             "recommendations": {{
#                 "topic": concept["topic"],
#                 "topic_id": topic_id,
#                 "reason": "解析异常，自动选真实掌握度最低的知识点"
#             }}
#         }}
        
#     # def _safe_parse_question(self, text: str, topic_id: int, mastery_levels: List[float]) -> Dict:
#     #     """题目推荐解析：失败时 fallback 到难度最适配的题"""
#     #     try:
#     #         json_str = re.search(r'\{.*\}', text, re.DOTALL).group()
#     #         return json.loads(json_str)
#     #     except Exception as e:
#     #         # 解析失败，选难度最接近掌握度的题
#     #         return self._fallback_to_best_difficulty_question(topic_id, mastery_levels)

#     # def _fallback_to_best_difficulty_question(self, topic_id: int, mastery_levels: List[float]) -> Dict:
#     #     """兜底逻辑：选难度最接近当前知识点掌握度的题"""
#     #     topic_questions = [q for q in self.questions if q["topic_id"] == topic_id]
#     #     target_mastery = mastery_levels[topic_id]
#     #     # 按难度差排序，选最接近的
#     #     topic_questions.sort(key=lambda x: abs(x["difficulty"] - target_mastery))
#     #     best_question = topic_questions[0] if topic_questions else {}
#     #     return {{
#     #         "recommended_question": {{
#     #             "name": best_question.get("name", "无适配题目"),
#     #             "difficulty": best_question.get("difficulty", 0.0),
#     #             "reason": "解析异常，自动选难度最适配的题目"
#     #         }}
#     #     }}
    
#     def _safe_parse_question(self, text: str, topic_id: int, mastery_levels: List[float]) -> Dict:
#         """增强版JSON解析，包含详细调试信息"""
#         try:
#             # # 打印原始文本的基本信息
#             # print(f"原始文本长度: {len(text)}")
#             # print(f"原始文本前200个字符: {text[:200]}")
#             # print(f"原始文本后200个字符: {text[-200:]}")
            
#             # 1. 严格提取JSON内容（处理Markdown格式）
#             json_match = re.search(r'```json\n?(\{.*?\})\n?```', text, re.DOTALL)
#             if not json_match:
#                 # 尝试直接匹配JSON对象
#                 json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            
#             if not json_match:
#                 raise ValueError("未找到有效的JSON结构")
            
#             json_str = json_match.group(1)
            
#             # 2. 清理字符串（移除不可见字符）
#             cleaned_json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', json_str)
            
#             # # 3. 打印提取的JSON内容（用于验证）
#             # print(f"提取的JSON内容（长度{len(cleaned_json_str)}）:")
#             # print(cleaned_json_str)
            
#             # 4. 尝试解析
#             try:
#                 return json.loads(cleaned_json_str)
#             except json.JSONDecodeError as e:
#                 # 打印详细的错误位置信息
#                 error_pos = e.pos
#                 context_start = max(0, error_pos - 50)
#                 context_end = min(len(cleaned_json_str), error_pos + 50)
#                 print(f"错误位置: 字符 {error_pos}, 行 {e.lineno}, 列 {e.colno}")
#                 print(f"错误附近内容: {cleaned_json_str[context_start:context_end]}")
#                 raise
        
#         except Exception as e:
#             print(f"JSON解析失败: {str(e)}")
#             # 记录原始文本用于后续分析
#             with open("failed_json.txt", "w", encoding="utf-8") as f:
#                 f.write(text)
#             return self._fallback_to_best_difficulty_question(topic_id, mastery_levels)

#     def _fallback_to_best_difficulty_question(self, topic_id: int, mastery_levels: List[float]) -> Dict:
#         """兜底逻辑：选难度最接近当前知识点掌握度的题"""
#         topic_questions = [q for q in self.questions if q["topic_id"] == topic_id]
#         if not topic_questions:
#             return {
#                 "recommended_question": {
#                     "question_id": -1,
#                     "name": "无适配题目",
#                     "difficulty": 0.0,
#                     "reason": "当前主题下没有可用题目"
#                 }
#             }
        
#         target_mastery = mastery_levels[topic_id]
#         # 按难度差排序，选最接近的
#         topic_questions.sort(key=lambda x: abs(x["difficulty"] - target_mastery))
#         best_question = topic_questions[0]
        
#         path = "/home/dell/workspace/lmf/llm_jsrl/data_junyi/envpath_data.csv"
#         df = pd.read_csv(path)
#         # 处理题目可能不存在于CSV中的情况
#         question_ids = df[df['name'] == best_question["name"]].index.tolist()
#         best_question_id = question_ids[0] if question_ids else -1
        
#         return {
#             "recommended_question": {
#                 "question_id": best_question_id,
#                 "name": best_question.get("name", "无名称题目"),
#                 "difficulty": best_question.get("difficulty", 0.0),
#                 "topic_id": topic_id,
#                 "reason": f"解析异常，自动选择难度最适配的题目（难度差: {abs(best_question['difficulty'] - target_mastery):.2f}）"
#             }
#         }

# # 使用示例
# if __name__ == "__main__":
#     recommender = RobustRecommender("/home/dell/workspace/lmf/llm_jsrl/data_junyi/envpath_data.csv")
#     # print(recommender.questions)
#     # print(recommender.concepts)
#     mastery_levels=[0.06, 0.045, 0.09, 0.059, 0.011]
#     result = recommender.recommend(
#         mastery_levels
#     )
#     print("稳定推荐结果：")
#     print(json.dumps(result, indent=2, ensure_ascii=False))
#     concept_id = result["recommendations"][0]["topic_id"]
#     concept_name = result["recommendations"][0]["topic"]
#     question_his = [45, 29, 31, 92, 84, 28, 107, 2, 101, 80, 10, 44, 67, 132, 8, 120, 5, 89, 130, 85, 37, 71, 68, 3, 27, 87, 72, 135, 118, 56, 18, 69, 13, 15, 48, 91, 70, 79, 74]
#     questions = recommender.recommend_ques(mastery_levels, concept_id, concept_name, question_his)
#     print(json.dumps(questions, indent=2, ensure_ascii=False))


# assist数据集
import pandas as pd
import random
import ollama
import json
import re
from typing import List, Dict, Union

class RobustRecommender:
    def __init__(self, data_path: str, prerequisites_path: str = None):
        """初始化加载数据"""
        self.data_path = data_path  # 保存数据路径
        self.original_df = pd.read_csv(data_path)  # 保存原始DataFrame用于索引映射
        self.questions = self._load_data(data_path)
        self.concepts = self._load_concepts()
        self.prerequisites = self._load_prerequisites(prerequisites_path) if prerequisites_path else {}
        # 预加载题目索引映射
        self.question_index_mapping = self._build_question_index_mapping()
        # self.client = ollama.Client(host='http://0.0.0.0:11435')
    
    def _load_data(self, path: str) -> List[Dict]:
        """读取ASSIST数据集并预处理"""
        df = pd.read_csv(path)
        
        # print(f"原始数据列名: {df.columns.tolist()}")
        
        # 检查列名并重命名
        column_mapping = {}
        if 'problem_id' in df.columns:
            column_mapping['problem_id'] = 'name'
        if 'skill_id' in df.columns:
            column_mapping['skill_id'] = 'topic_id'
        if 'skill_name' in df.columns:
            column_mapping['skill_name'] = 'topic'
        if 'difficulty_score' in df.columns:
            column_mapping['difficulty_score'] = 'normalized_difficulty'
        
        df = df.rename(columns=column_mapping)
        # print(f"重命名后列名: {df.columns.tolist()}")
        
        # 如果normalized_difficulty列不存在，使用默认值
        if 'normalized_difficulty' not in df.columns:
            print("警告: 未找到难度分数列，使用默认值0.5")
            df['normalized_difficulty'] = 0.5
        else:
            # 数据清洗：处理数值格式
            df['normalized_difficulty'] = pd.to_numeric(
                df['normalized_difficulty'].astype(str).str.replace(' ', '').str.replace('@', '0'),
                errors='coerce'
            ).fillna(0.5)
        
        # 处理topic_id，确保是整数
        if 'topic_id' in df.columns:
            df['topic_id'] = pd.to_numeric(
                df['topic_id'].astype(str).str.replace('@', '0').str.replace(' ', ''),
                errors='coerce'
            ).fillna(0).astype(int)
        else:
            print("警告: 未找到skill_id列，使用默认topic_id")
            df['topic_id'] = 0
        
        # 确保name列存在
        if 'name' not in df.columns:
            if 'problem_id' in df.columns:
                df['name'] = df['problem_id'].astype(str)
            else:
                print("警告: 未找到problem_id列，使用索引作为name")
                df['name'] = df.index.astype(str)
        
        # 确保topic列存在
        if 'topic' not in df.columns:
            if 'skill_name' in df.columns:
                df['topic'] = df['skill_name']
            else:
                print("警告: 未找到skill_name列，使用默认topic")
                df['topic'] = 'Unknown Topic'
        
        return [
            {
                "name": str(row["name"]),
                "topic": row["topic"],
                "topic_id": int(row["topic_id"]),
                "difficulty": float(row["normalized_difficulty"])
            }
            for _, row in df.iterrows()
        ]
        
    def _build_question_index_mapping(self) -> Dict[str, int]:
        """构建题目名称到数据集行索引的映射"""
        mapping = {}
        for idx, row in self.original_df.iterrows():
            # 确定题目名称列
            if 'problem_id' in self.original_df.columns:
                question_name = str(row['problem_id'])
            elif 'name' in self.original_df.columns:
                question_name = str(row['name'])
            else:
                question_name = str(row.iloc[0])  # 使用第一列作为名称
            
            mapping[question_name] = idx  # 这里存储的是DataFrame的行索引
        
        # print(f"构建了 {len(mapping)} 个题目的索引映射")
        # print(f"示例映射: {list(mapping.items())[:5]}")  # 显示前5个映射
        return mapping
    
    def _load_prerequisites(self, path: str) -> Dict[int, List[int]]:
        """读取先修关系数据"""
        if not path:
            return {}
        
        try:
            df = pd.read_csv(path)
            prerequisites = {}
            
            for _, row in df.iterrows():
                prerequisite = int(row['prerequisite'])
                item = int(row['item'])
                
                if prerequisite not in prerequisites:
                    prerequisites[prerequisite] = []
                prerequisites[prerequisite].append(item)
            
            # print(f"加载的先修关系: {prerequisites}")
            return prerequisites
        except Exception as e:
            print(f"加载先修关系数据失败: {e}")
            return {}
    
    def _load_concepts(self) -> List[Dict]:
        """从问题数据中提取不重复的知识点信息"""
        unique_concepts = set()
        concepts_list = []
        
        for question in self.questions:
            if not all(key in question for key in ["topic", "topic_id"]):
                continue
            
            concept_key = (question["topic"], question["topic_id"])
            
            if concept_key not in unique_concepts:
                unique_concepts.add(concept_key)
                concepts_list.append({
                    "topic": question["topic"],
                    "topic_id": question["topic_id"]
                })
        
        # print(f"提取的知识点数量: {len(concepts_list)}")
        return concepts_list
    
    def recommend(
        self,
        mastery_levels: List[float],
        num_recommendations: int = 1,
    ) -> Union[Dict, List[Dict]]:
        """
        根据当前的知识点掌握状态，选出要进行学习的知识点
        """
        
        # 1. 构建Prompt
        prompt = f"""
        ### 任务：通过科学推荐实现所有知识点掌握度最大化 
        ### 核心逻辑：帮助学生高效提升所有知识点的掌握程度，实现最大化学习收益。
        ### 掌握度与知识点的对应规则：  
        mastery_levels 的索引 0-4 ，严格对应 topic_id 0-4  
        即 mastery_levels[0] → topic_id=0 ，mastery_levels[1] → topic_id=1 ，以此类推  
        ### 学生知识点掌握度（topic_id 0-4）: {mastery_levels}
        
        ### 候选知识点（topic_id 0-4 分别对应掌握度索引 0-4） ：
        {self._format_concept(self.concepts)}
        
        ### 先修关系参考：{self.prerequisites}
        ### 决策要求
        基于以下信息，请严格选出1个最有助于提升整体掌握度的知识点：
        1. 当前各知识点的掌握度数值
        2. 知识点之间的先修关系（重要：必须掌握先修知识点才能学习后续知识点）
        3. 提升该知识点对整体知识体系的贡献度
        
        ### 输出要求（严格JSON格式）:
        {{
            "recommendations": [
                {{
                    "topic": "知识点名称",
                    "topic_id": 对应ID,
                    "reason": "需同时说明：①当前掌握度分析 ②先修关系考虑 ③对整体目标的贡献，请使用中文回答"
                }}
            ]
        }}
        """
        
        # print("知识点推荐Prompt:")
        # print(prompt)
        
        # 4. 调用大模型
        response = ollama.chat(
            model='qwen2.5:7b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3}
        )
        
        # print("模型响应:")
        # print(response['message']['content'])
        
        # 5. 健壮解析
        return self._safe_parse_concept(response['message']['content'], mastery_levels)
    
    def recommend_ques(
        self,
        mastery_levels: List[float],
        topic_id: int,
        topic_name: str,
        question_history
    ) -> Union[Dict, List[Dict]]:
        """根据推荐的 topic_id 获取对应的题目"""
        # 从 self.questions 中筛选出 topic_id 对应的题目
        related_questions = [q for q in self.questions if q["topic_id"] == topic_id]
        
        # print(f"找到 {len(related_questions)} 道相关题目，topic_id: {topic_id}")
        
        # 安全的题目过滤逻辑
        filtered_questions = []
        for q in related_questions:
            question_index = self._get_question_index(q['name'])
            if question_index != -1 and question_index not in question_history:
                filtered_questions.append(q)
            else:
                # print(f"跳过题目: {q['name']}, 索引: {question_index}, 在历史中: {question_index in question_history}")
                pass
        
        # print(f"过滤后剩余 {len(filtered_questions)} 道题目")
        
        if not filtered_questions:
            print("警告: 没有可用的题目，使用兜底逻辑")
            # return self._fallback_to_best_difficulty_question(topic_id, mastery_levels)
            # 节省时间，修改该部分
            return {
                "recommended_question": {
                    "question_id": 99,
                    "name": "无适配题目",
                    "difficulty": 0.0,
                    "topic_id": topic_id,
                    "reason": "当前主题下没有可用题目"
                }
            }
        
        # 构建引导词 - 明确说明question_id是数据集索引
        prompt_ques = f"""
        ### 任务：基于已选知识点和学生掌握情况，从候选题目中推荐 1 道最适配的题目
        ### 重要说明：
        - question_id 指的是题目在原始数据集中的行索引（从0开始）
        - 例如：question_id: 0 表示数据集的第一行，question_id: 1 表示第二行
        - 请根据题目难度和学生掌握程度匹配来推荐
        
        ### 核心信息：
        1. 已选定知识点 topic_id 为 {topic_id} ({topic_name})
        2. 学生对该知识点的掌握程度: {mastery_levels[topic_id]:.3f}
        3. 历史已做题目索引: {question_history}
        
        ### 候选题目（共{len(filtered_questions)}道）：
        {self._format_questions(filtered_questions)}
        
        ### 决策要求：
        1. 必须排除历史题目：不能推荐索引在 {question_history} 中的题目
        2. 难度匹配：选择难度与学生掌握度最匹配的题目
        3. 学习效果：选择最能提升学生掌握度的题目
        
        ### 输出要求（严格 JSON 格式）：
        {{
            "recommended_question": {{
                "name": "题目名称",
                "difficulty": 题目难度值,
                "topic_id": {topic_id},
                "question_id": 题目在数据集中的索引,
                "reason": "解释推荐理由，请使用中文"  
            }}
        }}
        """
        
        # print("题目推荐Prompt:")
        # print(prompt_ques)
        
        # 调用大模型
        response = ollama.chat(
            model='qwen2.5:7b',
            messages=[{'role': 'user', 'content': prompt_ques}],
            options={'temperature': 0.3}
        )

        # print("模型响应:")
        # print(response['message']['content'])

        return self._safe_parse_question(response['message']['content'], topic_id, mastery_levels)
    
    def _get_question_index(self, question_name: str) -> int:
        """根据题目名称获取在原始数据集中的行索引"""
        return self.question_index_mapping.get(question_name, -1)
    
    def _format_questions(self, questions: List[Dict]) -> str:
        """格式化题目展示 - 明确显示数据集索引"""
        formatted = []
        for i, q in enumerate(questions):
            question_index = self._get_question_index(q['name'])
            formatted.append(
                f"索引{question_index}: 名称='{q['name']}', 难度={q['difficulty']:.3f}, topic_id={q['topic_id']}"
            )
        return "\n".join(formatted)
    
    def _format_concept(self, concepts: List[Dict]) -> str:
        """格式化知识点：突出topic_id、知识点名称"""
        return "\n".join([
            f"topic_id {c['topic_id']}: {c['topic']}" 
            for c in self.concepts
        ])
    
    def _safe_parse_concept(self, text: str, mastery_levels: List[float]) -> Dict:
        """知识点推荐解析：失败时 fallback 到真实最低掌握度知识点"""
        try:
            json_str = re.search(r'\{.*\}', text, re.DOTALL).group()
            result = json.loads(json_str)
            return result
        except Exception as e:
            print(f"知识点解析失败: {e}")
            # 解析全失败，直接选真实最低掌握度知识点
            mastery_levels = mastery_levels.tolist() if hasattr(mastery_levels, 'tolist') else mastery_levels
            min_mastery_idx = mastery_levels.index(min(mastery_levels))
            return self._fallback_to_min_mastery_concept(min_mastery_idx)
        
    def _fallback_to_min_mastery_concept(self, topic_id: int) -> Dict:
        """构造真实最低掌握度知识点的推荐结果"""
        concept = next((c for c in self.concepts if c["topic_id"] == topic_id), None)
        if not concept:
            # 如果没有找到匹配的知识点，选择第一个
            concept = self.concepts[0] if self.concepts else {"topic": "Unknown", "topic_id": 0}
        
        return {
            "recommendations": [{
                "topic": concept["topic"],
                "topic_id": concept["topic_id"],
                "reason": "解析异常，自动选真实掌握度最低的知识点"
            }]
        }
        
    def _safe_parse_question(self, text: str, topic_id: int, mastery_levels: List[float]) -> Dict:
        """增强版JSON解析"""
        try:
            # 1. 严格提取JSON内容
            json_match = re.search(r'```json\n?(\{.*?\})\n?```', text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            
            if not json_match:
                raise ValueError("未找到有效的JSON结构")
            
            json_str = json_match.group(1)
            
            # 2. 清理字符串
            cleaned_json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', json_str)
            
            # 3. 尝试解析
            result = json.loads(cleaned_json_str)
            
            # 4. 验证question_id是否有效
            recommended_question = result.get("recommended_question", {})
            question_id = recommended_question.get("question_id", -1)
            
            # 验证question_id是否在有效范围内
            if question_id < 0 or question_id >= len(self.original_df):
                print(f"警告: 模型推荐的question_id {question_id} 超出有效范围 [0, {len(self.original_df)-1}]")
                return self._fallback_to_best_difficulty_question(topic_id, mastery_levels)
            
            return result
        
        except Exception as e:
            print(f"题目解析失败: {e}")
            return self._fallback_to_best_difficulty_question(topic_id, mastery_levels)

    def _fallback_to_best_difficulty_question(self, topic_id: int, mastery_levels: List[float]) -> Dict:
        """兜底逻辑：选难度最接近当前知识点掌握度的题"""
        topic_questions = [q for q in self.questions if q["topic_id"] == topic_id]
        if not topic_questions:
            return {
                "recommended_question": {
                    "question_id": -1,
                    "name": "无适配题目",
                    "difficulty": 0.0,
                    "topic_id": topic_id,
                    "reason": "当前主题下没有可用题目"
                }
            }
        
        target_mastery = mastery_levels[topic_id]
        # 按难度差排序，选最接近的
        topic_questions.sort(key=lambda x: abs(x["difficulty"] - target_mastery))
        best_question = topic_questions[0]
        
        best_question_index = self._get_question_index(best_question["name"])
        
        return {
            "recommended_question": {
                "question_id": best_question_index,
                "name": best_question.get("name", "无名称题目"),
                "difficulty": best_question.get("difficulty", 0.0),
                "topic_id": topic_id,
                "reason": f"解析异常，自动选择难度最适配的题目（难度差: {abs(best_question['difficulty'] - target_mastery):.2f}）"
            }
        }

# 使用示例
if __name__ == "__main__":
    # 请替换为实际的文件路径
    recommender = RobustRecommender(
        data_path="/home/dell/workspace/lmf/llm_jsrl/data_assist09/envpath_data_new.csv",
        prerequisites_path="/home/dell/workspace/lmf/llm_jsrl/data_assist09/prerequisites.csv"
    )
    
    mastery_levels = [0.06, 0.045, 0.09, 0.059, 0.011]
    result = recommender.recommend(mastery_levels)
    print("知识点推荐结果：")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    concept_id = result["recommendations"][0]["topic_id"]
    concept_name = result["recommendations"][0]["topic"]
    question_his = [45, 29, 31, 92, 84, 28, 2, 80, 10, 44, 67, 8, 5, 89, 85, 37, 71, 68, 3, 27, 87, 72, 56, 18, 69, 13, 15, 48, 91, 70, 79, 74]
    questions = recommender.recommend_ques(mastery_levels, concept_id, concept_name, question_his)
    print("题目推荐结果：")
    print(json.dumps(questions, indent=2, ensure_ascii=False))
