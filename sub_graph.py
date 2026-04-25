# import pandas as pd
# import networkx as nx
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.decomposition import PCA

# # 1. 数据加载与去重
# def load_and_preprocess(filepath):
#     df = pd.read_csv(filepath)
    
#     # 按topic_id去重，保留每个知识点的第一条记录
#     unique_topics = df.drop_duplicates('topic_id').copy()
    
#     # 确保topic_id是连续的0-N索引
#     unique_topics['topic_id'] = range(len(unique_topics))
    
#     return unique_topics

# # # 使用示例
# # df = load_and_preprocess("data_junyi\envpath_data.csv")
# # print(f"去重后知识点数量: {len(df)}")
# # print(df[['topic_id', 'topic', 'area']].head())

# class JunyiKnowledgeGraph:
#     def __init__(self, dataframe, target_dim=64):
#         self.df = dataframe
#         self.num_topics = len(self.df)
#         self.topic_ids = list(range(self.num_topics))  # 0-N的连续索引
#         self.target_dim =  target_dim
        
#         # 构建图结构
#         self.G = self._build_graph()
        
#         # 文本嵌入模型
#         self.text_model = SentenceTransformer('/home/dell/workspace/lmf/llm_jsrl/all-MiniLM-L6-v2')
#         self.text_embeddings = self._compute_text_embeddings()
#         print(self.text_embeddings.shape)
        
#         # 图嵌入训练
#         self.global_embeddings = self._train_global_embeddings()
        
#         # # 降维
#         # self._reduce_dimension()

#     def _build_graph(self):
#         """构建同area连接的图结构"""
#         G = nx.Graph()
        
#         # 添加所有节点（使用数字索引）
#         G.add_nodes_from(range(self.num_topics))
        
#         # 建立同area连接
#         area_groups = self.df.groupby('area')['topic_id'].apply(list)
#         for _, topic_list in area_groups.items():
#             for i in range(len(topic_list)):
#                 for j in range(i+1, len(topic_list)):
#                     G.add_edge(topic_list[i], topic_list[j])
#         return G

#     def _compute_text_embeddings(self):
#         """生成知识点文本嵌入"""
#         texts = self.df['topic'] + " " + self.df['area']  # 组合topic和area作为文本
#         return np.array([self.text_model.encode(text) for text in texts])

#     def _train_global_embeddings(self):
#         """训练全图Node2Vec嵌入"""
#         from gensim.models import Word2Vec
        
#         def generate_walks():
#             walks = []
#             for _ in range(100):  # 每个节点100条游走
#                 for node in self.G.nodes():
#                     walk = [node]
#                     current = node
#                     for _ in range(30):  # 游走长度30
#                         neighbors = list(self.G.neighbors(current))
#                         if not neighbors:
#                             break
#                         current = np.random.choice(neighbors)
#                         walk.append(current)
#                     walks.append([str(n) for n in walk])
#             return walks
        
#         model = Word2Vec(
#             generate_walks(),
#             vector_size=384,
#             window=10,
#             min_count=1,
#             workers=4
#         )
        
#         # 返回与topic_id顺序对齐的嵌入矩阵
#         return np.array([model.wv[str(i)] for i in range(self.num_topics)])
    
#     def _reduce_dimension(self):
#         combined = np.hstack([self.global_embeddings, self.text_embeddings])
#         print(self.global_embeddings.shape, self.text_embeddings.shape, combined.shape)
        
#         self.pca = PCA(n_components=self.target_dim)
#         self.embeddings = self.pca.fit_transform(combined)
        
#         print(f"降维后保留方差: {sum(self.pca.explained_variance_ratio_):.2%}")

#     def get_dynamic_embeddings(self, mastery_array, max_nodes=3):
#         """
#         动态子图嵌入生成
#         :param mastery_array: 与self.topic_ids顺序一致的掌握度数组
#         :param max_nodes: 最大节点数
#         :return: (selected_indices, embeddings)
#         """
#         assert len(mastery_array) == self.num_topics, "掌握度数组长度不匹配"
        
#         # 1. 确定中心节点（掌握度最高）
#         center_idx = np.argmax(mastery_array)
        
#         # 2. 选择子图节点
#         selected = self._select_nodes(center_idx, max_nodes)
        
#         # 3. 生成调整后的嵌入
#         embeddings = self._adjust_embeddings(selected, mastery_array)
        
#         return selected, embeddings

#     def _select_nodes(self, center_idx, max_nodes):
#         """节点选择策略"""
#         selected = [center_idx]
        
#         # 优先选择图邻居
#         if center_idx in self.G:
#             neighbors = list(self.G.neighbors(center_idx))
#             selected += neighbors[:max_nodes-1]
        
#         # 不足时补充文本相似节点
#         if len(selected) < max_nodes:
#             similar = self._find_similar_nodes(center_idx, excluded=selected)
#             selected += similar[:max_nodes - len(selected)]
        
#         return selected[:max_nodes]

#     def _find_similar_nodes(self, target_idx, excluded=None, top_k=5):
#         """基于文本相似度的节点查找"""
#         if excluded is None:
#             excluded = []
            
#         target_emb = self.text_embeddings[target_idx]
        
#         # 计算所有非排除节点的相似度
#         sim_scores = []
#         for idx in range(self.num_topics):
#             if idx != target_idx and idx not in excluded:
#                 sim = cosine_similarity(
#                     [target_emb],
#                     [self.text_embeddings[idx]]
#                 )[0][0]
#                 sim_scores.append((idx, sim))
        
#         # 返回相似度最高的top_k个节点索引
#         return [idx for idx, _ in sorted(sim_scores, key=lambda x: -x[1])[:top_k]]

#     def _adjust_embeddings(self, indices, mastery_array):
#         """嵌入动态调整"""
#         adjusted = []
#         for idx in indices:
#             # 基础嵌入（图嵌入和文本嵌入的加权平均）
#             emb = 0.6 * self.global_embeddings[idx] + 0.4 * self.text_embeddings[idx]
#             # emb =self.embeddings[idx].copy()
            
#             # 掌握度调整（0.5-1.5倍缩放）
#             mastery = mastery_array[idx]
#             emb *= 0.5 + mastery
            
#             # 邻居影响（如果存在邻居）
#             if idx in self.G and self.G.degree(idx) > 0:
#                 nbr_embs = np.mean([
#                     self.global_embeddings[n] 
#                     for n in self.G.neighbors(idx)
#                 ], axis=0)
#                 emb = 0.7 * emb + 0.3 * nbr_embs
            
#             adjusted.append(emb)
#         return np.array(adjusted)

# # ==================== 使用示例 ====================
# if __name__ == "__main__":
#     # 1. 数据预处理
#     df = load_and_preprocess("/home/dell/workspace/lmf/llm_jsrl/data_junyi/envpath_data.csv")
    
#     # 2. 初始化知识图谱
#     kg = JunyiKnowledgeGraph(df)
    
#     # 3. 模拟掌握度数据（与topic_id顺序一致）
#     mock_mastery = np.random.rand(len(df))  # 随机生成0-1之间的掌握度
    
#     # 4. 获取动态嵌入
#     selected_indices, embeddings = kg.get_dynamic_embeddings(mock_mastery, max_nodes=3)
    
#     # 5. 结果解析
#     print("选中的节点索引:", selected_indices)
#     print("对应的知识点:")
#     for idx in selected_indices:
#         print(f"  {df.iloc[idx]['topic_id']}: {df.iloc[idx]['topic']} ({df.iloc[idx]['area']})")
#     print("嵌入矩阵形状:", embeddings.shape)
#     print(mock_mastery, embeddings)
    
#     # 4. 掌握度变化后的新子图
#     mock_mastery2 = np.random.rand(len(df))
#     selected2, embs2 = kg.get_dynamic_embeddings(mock_mastery2)
#     print(f"第二次选择: {selected2}, 嵌入形状: {embs2.shape}")
#     print(mock_mastery2, embs2)

# 改为bert模型，降低维度
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import os

# 1. 数据加载与预处理
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    unique_topics = df.drop_duplicates('topic_id').copy()
    unique_topics['topic_id'] = range(len(unique_topics))
    return unique_topics

class LocalBertEmbedder(nn.Module):
    """使用本地BERT模型的文本嵌入器"""
    def __init__(self, model_path, output_dim=64, device='cpu'):
        super().__init__()
        self.device = device
        
        # 从本地路径加载模型和分词器
        self.bert = BertModel.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.proj = nn.Linear(768, output_dim).to(device)
        
        # 冻结BERT参数
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.bert = self.bert.to(device)
        self.to(device)
        
    def forward(self, texts):
        # 批处理tokenization
        inputs = self.tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # 平均池化
            projected = self.proj(embeddings)  # 投影到目标维度
            
        return projected

class JunyiKnowledgeGraph:
    def __init__(self, dataframe, model_path, target_dim=64, device='cpu'):
        self.df = dataframe
        self.num_topics = len(self.df)
        self.topic_ids = list(range(self.num_topics))
        self.target_dim = target_dim
        self.device = device
        
        # 构建图结构
        self.G = self._build_graph()
        
        # 初始化本地BERT嵌入器
        self.text_model = LocalBertEmbedder(
            model_path=model_path,
            output_dim=target_dim,
            device=device
        )
        
        # 计算文本嵌入
        self.text_embeddings = self._compute_text_embeddings()
        
        # 训练图嵌入
        self.global_embeddings = self._train_global_embeddings()
        
        print(f"系统初始化完成 - 文本嵌入: {self.text_embeddings.shape}, 图嵌入: {self.global_embeddings.shape}")

    def _build_graph(self):
        """构建同area连接的图结构"""
        G = nx.Graph()
        G.add_nodes_from(range(self.num_topics))
        
        area_groups = self.df.groupby('area')['topic_id'].apply(list)
        for _, topic_list in area_groups.items():
            for i in range(len(topic_list)):
                for j in range(i+1, len(topic_list)):
                    G.add_edge(topic_list[i], topic_list[j])
        return G

    def _compute_text_embeddings(self):
        """生成知识点文本嵌入"""
        texts = (self.df['topic'] + " " + self.df['area']).tolist()
        
        # 分批处理避免内存溢出
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = self.text_model(batch)
            embeddings.append(emb.cpu())
        
        return torch.cat(embeddings, dim=0).numpy()

    def _train_global_embeddings(self):
        """训练全图Node2Vec嵌入"""
        def generate_walks():
            walks = []
            for _ in range(100):  # 每个节点100条游走
                for node in self.G.nodes():
                    walk = [node]
                    current = node
                    for _ in range(30):  # 游走长度30
                        neighbors = list(self.G.neighbors(current))
                        if not neighbors:
                            break
                        current = np.random.choice(neighbors)
                        walk.append(current)
                    walks.append([str(n) for n in walk])
            return walks
        
        model = Word2Vec(
            generate_walks(),
            vector_size=self.target_dim,
            window=10,
            min_count=1,
            workers=4
        )
        
        return np.array([model.wv[str(i)] for i in range(self.num_topics)])

    def get_dynamic_embeddings(self, mastery_array, max_nodes=3):
        """
        动态子图嵌入生成
        :param mastery_array: 与self.topic_ids顺序一致的掌握度数组
        :param max_nodes: 最大节点数
        :return: (selected_indices, embeddings)
        """
        assert len(mastery_array) == self.num_topics, "掌握度数组长度不匹配"
        
        # 1. 确定中心节点（掌握度最高）
        center_idx = np.argmax(mastery_array)
        
        # 2. 选择子图节点
        selected = self._select_nodes(center_idx, max_nodes)
        
        # 3. 生成调整后的嵌入
        embeddings = self._adjust_embeddings(selected, mastery_array)
        
        return selected, embeddings

    def _select_nodes(self, center_idx, max_nodes):
        """节点选择策略"""
        selected = [center_idx]
        
        # 优先选择图邻居
        if center_idx in self.G:
            neighbors = list(self.G.neighbors(center_idx))
            selected += neighbors[:max_nodes-1]
        
        # 不足时补充文本相似节点
        if len(selected) < max_nodes:
            similar = self._find_similar_nodes(center_idx, excluded=selected)
            selected += similar[:max_nodes - len(selected)]
        
        return selected[:max_nodes]

    def _find_similar_nodes(self, target_idx, excluded=None, top_k=5):
        """基于文本相似度的节点查找"""
        if excluded is None:
            excluded = []
            
        target_emb = self.text_embeddings[target_idx]
        
        # 计算所有非排除节点的相似度
        sim_scores = []
        for idx in range(self.num_topics):
            if idx != target_idx and idx not in excluded:
                sim = cosine_similarity(
                    [target_emb],
                    [self.text_embeddings[idx]]
                )[0][0]
                sim_scores.append((idx, sim))
        
        # 返回相似度最高的top_k个节点索引
        return [idx for idx, _ in sorted(sim_scores, key=lambda x: -x[1])[:top_k]]

    def _adjust_embeddings(self, indices, mastery_array):
        """嵌入动态调整"""
        adjusted = []
        for idx in indices:
            # 基础嵌入（图嵌入和文本嵌入的加权平均）
            emb = 0.6 * self.global_embeddings[idx] + 0.4 * self.text_embeddings[idx]
            
            # 掌握度调整（0.5-1.5倍缩放）
            mastery = mastery_array[idx]
            emb *= 0.5 + mastery
            
            # 邻居影响（如果存在邻居）
            if idx in self.G and self.G.degree(idx) > 0:
                nbr_embs = np.mean([
                    self.global_embeddings[n] 
                    for n in self.G.neighbors(idx)
                ], axis=0)
                emb = 0.7 * emb + 0.3 * nbr_embs
            
            adjusted.append(emb)
        return np.array(adjusted)

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置参数
    DATA_PATH = "/home/dell/workspace/lmf/llm_jsrl/data_junyi/envpath_data.csv"
    MODEL_PATH = "/home/dell/workspace/lmf/llm_jsrl/bert-base-uncased"  # 本地模型路径
    TARGET_DIM = 64  # 目标嵌入维度
    
    # 1. 数据预处理
    df = load_and_preprocess(DATA_PATH)
    
    # 2. 初始化知识图谱（使用本地模型）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用计算设备: {device}")
    
    kg = JunyiKnowledgeGraph(
        dataframe=df,
        model_path=MODEL_PATH,
        target_dim=TARGET_DIM,
        device=device
    )
    
    # 3. 模拟掌握度数据（与topic_id顺序一致）
    mock_mastery = np.random.rand(len(df))
    print(mock_mastery)
    
    # 4. 获取动态嵌入
    selected_indices, embeddings = kg.get_dynamic_embeddings(mock_mastery, max_nodes=3)
    
    # 5. 结果解析
    print("\n选中的节点索引:", selected_indices)
    print("对应的知识点:")
    for idx in selected_indices:
        print(f"  {df.iloc[idx]['topic_id']}: {df.iloc[idx]['topic']} ({df.iloc[idx]['area']})")
    print("动态嵌入矩阵形状:", embeddings.shape)
    # print(embeddings)
    # knowledge_rep = embeddings.flatten()
    # print(knowledge_rep.shape)
    # print(knowledge_rep)