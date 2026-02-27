import os
import sys
from sentence_transformers import SentenceTransformer
import time
from app.services.summary.structure_analsis.python.ENRE_py.enre.__main__ import main as enre_main
from model.models import Function, method_Cluster
from utils.file_operations import create_directory_summary, add_functions_to_files
from utils.file_clustering import find_best_resolution, save_to_file_cluster
from utils.function_clustering import cluster_all_functions_to_features, set_func_adj_matrix
from utils.feature_generation import merge_features_by_method_cluster, features_to_csv, generate_feature_description_parallel, generate_feature_description
from utils.method_summary import method_summary


class TeeOutput:
    """同时将输出写入文件和控制台的类"""
    def __init__(self, file_path, mode='w', encoding='utf-8'):
        self.file = open(file_path, mode, encoding=encoding)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()  # 确保立即写入文件
        self.stdout.write(text)
        self.stdout.flush()
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        if self.file:
            self.file.close()
    
    def isatty(self):
        """返回False，表示这不是一个终端"""
        return False
    
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.close()



def summary_main(project_root: str, output_dir: str):
    start_time = time.time()

    project_root = os.path.abspath(project_root)

    enre_main([project_root, output_dir])

    # 生成函数描述,可选择用函数名(function_name)/CodeT5(code_t5)/LLM(llm)生成
    language = "python"
    functions = method_summary(output_dir, strategy="code_t5", language=language)

    # 生成文件描述，在这里固定使用文件名
    files = create_directory_summary(project_root)
    add_functions_to_files(files, functions, language=language)

    # 可选择是否并行，在服务器上运行不确定是否可行，建议关闭
    is_parallel = True

    # 打印一些文件和函数信息
    for file in files[:5]:
        print(f"File ID: {file.file_id}, Name: {file.file_name}, Path: {file.file_path}, Description: {file.file_desc}")
        for function in file.func_list[:5]:
            print(f"  Function ID: {function.func_id}, Name: {function.func_name}, Description: {function.func_desc}")
        print("\n")
    
    # 生成文本向量
    model = SentenceTransformer('all-mpnet-base-v2')
    for file in files:
        file.file_txt_vector = model.encode(file.file_desc).tolist()

    # 文件聚类
    best_gamma, best_labels, results = find_best_resolution(
        files,
        a=0.5,
        n_points=25,
        gamma_min=0.01, gamma_max=0.4,
        seeds_per_gamma=8,
        use_knn=True, knn_k=20,
        use_threshold=False, threshold_tau=0.0,
        min_clusters=3, max_clusters_ratio=0.15,
        min_cluster_size=3,
        use_silhouette=False,
    )

    clusters = save_to_file_cluster(files, best_labels)

    for c in clusters:
        print(f"Cluster ID: {c.cluster_id}, {len(c.cluster_file_list)} Files: {[file.file_name for file in c.cluster_file_list]}")

    # 将clusters展开到函数层
    method_clusters = []
    for cluster in clusters:
        func_list = []
        for file in cluster.cluster_file_list:
            for function in file.func_list:
                function.func_txt_vector = model.encode(function.func_desc).tolist()
                func_list.append(function)
        method_cluster = method_Cluster(cluster.cluster_id, "", func_list)
        method_clusters.append(method_cluster)

    for method_cluster in method_clusters:
        print(f"Cluster ID: {method_cluster.cluster_id}, Functions: {[f.func_fullName for f in method_cluster.cluster_func_list]}")

    # 函数聚类
    feature_list, summary = cluster_all_functions_to_features(
        method_clusters,
        weight_parameter=0.25,
        gamma_min=0.05, gamma_max=0.5, n_points=24, 
        seeds_per_gamma=8, # 每个gamma
        use_knn=True, knn_k=20, # 使用KNN稀疏化，指保留每个节点与其最近的k个节点的边
        use_threshold=False, threshold_tau=0.0, # 使用阈值稀疏化，指保留边权重大于阈值的边
        min_clusters=2, max_clusters_ratio=0.4, # 最小簇数和最大簇数比例
        use_silhouette=False, silhouette_sample_size=None, # 使用轮廓系数评估簇的分离度
        objective="CPM", # 目标函数，CPM表示聚类质量最大化
        consensus_tau=0.6, consensus_gamma=0.1, # 共识算法参数
        rng_seed=2025, # 随机种子
        target_total_features=None, # 目标总特征数
    )
    print(f"Total Features: {len(feature_list)}")
    for f in feature_list:
        print(f"Feature ID {f.feature_id}: {f.cluster_id} {set(x.func_file for x in f.feature_func_list)}")
        #print(f"Feature ID {f.feature_id}: {[function.func_fullName for function in f.feature_func_list]}")
    
    modelname = os.getenv("LLM2_API_MODEL")
    # 生成特征描述
    if is_parallel:
        generate_feature_description_parallel(feature_list, modelname=modelname, max_workers=8)
    else:
        generate_feature_description(feature_list, modelname=modelname)

    # 合并特征
    merge_features_by_method_cluster(feature_list, method_clusters, modelname=modelname)

    # 保存到CSV
    features_to_csv(feature_list, method_clusters, os.path.join(output_dir, "features.csv"))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    # project_root = "E:\\LoTM_Repos\\23\\original"
    project_root = "C:\\Users\\lixutian\\Desktop\\FeatX\\PyBackend\\app\\services\\summary"
    output_dir = os.path.join(here, "out")
    
    # 创建日志文件路径
    log_file = os.path.join(output_dir, "output_log.txt")
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用TeeOutput重定向所有输出到文件和控制台
    with TeeOutput(log_file, mode='w', encoding='utf-8'):
        summary_main(
            project_root=project_root,
            output_dir=output_dir
        ) 
    
    print(f"\n所有输出已保存到: {log_file}")
