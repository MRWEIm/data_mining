import torch
import utils
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

dev_set = utils.load_data('./BEA/data/mrbench_v3_devset.json')
task_list = ['Mistake_Identification', 'Mistake_Location', 'Providing_Guidance', 'Actionability', 'Tutor_Identity', 'First_Four', 'All']
_, _, labels = utils.data_process(dev_set, task_type=task_list[5])
labels = utils.label_convert(labels)
embeddings = torch.load(f'./BEA/tensor/Qwen3-Embedding-4B_response_tensor.pt').cpu()
embeddings = embeddings.squeeze(1)

# 转为 numpy 以供 sklearn 使用
embeddings_np = embeddings.numpy()
labels_np = labels[:, 3].numpy()
print(embeddings_np.shape, labels_np.shape)

# Step 1: 聚类成3类
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_ids = kmeans.fit_predict(embeddings_np)

# Step 2: t-SNE 降维到2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_np)

# Step 3: 可视化，颜色基于原始 labels
plt.figure(figsize=(8, 6))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels_np, cmap='Set1', alpha=0.7)
plt.title("t-SNE of clustered embeddings (colored by true labels)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar(scatter, label='True Label')
plt.grid(False)
plt.tight_layout()
plt.savefig("./BEA/tsne_clustered_embeddings.png")
