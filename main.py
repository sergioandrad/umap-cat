# Titulo: PCA, t-SNE e UMAP
# Autor: Sergio Andrade
# Descricao: Exemplo simples de aplicacao do UMAP
# na reducao de um data mesh de um objeto 3D.

# Carregando libs
import pandas as pd
import seaborn as sb
import umap
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import sklearn.manifold as manifold

# Transformando o .obj num DataFrame do pandas
cat_df = pd.DataFrame()
cat = open('cat_mesh.obj')
it = 0
line_list = []
for line in cat:
    if line[0:2] == "v ":
        line_list.append(line.replace('\n', '').split(' ')[1:4])

cat_frame = pd.DataFrame(line_list, columns=['x', 'y', 'z'], dtype=float).sample(n=10000)
print(cat_frame)

# Gato 3D -> Gato 2D
cat_pca = decomposition.PCA(n_components=2).fit_transform(cat_frame)
cat_tsne = manifold.TSNE(n_components=2, perplexity=50).fit_transform(cat_frame)
cat_umap = umap.UMAP(n_neighbors=50, min_dist=.1, metric='euclidean').fit_transform(cat_frame)

# Visualizando o gato em 3D
sb.scatterplot(data=cat_frame, x="z", y="y")


sb.set(style="darkgrid")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs='z', ys='x', zs='y', data=cat_frame)

# Visualizando o Gato 2D
plt.scatter(cat_pca[:, 0], cat_pca[:, 1])
plt.scatter(cat_tsne[:, 0], cat_tsne[:, 1])
plt.scatter(cat_umap[:, 0], cat_umap[:, 1])

# Segmentando o gato
# 9 clusters: 4 patas, 2 orelhas, face, rabo e tronco
cat_means_pca = cluster.KMeans(n_clusters=9, random_state=0).fit(cat_pca)
cat_means_tsne = cluster.KMeans(n_clusters=9, random_state=0).fit(cat_pca)
cat_means_umap = cluster.KMeans(n_clusters=9, random_state=0).fit(cat_umap)

plt.subplot(1, 3, 1)
plt.scatter(x=cat_pca[:, 0], y=cat_pca[:, 1], c=cat_means_pca.labels_)
plt.title('PCA')

plt.subplot(1, 3, 2)
plt.scatter(x=cat_tsne[:, 0], y=cat_tsne[:, 1], c=cat_means_tsne.labels_)
plt.title('t-SNE')

plt.subplot(1, 3, 3)
plt.scatter(x=cat_umap[:, 0], y=cat_umap[:, 1], c=cat_means_umap.labels_)
plt.title('UMAP')
