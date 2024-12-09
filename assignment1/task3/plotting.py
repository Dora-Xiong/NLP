import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import numpy as np
import random
import matplotlib.font_manager as fm

def plot_embeddings_tsne(word_vectors, title, lang="en"):
    """将词嵌入使用t-SNE降维并绘制图像"""
    words = list(word_vectors.index_to_key)  
    selected_words = random.sample(words, 100)  
    embeddings = np.array([word_vectors[word] for word in selected_words])  

    # 使用 t-SNE 降维到 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 设置日语字体，如果是日语语言则使用日语字体
    if lang == "jp":
        font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"  # TODO: Change this path to your Japanese font path
        prop = fm.FontProperties(fname=font_path)
    else:
        prop = None

    # 绘制散点图
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='o')

    # 为每个点标注对应的词
    for i, word in enumerate(selected_words):
        plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]), fontproperties=prop)

    plt.title(title, fontproperties=prop)
    plt.show()

def main():
    # 加载保存的Word2Vec模型
    eng_word2vec = KeyedVectors.load("eng_word2vec.kv")
    jpn_word2vec = KeyedVectors.load("jpn_word2vec.kv")

    # 可视化英文词向量
    plot_embeddings_tsne(eng_word2vec, "English Word Embeddings Visualization", lang="en")

    # 可视化日文词向量
    plot_embeddings_tsne(jpn_word2vec, "Japanese Word Embeddings Visualization", lang="jp")

if __name__ == "__main__":
    main()
