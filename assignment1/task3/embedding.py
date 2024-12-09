import sentencepiece as spm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
from collections import defaultdict
import random
import argparse

def load_data(file_path):
    """读取文件并将句子分成英文和日文两个列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.readlines()
    eng_sentences, jpn_sentences = [], []
    for line in data:
        jpn, eng = line.strip().split("\t")
        eng_sentences.append(eng)
        jpn_sentences.append(jpn)
    return eng_sentences, jpn_sentences

def train_bpe_model(sentences, prefix, vocab_size=8000):
    """根据给定的句子训练BPE模型，并保存为指定前缀的模型文件"""
    with open(f"{prefix}_sentences.txt", "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(f"{sentence}\n")

    spm.SentencePieceTrainer.Train(
        f"--input={prefix}_sentences.txt --model_prefix={prefix} --vocab_size={vocab_size} --character_coverage=1.0 --model_type=bpe"
    )

def load_bpe_model(model_file):
    """加载已训练的BPE模型"""
    return spm.SentencePieceProcessor(model_file=model_file)

def tokenize_sentences(sp_model, sentences):
    """对句子列表进行BPE分词"""
    return [sp_model.encode(sentence, out_type=str, add_bos=True, add_eos=True) for sentence in sentences]

def train_word2vec(tokenized_sentences, vector_size=100, window=2, min_count=1, sg=0):
    """训练Word2Vec模型并返回训练好的模型"""
    model = Word2Vec(sentences=tokenized_sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    return model

def save_word_vectors(model, file_name):
    """保存Word2Vec的词向量文件"""
    model.wv.save(file_name)

def main():
    parser = argparse.ArgumentParser(description="训练BPE和Word2Vec模型的脚本")
    parser.add_argument("file_path", type=str, help="输入的训练数据文件路径")
    args = parser.parse_args()

    file_path = args.file_path

    eng_sentences, jpn_sentences = load_data(file_path)

    # 训练BPE模型
    train_bpe_model(eng_sentences, "eng_bpe", vocab_size=8000)
    train_bpe_model(jpn_sentences, "jpn_bpe", vocab_size=8000)

    # 加载BPE模型
    sp_eng = load_bpe_model("eng_bpe.model")
    sp_jpn = load_bpe_model("jpn_bpe.model")

    # 对句子进行BPE分词
    eng_tokenized = tokenize_sentences(sp_eng, eng_sentences)
    jpn_tokenized = tokenize_sentences(sp_jpn, jpn_sentences)

    # 训练Word2Vec模型
    eng_word2vec = train_word2vec(eng_tokenized, vector_size=1024)
    jpn_word2vec = train_word2vec(jpn_tokenized, vector_size=1024)

    # 保存模型词向量
    save_word_vectors(eng_word2vec, "eng_word2vec.kv")
    save_word_vectors(jpn_word2vec, "jpn_word2vec.kv")

    print("BPE 模型和 Word2Vec 模型训练及保存成功。")


if __name__ == "__main__":
    main()