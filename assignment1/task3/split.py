import random
import argparse


def read_data(file_path):
    """读取文件内容并返回每一行作为列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines[-1] = lines[-1] + "\n"
    return lines


def split_data(lines, train_ratio=0.8, val_ratio=0.1):
    """根据指定的比例划分数据集"""
    total = len(lines)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    
    train_data = lines[:train_size]          
    val_data = lines[train_size:train_size + val_size] 
    test_data = lines[train_size + val_size:]

    return train_data, val_data, test_data


def write_data(file_name, data):
    """将数据写入到指定文件"""
    with open(file_name, "w", encoding="utf-8") as file:
        file.writelines(data)


def main(file_path):
    
    lines = read_data(file_path)

    # 打乱数据
    random.seed(42)
    random.shuffle(lines)

    train_data, val_data, test_data = split_data(lines)

    write_data("train.txt", train_data)
    write_data("val.txt", val_data)
    write_data("test.txt", test_data)

    print(f"数据集已成功划分并写入文件：\n训练集：{len(train_data)} 条\n验证集：{len(val_data)} 条\n测试集：{len(test_data)} 条")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将文件内容划分为训练、验证和测试集")
    parser.add_argument("file_path", type=str, help="输入文件路径 (如: eng_jpn.txt)")
    args = parser.parse_args()
    main(args.file_path)
