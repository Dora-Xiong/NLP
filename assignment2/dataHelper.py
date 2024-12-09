from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
import json
from collections import defaultdict
import pandas as pd
import random

FEW_SHOT = 32

def get_dataset(dataset_name, sep_token):
	'''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''
	dataset = None
	if isinstance(dataset_name, str):
		dataset = load_single_dataset(dataset_name, sep_token)
	elif isinstance(dataset_name, list):
		dataset = aggregate_datasets(dataset_name, sep_token)
	else:
		raise ValueError('dataset_name should be a string or a list of strings')

	return dataset

def load_single_dataset(dataset_name, sep_token):
	'''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''
	if dataset_name in ["restaurant_sup", "laptop_sup"]:
		return load_absa_dataset(dataset_name, sep_token, few_shot=False)
	elif dataset_name == "acl_sup":
		return load_acl_dataset(sep_token, few_shot=False)
	elif dataset_name == "agnews_sup":
		return load_agnews_dataset(few_shot=False)
	elif dataset_name in ["restaurant_fs", "laptop_fs"]:
		return load_absa_dataset(dataset_name.split('_')[0] + '_sup', sep_token, few_shot=True)
	elif dataset_name == "acl_fs":
		return load_acl_dataset(sep_token, few_shot=True)
	elif dataset_name == "agnews_fs":
		return load_agnews_dataset(few_shot=True)


def load_agnews_dataset(few_shot=False):
	'''
	few_shot: bool, whether to load the few-shot version of the dataset
	'''
	dataset = load_dataset(
		"datasets/ag_news",
		split="test",
	)
	
	split_dataset = dataset.train_test_split(test_size=0.1, seed=2022)
	if few_shot:
		split_dataset['train'] = sample_few_shot(split_dataset['train'], FEW_SHOT)
 
	return DatasetDict({
		'train': split_dataset['train'],
		'test': split_dataset['test']
	})

def load_acl_dataset(sep_token, few_shot=False):
	'''
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	few_shot: bool, whether to load the few-shot version of the dataset
	'''
	data_dict = load_dataset(
		"json",
		data_files={
			"train": "datasets/acl_sup/train.jsonl",
			"test": "datasets/acl_sup/test.jsonl",
		}
	)
	train_texts = data_dict['train']['text']
	train_labels = data_dict['train']['label']
	test_texts = data_dict['test']['text']
	test_labels = data_dict['test']['label']
 
	# 构建标签到数字的映射
	unique_labels = list(set(train_labels + test_labels))
	label2id = {label: i for i, label in enumerate(unique_labels)}
	
	train_labels = [label2id[label] for label in train_labels]
	test_labels = [label2id[label] for label in test_labels]
 
	train = Dataset.from_dict({'text': train_texts, 'label': train_labels}, split='train')
	test = Dataset.from_dict({'text': test_texts, 'label': test_labels}, split='test')
	if few_shot:
		train = sample_few_shot(train, FEW_SHOT)
 
	return DatasetDict({
		'train': train,
		'test': test
	})
	
	
def load_absa_dataset(dataset_name, sep_token, few_shot=False):
	'''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	few_shot: bool, whether to load the few-shot version of the dataset
	'''
	name2path = {
		"restaurant_sup": "datasets/SemEval14-res",
		"laptop_sup": "datasets/SemEval14-laptop"
	}
	path = name2path[dataset_name]
	
	def read_file(file_path: str, split: str):
		polarity_map = {
			'positive': 2,
			'neutral': 1,
			'negative': 0,
		}
		with open(file_path, 'r') as f:
			raw_data = json.load(f)
		texts = []
		labels = []
		for entry in raw_data.values():
			text = f"{entry['term']} {sep_token} {entry['sentence']}"
			texts.append(text)
			labels.append(polarity_map[entry['polarity']])
		dataset = Dataset.from_dict({'text': texts, 'label': labels}, split=split)
		return dataset

	train = read_file(f"{path}/train.json", split='train')
	test = read_file(f"{path}/test.json", split='test')
	if few_shot:
		train = sample_few_shot(train, FEW_SHOT)
	return DatasetDict({'train': train, 'test': test})

def sample_few_shot(dataset: Dataset, num_samples: int):
	'''
	dataset: Dataset, the dataset to sample from
	num_samples: int, the number of samples to sample
	'''
	label2samples = defaultdict(list)
	for item in dataset:
		label2samples[item['label']].append(item)
	
	num_classes = len(label2samples)
	base_samples_per_class = num_samples // num_classes
	remainder = num_samples % num_classes
	
	few_shot_data = []
	for label, samples in label2samples.items():
		num_shots = base_samples_per_class + (1 if remainder > 0 else 0)
		remainder -= 1
		few_shot_data.extend(random.sample(samples, min(num_shots, len(samples))))
	
	return Dataset.from_dict({
		'text': [item['text'] for item in few_shot_data],
		'label': [item['label'] for item in few_shot_data]
	})

def aggregate_datasets(dataset_names, sep_token, seed=42):
	'''
	Aggregates multiple datasets into one, relabeling to avoid label conflicts.
	'''
	aggregated_train = []
	aggregated_test = []
	label_offset = 0  
	
	for name in dataset_names:
		data = load_single_dataset(name, sep_token)
		for split in ['train', 'test']:
			relabelled_data = relabel_dataset(data[split], label_offset)
			if split == 'train':
				aggregated_train.append(relabelled_data)
			else:
				aggregated_test.append(relabelled_data)
	
		label_offset += max(data['train']['label']) + 1
	return DatasetDict({
		'train': concatenate_datasets(aggregated_train).shuffle(seed=seed),
		'test': concatenate_datasets(aggregated_test).shuffle(seed=seed)
	})

def relabel_dataset(dataset, label_offset):
	'''
	Applies a label offset to avoid overlapping labels between datasets.
	'''
	new_labels = [label + label_offset for label in dataset['label']]
	return Dataset.from_dict({'text': dataset['text'], 'label': new_labels})