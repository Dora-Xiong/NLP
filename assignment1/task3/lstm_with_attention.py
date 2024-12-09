import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import argparse
from tqdm.auto import tqdm
from gensim.models import KeyedVectors
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torcheval.metrics.functional import perplexity
import time


# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Seq2Seq model with attention.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--units", type=int, default=50, help="Number of hidden units for LSTM layers")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--train_file", type=str, default="train.txt", help="Training file path")
    parser.add_argument("--val_file", type=str, default="val.txt", help="Validation file path")
    parser.add_argument("--test_file", type=str, default="test.txt", help="Testing file path")
    parser.add_argument("--eng_vocab", type=str, default="eng_bpe.vocab", help="English vocabulary file path")
    parser.add_argument("--jpn_vocab", type=str, default="jpn_bpe.vocab", help="Japanese vocabulary file path")
    parser.add_argument("--eng_kv", type=str, default="eng_word2vec.kv", help="Pre-trained English word vectors")
    parser.add_argument("--jpn_kv", type=str, default="jpn_word2vec.kv", help="Pre-trained Japanese word vectors")
    parser.add_argument("--eng_model", type=str, default="eng_bpe.model", help="Pre-trained SentencePiece model for English")
    parser.add_argument("--jpn_model", type=str, default="jpn_bpe.model", help="Pre-trained SentencePiece model for Japanese")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--test", action='store_true', help="Test the existing model given by --model_path")
    parser.add_argument("--encoder_path", type=str, help="The path to the encoder being tested.")
    parser.add_argument("--decoder_path", type=str, help="The path to the decoder being tested.")
    return parser.parse_args()

# 加载 BPE 语言索引
class BPELanguageIndex:
    def __init__(self, vocab_file):
        self.word2idx, self.idx2word = self.load_bpe_vocab(vocab_file)
        self.vocab = set(self.word2idx.keys())

    def load_bpe_vocab(self, vocab_file):
        word2idx = {}
        idx2word = {}
        word2idx["<pad>"] = 0
        idx2word[0] = "<pad>"

        with open(vocab_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                word, _ = line.strip().split("\t")
                word2idx[word] = idx
                idx2word[idx] = word
        return word2idx, idx2word

    def convert_words(self, sentence: list):
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in sentence]


# 创建嵌入矩阵
def create_embedding_matrix(word2idx, vectors, embedding_dim):
    vocab_size = len(word2idx)
    embedding_matrix = torch.zeros((vocab_size, embedding_dim))
    for word, idx in word2idx.items():
        if word == "<pad>":
            continue
        try:
            embedding_matrix[idx] = torch.tensor(vectors[word], dtype=torch.float32)
        except KeyError:
            embedding_matrix[idx] = torch.randn(embedding_dim)
    return embedding_matrix


# 定义encoder
class Encoder(nn.Module):
    def __init__(self, embedding_matrix, units):
        super(Encoder, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.rnn = nn.LSTM(embed_dim, units, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(2 * units, units)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed_x)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        outputs = self.fc(outputs)
        return outputs, (hidden, cell)


# 加性注意力层
class AdditiveAttention(nn.Module):
    def __init__(self, units):
        super(AdditiveAttention, self).__init__()
        self.W_q = nn.Linear(units, units, bias=False)
        self.W_k = nn.Linear(units, units, bias=False)
        self.W_v = nn.Linear(units, 1, bias=False)

    def forward(self, query, key, value, mask=None):
        query = self.W_q(query)
        key = self.W_k(key)
        score = self.W_v(torch.tanh(query.unsqueeze(2) + key.unsqueeze(1))).squeeze(-1)
        if mask is not None:
            score.masked_fill_(mask == 0, float('-inf'))
        attention_weights = torch.softmax(score, dim=-1)
        context = torch.bmm(attention_weights, value)
        return context, attention_weights


class Decoder(nn.Module):
    def __init__(self, embedding_matrix, units):
        super(Decoder, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.rnn = nn.LSTM(embed_dim, units, num_layers=4, batch_first=True)
        self.attention = AdditiveAttention(units)
        self.fc = nn.Linear(2 * units, vocab_size)

    def forward(self, x, enc_outputs, state, mask=None):
        x = self.embedding(x)
        output, (hidden, cell) = self.rnn(x, state)
        context, attention_weights = self.attention(query=output, key=enc_outputs, value=enc_outputs, mask=mask)
        context_output = torch.cat((context, output), dim=-1)
        pred = self.fc(context_output)
        return pred, (hidden, cell), attention_weights


# 数据集定义
class Seq2SeqDataset(Dataset):
    def __init__(self, file_path, inp_lang, tgt_lang, inp_model, tgt_model):
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = [line.strip().split("\t") for line in f.readlines()]
        self.inp_lang = inp_lang
        self.tgt_lang = tgt_lang
        self.inp_model = inp_model
        self.tgt_model = tgt_model
        self.inputs = [self.inp_lang.convert_words(self.inp_model.encode(line[0], out_type=str, add_bos=True, add_eos=True)) for line in self.data]
        self.targets = [self.tgt_lang.convert_words(self.tgt_model.encode(line[1], out_type=str, add_bos=True, add_eos=True)) for line in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# 数据处理
def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lengths = [len(inp) for inp in inputs]
    target_lengths = [len(tgt) for tgt in targets]
    inputs = pad_sequence([torch.tensor(inp) for inp in inputs], batch_first=True, padding_value=0)
    targets = pad_sequence([torch.tensor(tgt) for tgt in targets], batch_first=True, padding_value=0)
    return inputs, targets, torch.tensor(input_lengths), torch.tensor(target_lengths)

# 计算bleu score
def calculate_bleu_score(predicted_sentence, reference_sentence):
    reference_tokens = reference_sentence.split()
    predicted_tokens = predicted_sentence.split()
    smooth_fn = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], predicted_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)

# 预测函数
def predict_seq2seq(encoder, decoder, input_seq, tgt_lang, max_length_tgt=100):
    input_seq = torch.tensor(input_seq).unsqueeze(0)
    input_length = [len(input_seq[0])]
    inp_mask = input_seq != 0

    with torch.no_grad():
        enc_outputs, enc_state = encoder(input_seq, input_length)
        dec_input = torch.tensor([tgt_lang.word2idx["<s>"]]).unsqueeze(0)
        translation = ["<s>"]

        for t in range(max_length_tgt):
            dec_output, enc_state, _ = decoder(dec_input, enc_outputs, enc_state, inp_mask.unsqueeze(1))
            pred_word_idx = torch.argmax(dec_output.squeeze(1), dim=-1).item()
            translation.append(tgt_lang.idx2word[pred_word_idx])
            if pred_word_idx == tgt_lang.word2idx["</s>"]:
                break
            dec_input = torch.tensor([[pred_word_idx]])

        return translation


# 评估模型
def evaluate_model(encoder, decoder, data_loader, tgt_lang, tgt_model, max_length_tgt=100):
    total_bleu = 0
    total_loss = 0
    count = 0
    all_logits = []  
    all_targets = []
    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        for _, (x, y, input_lengths, target_lengths) in tqdm(enumerate(data_loader), total=len(data_loader)):
            inp_mask = x != 0
            max_length = max(input_lengths)

            batch_logits = torch.zeros((x.size(0), max_length_tgt - 1, len(tgt_lang.word2idx))).to(x.device)  
            batch_targets = torch.zeros((x.size(0), max_length_tgt - 1)).long().to(x.device)  
            
            for i in range(x.size(0)):  
                input_seq = x[i].tolist()
                reference_seq = y[i].tolist()
                
                predicted_list = predict_seq2seq(encoder, decoder, input_seq, tgt_lang, max_length_tgt=max_length_tgt)
                predicted_sentence = tgt_model.decode(predicted_list)
                reference_sentence = tgt_model.decode([tgt_lang.idx2word[idx] for idx in reference_seq if idx != 0])

                bleu_score = calculate_bleu_score(predicted_sentence, reference_sentence)
                total_bleu += bleu_score

                enc_outputs, enc_state = encoder(x[i].unsqueeze(0), [max_length])  
                dec_input = y[i, 0].unsqueeze(0).unsqueeze(1) 

                logits = []
                all_outputs = []
                all_truths = []

                for t in range(1, len(reference_seq)): 
                    dec_output, enc_state, _ = decoder(dec_input, enc_outputs, enc_state, inp_mask[i].unsqueeze(0).unsqueeze(1))
                    # print(dec_output.squeeze(1))
                    # print(y[i, t].unsqueeze(0))
                    # print(loss_fn(dec_output.squeeze(1), y[i, t].unsqueeze(0)).item())
                    all_outputs.append(dec_output.squeeze(1))
                    all_truths.append(y[i, t].unsqueeze(0))
                    dec_input = y[i, t].unsqueeze(0).unsqueeze(1)
                    logits.append(dec_output.squeeze(1)) 

                all_outputs = torch.cat(all_outputs, dim=0)
                all_truths = torch.cat(all_truths, dim=0)
                # print(loss_fn(all_outputs, all_truths).item())
                total_loss += loss_fn(all_outputs, all_truths).item()
                logits_tensor = torch.stack(logits, dim=0)

                batch_logits[i, :logits_tensor.size(0), :] = logits_tensor.squeeze(1)
                batch_targets[i, :len(reference_seq) - 1] = y[i, 1:len(reference_seq)] 

                count += 1

            all_logits.append(batch_logits)  
            all_targets.append(batch_targets)  
            
    all_logits = torch.cat(all_logits, dim=0) 
    all_targets = torch.cat(all_targets, dim=0)
    avg_ppl = perplexity(all_logits, all_targets, ignore_index=0).item()

    avg_bleu = total_bleu / count  
    avg_loss = total_loss / count

    return avg_loss, avg_bleu, avg_ppl


# 主训练流程
def train_model(encoder, decoder, train_loader, val_loader, num_epochs, save_dir, tgt_lang, tgt_model, learning_rate):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    
    history = {'training_loss': [], 'validation_loss': [], 'validation_bleu': [], 'validation_ppl': []}
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0
        start_time = time.time()
        count = 0
        for batch, (x, y, input_lengths, target_lengths) in tqdm(enumerate(train_loader), total=len(train_loader)):
            count += 1 * x.size(0)
            inp_mask = x != 0
            target = y != 0
            optimizer.zero_grad()
            enc_outputs, enc_state = encoder(x, input_lengths)
            dec_input = y[:, 0].unsqueeze(1)
            loss = 0
            for t in range(1, y.size(1)):
                dec_output, enc_state, _ = decoder(dec_input, enc_outputs, enc_state, inp_mask.unsqueeze(1))
                loss += loss_fn(dec_output.squeeze(1), y[:, t])
                dec_input = y[:, t].unsqueeze(1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        end_time = time.time()
        print(f"Epoch: {epoch + 1} | Time: {end_time - start_time:.2f}s")
        avg_loss = total_loss / count
        history['training_loss'].append(avg_loss)

        encoder.eval()
        decoder.eval()
        val_loss, avg_bleu, avg_ppl = evaluate_model(encoder, decoder, val_loader, tgt_lang, tgt_model)
        history['validation_loss'].append(val_loss)
        history['validation_bleu'].append(avg_bleu)
        history['validation_ppl'].append(avg_ppl)
        print(f'Epoch: {epoch + 1} | training loss: {avg_loss:.4f} | validation loss: {val_loss:.4f} | validation bleu: {avg_bleu:.4f} | validation ppl: {avg_ppl:.4f}')

        torch.save(encoder.state_dict(), f"{save_dir}/encoder_epoch{epoch + 1}.pth")
        torch.save(decoder.state_dict(), f"{save_dir}/decoder_epoch{epoch + 1}.pth")

    torch.save(encoder.state_dict(), "models/encoder_final.pth")
    torch.save(decoder.state_dict(), "models/decoder_final.pth")

    with open(f"{save_dir}/history.json", "w") as f:
        json.dump(history, f)
        
    return encoder, decoder

# 主函数入口
def main():
    args = parse_args()
    if not args.test:
        sp_eng = spm.SentencePieceProcessor(model_file=args.eng_model)
        sp_jpn = spm.SentencePieceProcessor(model_file=args.jpn_model)

        eng_lang = BPELanguageIndex(args.eng_vocab)
        jpn_lang = BPELanguageIndex(args.jpn_vocab)

        eng_vectors = KeyedVectors.load(args.eng_kv)
        jpn_vectors = KeyedVectors.load(args.jpn_kv)

        eng_embedding_matrix = create_embedding_matrix(eng_lang.word2idx, eng_vectors, eng_vectors.vector_size)
        jpn_embedding_matrix = create_embedding_matrix(jpn_lang.word2idx, jpn_vectors, eng_vectors.vector_size)

        train_data = Seq2SeqDataset(args.train_file, jpn_lang, eng_lang, sp_jpn, sp_eng)
        val_data = Seq2SeqDataset(args.val_file, jpn_lang, eng_lang, sp_jpn, sp_eng)
        test_data = Seq2SeqDataset(args.test_file, jpn_lang, eng_lang, sp_jpn, sp_eng)
        
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        encoder = Encoder(jpn_embedding_matrix, args.units)
        decoder = Decoder(eng_embedding_matrix, args.units)

        encoder, decoder = train_model(encoder, decoder, train_loader, val_loader, args.epochs, args.save_dir, eng_lang, sp_eng, args.learning_rate)

        # 测试集上测试
        encoder.eval()
        decoder.eval()
        test_loss, test_bleu, test_ppl = evaluate_model(encoder, decoder, test_loader, eng_lang, sp_eng)
        print(f"test loss: {test_loss:.4f}, test bleu: {test_bleu:.4f}, test ppl: {test_ppl:.4f}")
        with open(f"{args.save_dir}/test.txt", "w") as f:
            f.write(f"test loss: {test_loss:.4f}, test bleu: {test_bleu:.4f}, test ppl: {test_ppl:.4f}")
        
        case_1 = "私の名前は愛です"
        case_2 = "昨日はお肉を食べません"
        case_3 = "いただきますよう"
        case_4 = "秋は好きです"
        case_5 = "おはようございます"

        cases = [case_1, case_2, case_3, case_4, case_5]
        with torch.no_grad():
            for case in cases:
                input_seq = jpn_lang.convert_words(sp_jpn.encode(case, out_type=str, add_bos=True, add_eos=True))
                predicted_list = predict_seq2seq(encoder, decoder, input_seq, eng_lang)
                predicted_sentence = sp_eng.decode(predicted_list)
                print(f"Input: {case} | Predicted: {predicted_sentence}") 
    else:
        
        sp_eng = spm.SentencePieceProcessor(model_file=args.eng_model)
        sp_jpn = spm.SentencePieceProcessor(model_file=args.jpn_model)

        eng_lang = BPELanguageIndex(args.eng_vocab)
        jpn_lang = BPELanguageIndex(args.jpn_vocab)

        eng_vectors = KeyedVectors.load(args.eng_kv)
        jpn_vectors = KeyedVectors.load(args.jpn_kv)
        
        eng_embedding_matrix = create_embedding_matrix(eng_lang.word2idx, eng_vectors, eng_vectors.vector_size)
        jpn_embedding_matrix = create_embedding_matrix(jpn_lang.word2idx, jpn_vectors, eng_vectors.vector_size)

        encoder = Encoder(jpn_embedding_matrix, args.units)
        encoder.load_state_dict(torch.load(args.encoder_path))
        decoder = Decoder(eng_embedding_matrix, args.units)
        decoder.load_state_dict(torch.load(args.decoder_path))
        
        test_data = Seq2SeqDataset(args.test_file, jpn_lang, eng_lang, sp_jpn, sp_eng)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        encoder.eval()
        decoder.eval()
        
        test_loss, test_bleu, test_ppl = evaluate_model(encoder, decoder, test_loader, eng_lang, sp_eng)
        print(f"test loss: {test_loss:.4f}, test bleu: {test_bleu:.4f}, test ppl: {test_ppl:.4f}")
        with open(f"{args.save_dir}/test.txt", "w") as f:
            f.write(f"test loss: {test_loss:.4f}, test bleu: {test_bleu:.4f}, test ppl: {test_ppl:.4f}")
        
        case_1 = "私の名前は愛です"
        case_2 = "昨日はお肉を食べません"
        case_3 = "いただきますよう"
        case_4 = "秋は好きです"
        case_5 = "おはようございます"
        
        cases = [case_1, case_2, case_3, case_4, case_5]
        with torch.no_grad():
            for case in cases:
                input_seq = jpn_lang.convert_words(sp_jpn.encode(case, out_type=str, add_bos=True, add_eos=True))
                predicted_list = predict_seq2seq(encoder, decoder, input_seq, eng_lang)
                predicted_sentence = sp_eng.decode(predicted_list)
                print(f"Input: {case} | Predicted: {predicted_sentence}") 
    
if __name__ == "__main__":
    main()
