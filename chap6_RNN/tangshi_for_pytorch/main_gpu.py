import collections
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim

import rnn_gpu as rnn

start_token = 'G'
end_token = 'E'


def process_poems1(file_name):
    poems = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError:
                print("error")
                pass

    poems = sorted(poems, key=lambda line: len(line))

    all_words = []
    for poem in poems:
        all_words += [word for word in poem]

    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec):
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y = row[1:]
            y.append(row[-1])
            y_data.append(y)
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training():
    poems_vector, word_to_int, _ = process_poems1('./poems.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print('finish loading data')
    print('training device:', device)

    batch_size = 100
    torch.manual_seed(5)

    if device.type == 'cuda':
        torch.cuda.manual_seed_all(5)

    word_embedding = rnn.word_embedding(vocab_length=len(word_to_int) + 1, embedding_dim=100)
    rnn_model = rnn.RNN_model(
        batch_sz=batch_size,
        vocab_len=len(word_to_int) + 1,
        word_embedding=word_embedding,
        embedding_dim=100,
        lstm_hidden_dim=128
    ).to(device)

    optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.01)
    loss_fun = torch.nn.NLLLoss()

    for epoch in range(30):
        batches_inputs, batches_outputs = generate_batch(batch_size, poems_vector)
        n_chunk = len(batches_inputs)

        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch]

            loss = torch.tensor(0.0, device=device)
            for index in range(batch_size):
                x_np = np.array(batch_x[index], dtype=np.int64)
                y_np = np.array(batch_y[index], dtype=np.int64)

                x = torch.from_numpy(np.expand_dims(x_np, axis=1)).long().to(device)
                y = torch.from_numpy(y_np).long().to(device)

                pre = rnn_model(x)
                loss = loss + loss_fun(pre, y)

                if index == 0:
                    _, pred_idx = torch.max(pre, dim=1)
                    print('prediction', pred_idx.detach().cpu().tolist())
                    print('b_y       ', y.detach().cpu().tolist())
                    print('*' * 30)

            loss = loss / batch_size
            print('epoch', epoch, 'batch number', batch, 'loss is:', loss.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 1)
            optimizer.step()

            if batch % 20 == 0:
                torch.save(rnn_model.state_dict(), './poem_generator_rnn_gpu')
                print('finish save model')


def to_word(predict, vocabs):  # 预测的结果转化成汉字
    sample = np.argmax(predict)

    if sample >= len(vocabs):
        sample = len(vocabs) - 1

    return vocabs[sample]


def pretty_print_poem(poem):  # 令打印的结果更工整
    shige=[]
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 1:
            print(s + '。')


def gen_poem(begin_word):
    # poems_vector, word_int_map, vocabularies = process_poems2('./tangshi.txt')  #  use the other dataset to train the network
    poems_vector, word_int_map, vocabularies = process_poems1('./poems.txt')
    word_embedding = rnn.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
    rnn_model = rnn.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,
                                   embedding_dim=100, lstm_hidden_dim=128)

    rnn_model.load_state_dict(torch.load('./poem_generator_rnn_gpu', weights_only=True))

    # 指定开始的字

    poem = begin_word
    word = begin_word
    while word != end_token:
        input = np.array([word_int_map[w] for w in poem],dtype= np.int64)
        input = Variable(torch.from_numpy(input))
        output = rnn_model(input, is_test=True)
        word = to_word(output.data.tolist()[-1], vocabularies)
        poem += word
        # print(word)
        # print(poem)
        if len(poem) > 30:
            break
    return poem

if __name__ == '__main__':
    #run_training()
    pretty_print_poem(gen_poem("日"))
    pretty_print_poem(gen_poem("红"))
    pretty_print_poem(gen_poem("山"))
    pretty_print_poem(gen_poem("夜"))
    pretty_print_poem(gen_poem("湖"))
    pretty_print_poem(gen_poem("湖"))
    pretty_print_poem(gen_poem("湖"))
    pretty_print_poem(gen_poem("君"))