from constants import *

def encode_data(data_path, vocab_path):
    with open(vocab_path) as f:
        vocab = dict([(word, word_ID + 2)
                      for word_ID, word in enumerate(f.read().splitlines())])
    with open(data_path) as f:
        documents = [(line.split('<fff>')[0], line.split('<fff>')[1], line.split('<fff>')[2])
                     for line in f.read().splitlines()]
        
    encoded_data = []
    current_label = -1

    for document in documents:
        label, doc_id, text = document
        if label != current_label:
            current_label = label
            print(current_label)

        words = text.split()[:MAX_DOC_LENGTH]
        sentence_length = len(words)
        encoded_text = []
        for word in words:
            if word in vocab:
                encoded_text.append(str(vocab[word]))
            else:
                encoded_text.append(str(unknown_ID))
        
        if len(words) < MAX_DOC_LENGTH:
            num_padding = MAX_DOC_LENGTH - len(words)
            for _ in range(num_padding):
                encoded_text.append(str(padding_ID))
        encoded_data.append(str(label) + '<fff>' + str(doc_id) + '<fff>' + str(sentence_length) 
                            + '<fff>' + ' '.join(encoded_text))

    dir_name = '/'.join(data_path.split('/')[:-1])
    file_name = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt'
    with open(dir_name + r'/' + file_name, 'w') as f:
        f.write('\n'.join(encoded_data))