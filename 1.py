from __future__ import print_function
import torch


from bert_serving.client import BertClient
import os, string, pickle


#path = "C:/Users/XCN/Desktop/conf_chi"
path = "/Users/xcn/Downloads/hci/conf_chi"



files = os.listdir(path)
s = []
print(files)
for file in files:
    if not os.path.isdir(file):
        f = open(path+'/'+file);
        iter_f = iter(f)
        for line in iter_f:
            line = line.strip()
            line = line.translate(str.maketrans("", "", string.punctuation))
            line = line.translate(str.maketrans("", "", string.digits))
            if line != '':
                s.append(line)
for i in range(10):
    print(s[i])
print(len(s))

#bc = BertClient()
bc = BertClient()
result = bc.encode(s[:2048])
print(result)
#save_file = 'C:/Users/XCN/Desktop/result.pk'
save_file = '/Users/xcn/Desktop/result.pk'
with open(save_file, 'wb') as f:
    pickle.dump([result, s[:2048]], f)
