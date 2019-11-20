import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from bert_model import Bert_Similarity, Bert_Embedding
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import loader, utils
import io
import os



# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
USE_CUDA = torch.cuda.is_available()
gpus = [1]
torch.cuda.set_device(gpus[0])
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


class Paras(object):
    def __init__(self, server_flag=1):
        self.bert_path = '/root/xcn/bert_pytorch'
        self.bert_maxlen = 50
        self.data_file = '/root/xcn/file4lab/paraphrase/data/para-nmt-5m-processed.txt'  # 训练集数据
        self.evaluate_file = '/root/xcn/file4lab/paraphrase/data/annotated-ppdb-dev'  # 验证集数据
        self.save_model_path = 'model/best_single_model2.plk'
        self.batch_size = 8 * len(gpus)
        self.epoch = 5
        self.model_type = 'avg'  # 句子矩阵到句子向量的方式，'avg'表示直接求平均， 'lstm'表示lstm加权平均
        self.margin = 1.  # 论文参数





        self.lr = 1e-5  # Learning Rate
        self.warmup_proportion = 0.1
        self.max_sentence = 30000  # 一个奇怪的问题导致的，不能一次性把所有向量写入文件

        self.lasagne_model_path = '../../data/bilstmavg-4096-40.pickle'  # Wieting论文提供的词向量

        if server_flag == 2:
            self.bert_path = '/home/cxs/xcn/xcnfile/bert_torch'



class Evaluator:
    def __init__(self, model, tokenizer, para):
        self.best_p_score = 0.0
        self.my_model = model.eval()  # 不加 eval()的话每次输出的句子向量都不一样（受模型中一些随机层的影响)
        self.tokenizer = tokenizer
        self.para = para

    # get_embedding 得到的是可以直接计算句子相似度的向量
    # get_embedding_list 得到的是sequence_len长度的向量列表, 相当于未经处理的句子向量

    def get_embedding(self, text1):
        x1token = ['[CLS]'] + self.tokenizer.tokenize(text1) + ['[SEP]']
        x1mask = [1] * len(x1token)
        x1mask = FloatTensor(utils.seq_padding(np.array([x1mask]), self.para.bert_maxlen))
        x1ids = self.tokenizer.convert_tokens_to_ids(x1token)
        x1ids = LongTensor(utils.seq_padding(np.array([x1ids]), self.para.bert_maxlen))
        x1 = self.my_model.bert_embedding([x1ids, x1mask])
        x1 = x1 / torch.sqrt(torch.sum(x1 * x1, -1, keepdim=True))
        print('x1 shape: ', np.shape(x1[0]))
        return x1[0].cpu().detach().numpy()

    def get_embedding_list(self, text1):
        x1token = ['[CLS]'] + self.tokenizer.tokenize(text1) + ['[SEP]']
        x1mask = [1] * len(x1token)
        x1mask = FloatTensor(utils.seq_padding(np.array([x1mask]), self.para.bert_maxlen))
        x1ids = self.tokenizer.convert_tokens_to_ids(x1token)
        x1ids = LongTensor(utils.seq_padding(np.array([x1ids]), self.para.bert_maxlen))
        x1 = self.my_model.bert_embedding.bert_embedding_model(x1ids, attention_mask=x1mask)
        x1 = x1[0][-2:]
        x1 = torch.cat(x1, -1)
        x1mask = x1mask.view(-1, self.para.bert_maxlen, 1)
        x1mask = x1mask.expand(-1, -1, 2048)
        x1 = x1 * x1mask  # 将mask位置的向量置零
        return x1[0, :len(x1token), :].cpu().detach().numpy()

    def get_batch_embedding(self, text_batch):
        X1ids = []
        X1mask = []
        for text1 in text_batch:
            x1token = ['[CLS]'] + self.tokenizer.tokenize(text1) + ['[SEP]']
            x1mask = [1] * len(x1token)
            x1ids = self.tokenizer.convert_tokens_to_ids(x1token)
            X1ids.append(x1ids)
            X1mask.append(x1mask)
        X1ids = LongTensor(utils.seq_padding(np.array(X1ids), self.para.bert_maxlen))
        X1mask = FloatTensor(utils.seq_padding(np.array(X1mask), self.para.bert_maxlen))
        X1 = self.my_model.bert_embedding([X1ids, X1mask])
        X1 = X1 / torch.sqrt(torch.sum(X1 * X1, -1, keepdim=True))
        # print('X1 shape: ', np.shape(X1))
        return X1


    def get_sim(self, text1, text2):
        x1token = ['[CLS]'] + tokenizer.tokenize(text1) + ['[SEP]']
        x2token = ['[CLS]'] + tokenizer.tokenize(text2) + ['[SEP]']

        x1mask = [1] * len(x1token)
        x2mask = [1] * len(x2token)
        x1mask = FloatTensor(utils.seq_padding(np.array([x1mask]), para.bert_maxlen))
        x2mask = FloatTensor(utils.seq_padding(np.array([x2mask]), para.bert_maxlen))
        x1ids = tokenizer.convert_tokens_to_ids(x1token)
        x2ids = tokenizer.convert_tokens_to_ids(x2token)
        x1ids = LongTensor(utils.seq_padding(np.array([x1ids]), para.bert_maxlen))
        x2ids = LongTensor(utils.seq_padding(np.array([x2ids]), para.bert_maxlen))
        x1 = self.my_model.bert_embedding([x1ids, x1mask])
        x2 = self.my_model.bert_embedding([x2ids, x2mask])

        x1 = x1 / torch.sqrt(torch.sum(x1 * x1, -1, keepdim=True))
        x2 = x2 / torch.sqrt(torch.sum(x2 * x2, -1, keepdim=True))
        x1x2 = torch.sum(x1 * x2, -1)
        return x1x2[0].item()



    def evaluate(self):
        f = io.open(self.para.evaluate_file, 'r', encoding='utf-8')
        lines = f.readlines()
        preds = []
        golds = []

        for i in lines:
            i = i.split("\t")
            p1 = i[0]
            p2 = i[1]
            score = float(i[2])
            preds.append(self.get_sim(p1, p2))
            golds.append(score)
        return pearsonr(preds, golds)[0], spearmanr(preds, golds)[0]




if __name__ == '__main__':
    para = Paras()
    tokenizer = BertTokenizer.from_pretrained(para.bert_path)
    train_loader = DataLoader(loader.TrainSet(para), batch_size=para.batch_size)
    text1 = 'However, it is too late'
    text2 = 'hello i\'m cat'
    text_ids = LongTensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text1))])
    text_mask = FloatTensor([[1] * len(text_ids)])
    model = Bert_Similarity(para).cuda()
    # model = torch.nn.DataParallel(model)
    # optimizer = torch.nn.DataParallel(optim.SGD(model.parameters(), lr=para.lr))
    # optimizer = optim.SGD(model.parameters(), lr=para.lr)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_steps = len(train_loader) // para.batch_size * para.epoch
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=para.lr,
    #                      warmup=para.warmup_proportion,
    #                      t_total=num_train_steps)
    optimizer = optim.SGD(model.parameters(), lr=para.lr)
    evaluator = Evaluator(model, tokenizer, para)

    print(utils.get_parameter_number(model))

    # result = model([text_ids, text_mask])
    # print(result)
    # print(np.shape(result))
    print(evaluator.get_embedding(text1))

    for epoch in range(para.epoch):
        loss_list = []
        for step, data in tqdm(enumerate(train_loader)):
            X1, X2 = data
            X1ids, X2ids, X1mask, X2mask = [], [], [], []
            for i in range(len(X1)):
                x1token = ['[CLS]'] + tokenizer.tokenize(X1[i]) + ['[SEP]']
                x2token = ['[CLS]'] + tokenizer.tokenize(X2[i]) + ['[SEP]']
                X1ids.append(tokenizer.convert_tokens_to_ids(x1token))
                X2ids.append(tokenizer.convert_tokens_to_ids(x2token))
                X1mask.append([1] * len(x1token))
                X2mask.append([1] * len(x2token))
            X1ids = LongTensor(utils.seq_padding(np.array(X1ids), para.bert_maxlen))
            X2ids = LongTensor(utils.seq_padding(np.array(X2ids), para.bert_maxlen))
            X1mask = FloatTensor(utils.seq_padding(np.array(X1mask), para.bert_maxlen))
            X2mask = FloatTensor(utils.seq_padding(np.array(X2mask), para.bert_maxlen))

            # 使用dataParallel之后下面两行需要加module()
            X1embed = model.bert_embedding([X1ids, X1mask])
            X2embed = model.bert_embedding([X2ids, X2mask])

            T1embed, T2embed = \
                utils.get_pairs([X1embed.cpu().data.numpy(), X2embed.cpu().data.numpy()])

            T1embed = FloatTensor(T1embed)
            T2embed = FloatTensor(T2embed)


            model.zero_grad()
            _, loss = model([X1embed, X2embed, T1embed, T2embed], is_training=True)
            # print('loss: ', loss.sum())
            loss_list.append(loss.sum().cpu().data.numpy())
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print("\nepoch[%d/%d] mean_loss : %0.4f" % (epoch + 1, para.epoch, np.mean(loss_list)))
                loss_list = []
            if (step > 0 and step % 2000 == 0) or step == len(train_loader) - 1:
                p_score, _ = evaluator.evaluate()
                print('p_socre: {}, best_p_score: {}'.format(p_score, evaluator.best_p_score))
                if evaluator.best_p_score < p_score:
                    evaluator.best_p_score = p_score
                    torch.save(model.state_dict(), 'model/best_cls_model.plk')  # 这是一个多卡模型
                if step == len(train_loader) - 1:
                    torch.save(model.state_dict(), 'model/best_cls_model_epoch{}_end.plk'.format(epoch))

