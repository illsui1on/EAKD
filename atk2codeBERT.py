from dataset import CODECHEF
from modifier import TokenModifier, InsModifier
from modifier import get_batched_data_1
import json
import numpy
import random
import torch
import torch.nn as nn
import argparse
import pickle, gzip
import os, sys, time

from sklearn import metrics


class Attacker(object):

    def __init__(self, classifier):

        #self.txt2idx = dataset.get_txt2idx()
        #self.idx2txt = dataset.get_idx2txt()
        # self.tokenM = TokenModifier(classifier=classifier,
        #                             loss=torch.nn.CrossEntropyLoss(),
        #                             uids=None,
        #                             txt2idx=None,
        #                             idx2txt=None
        #                             )
        self.cl = classifier
       # self.data = dataset
       #self.syms = symtab

    def attack(self):
        n_succ = 0
        with open('attacker_test.json', 'r') as file:
            data = json.load(file)
        #batch = get_batched_data([x_raw], [y], self.txt2idx)
        #start_time = time.time()
            for item in data:
                #new_x_raw = item['new_x_raw']
                #batch= get_batched_data_1(new_x_raw)
                batch=item['new_x_raw']
                y=item['y']
                new_prob = self.cl.prob(batch)
                new_pred = torch.argmax(new_prob, dim=-1)
                if new_pred == y:
                    print("FAIL")
                   
                else:
                    print("SUCC!\t\t%d =>%d" % \
                          (y, new_pred))
                    n_succ += 1
        #total_time += time.time() - start_time
            print("\t succ rate = %.3f" \
                % (n_succ / (len(data) + 1)), flush=True)

        # old_prob = old_prob[y]
        # while iter < n_iter:
        #     keys = list(uids.keys())
        #     for k in keys:
        #         if iter >= n_iter:
        #             break
        #         if n_stop >= len(uids):
        #             iter = n_iter
        #             break
        #         if k in self.tokenM.forbidden_uid:
        #             n_stop += 1
        #             continue
        #
        #         # don't waste iteration on the "<unk>"s
        #         assert not k.startswith('Ġ')
        #         Gk = 'Ġ' + k
        #         Gk_idx = self.cl.tokenizer.convert_tokens_to_ids(Gk)
        #         if Gk_idx == self.cl.tokenizer.unk_token_id:
        #             continue
        #
        #         iter += 1
        #         new_x_raw, new_x_uid = self.tokenM.rename_uid(x_raw, y, k, n_candidate)
        #         if new_x_raw is None:
        #             n_stop += 1
        #             print("skip unk\t%s" % k)
        #             continue
        #         batch = get_batched_data(new_x_raw, [y] * len(new_x_raw), self.txt2idx)
        #         new_prob = self.cl.prob(batch['x'])
        #         new_pred = torch.argmax(new_prob, dim=-1)
        #         for uid, p, pr in zip(new_x_uid, new_pred, new_prob):
        #             if p != y:
        #                 print("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
        #                       (k, uid, y, old_prob, y, pr[y], p, pr[p]))
        #                 return True, p
        #
        #         new_prob_idx = torch.argmin(new_prob[:, y])
        #         if new_prob[new_prob_idx][y] < old_prob:
        #             x_raw = new_x_raw[new_prob_idx]
        #             uids[new_x_uid[new_prob_idx]] = uids.pop(k)
        #             n_stop = 0
        #             print("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
        #                   (k, new_x_uid[new_prob_idx], y, old_prob, y, new_prob[new_prob_idx][y]))
        #             old_prob = new_prob[new_prob_idx][y]
        #         else:
        #             n_stop += 1
        #             print("rej\t%s" % k)
        # print("FAIL!")
        # return False, y
        # file_attack.close()

    # def attack_all(self):
    #
    #     n_succ = 0
    #
    #
    #
    #     for i in range(self.d.test.get_size()):
    #         b = self.data.next_batch(1)
    #         # print("\t%d/%d\tID = %d\tY = %d" % (i + 1, self.d.test.get_size(), b['id'][0], b['y'][0]))
    #         # start_time = time.time()
    #         tag, pred = self.attack(b['raw'][0], b['y'][0], self.syms['te'][b['id'][0]])
    #         if tag == True:
    #             n_succ += 1




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default="0")
    parser.add_argument('--model_dir', type=str, default="../model_defect/codebert_60_1/18")
    parser.add_argument('--bs', type=int, default=16)

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    n_class = 4

    batch_size = opt.bs
    rand_seed = 1726

    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)

    # cc = CODECHEF(path=opt.data)
    # training_set = cc.train
    # valid_set = cc.dev
    # test_set = cc.test

    # import transformers after gpu selection
    from codebert import CodeBERTClassifier

    # with gzip.open('../data_defect/codechef_uid.pkl.gz', "rb") as f:
    #     symtab = pickle.load(f)

    # with gzip.open('../data_defect/codechef_inspos.pkl.gz', "rb") as f:
    #     instab = pickle.load(f)

    classifier = CodeBERTClassifier(model_path=opt.model_dir,
                                    num_labels=n_class,
                                    device=device).to(device)
    classifier.eval()

    atk = Attacker(classifier)
    atk.attack()
    # atk.attack_all(5, 40)