
from dataset import CODECHEF
from modifier import TokenModifier, InsModifier
from modifier import get_batched_data
import json
import numpy
import random
import torch
import torch.nn as nn
import argparse
import pickle, gzip
import os, sys, time

from sklearn import metrics
file_attack=open('attacker_base.json', 'w')
atk=[]
result = []
class Attacker(object):

    def __init__(self, dataset, symtab, classifier):
        
        self.txt2idx = dataset.get_txt2idx()
        self.idx2txt = dataset.get_idx2txt()
        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=self.txt2idx,
                                    idx2txt=self.idx2txt)
        self.cl = classifier
        self.d = dataset
        self.syms = symtab
    
    def attack(self, x_raw, y, uids, n_candidate=100, n_iter=20):

        iter = 0
        n_stop = 0

        batch = get_batched_data([x_raw], [y], self.txt2idx)
        old_prob = self.cl.prob(batch['x'])[0]
        old_pred= torch.argmax(old_prob, dim=-1)
        if old_pred != y:
            print ("SUCC! Original mistake.")
            row = {
                'new_x_raw': batch['x'][0],
                'y': int(y)
            }
            result.append(row)    
            return False, torch.argmax(old_prob).cpu().numpy()

        old_prob = old_prob[y]
        while iter < n_iter:
            keys = list(uids.keys())
            for k in keys:
                if iter >= n_iter:
                    break
                if n_stop >= len(uids):
                    iter = n_iter
                    break
                if k in self.tokenM.forbidden_uid:
                    n_stop += 1
                    continue

                # don't waste iteration on the "<unk>"s
                assert not k.startswith('Ġ')
                Gk = 'Ġ' + k
                Gk_idx = self.cl.tokenizer.convert_tokens_to_ids(Gk)
                if Gk_idx == self.cl.tokenizer.unk_token_id:
                    continue

                iter += 1
                new_x_raw, new_x_uid = self.tokenM.rename_uid(x_raw, y, k, n_candidate)
                if new_x_raw is None:
                    n_stop += 1
                    print ("skip unk\t%s" % k)
                    continue
                batch = get_batched_data(new_x_raw, [y]*len(new_x_raw), self.txt2idx)
                new_prob = self.cl.prob(batch['x'])
                new_pred = torch.argmax(new_prob, dim=-1)
                index = -1
                ss=0
                for uid, p, pr in zip(new_x_uid, new_pred, new_prob):
                    index+=1
                    if p != y:
                        print ("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                               (k, uid, y, old_prob, y, pr[y], p, pr[p]))
                        row = {
                            'new_x_raw':batch['x'][index],
                            'y': int(y)
                        }
                        result.append(row)
                        ss+=1

                if ss!=0:
                    return True, p




                new_prob_idx = torch.argmin(new_prob[:, y])
                if new_prob[new_prob_idx][y] < old_prob:
                    x_raw = new_x_raw[new_prob_idx]
                    uids[new_x_uid[new_prob_idx]] = uids.pop(k)
                    n_stop = 0
                    print ("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                           (k, new_x_uid[new_prob_idx], y, old_prob, y, new_prob[new_prob_idx][y]))
                    old_prob = new_prob[new_prob_idx][y]
                else:
                    n_stop += 1
                    print ("rej\t%s" % k)
        print ("FAIL!")
        return False, y


    def attack_all(self, n_candidate=100, n_iter=20):
        
        n_succ = 0
        total_time = 0
        trues = []
        preds = []

        st_time = time.time()
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, pred = self.attack(b['raw'][0], b['y'][0], self.syms['te'][b['id'][0]], n_candidate, n_iter)
            if tag==True:
                n_succ += 1
                total_time += time.time() - start_time
            preds.append(int(pred))
            trues.append(int(b['y'][0]))
            if n_succ <= 0:
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                       % (n_succ/(i+1), total_time/n_succ), flush=True)
            precision = metrics.precision_score(trues, preds, average='macro')
            recall = metrics.recall_score(trues, preds, average='macro')
            f1 = metrics.f1_score(trues, preds, average='macro')
            print("\t(P, R, F1) = (%.3f, %.3f, %.3f)" % (precision, recall, f1))
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/self.d.test.get_size()))

class InsAttacker(object):
    
    def __init__(self, dataset, instab, classifier):
        
        self.txt2idx = dataset.get_txt2idx()
        self.idx2txt = dataset.get_idx2txt()
        self.insM = InsModifier(classifier=classifier,
                                txt2idx=self.txt2idx,
                                idx2txt=self.idx2txt,
                                poses=None) # wait to init when attack
        self.cl = classifier
        self.d = dataset
        self.inss = instab
    
    # only support single x: a token-idx list
    def attack(self, x_raw, y, poses, n_candidate=100, n_iter=20):
        
        self.insM.initInsertDict(poses)

        iter = 0
        n_stop = 0

        batch = get_batched_data([x_raw], [y], self.txt2idx)
        old_prob = self.cl.prob(batch['x'])[0]
        if torch.argmax(old_prob) != y:
            print ("SUCC! Original mistake.")
            return True, torch.argmax(old_prob).cpu().numpy()
        old_prob = old_prob[y]
        
        while iter < n_iter:
            iter += 1

            # get insertion candidates
            n_could_del = self.insM.insertDict["count"]
            n_candidate_del = n_could_del
            n_candidate_ins = n_candidate - n_candidate_del
            assert n_candidate_del >= 0 and n_candidate_ins >= 0
            new_x_raw_del, new_insertDict_del = self.insM.remove(x_raw, n_candidate_del)
            new_x_raw_add, new_insertDict_add = self.insM.insert(x_raw, n_candidate_ins)
            new_x_raw = new_x_raw_del + new_x_raw_add
            new_insertDict = new_insertDict_del + new_insertDict_add
            if new_x_raw == []: # no valid candidates
                n_stop += 1
                continue

            # find if there is any candidate successful wrong classfied
            batch = get_batched_data(new_x_raw, [y]*len(new_x_raw), self.txt2idx) 
            new_prob = self.cl.prob(batch['x'])
            new_pred = torch.argmax(new_prob, dim=-1)
            for insD, p, pr in zip(new_insertDict, new_pred, new_prob):
                if p != y:
                    print ("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                            (self.insM.insertDict["count"], insD["count"], 
                                y, old_prob, y, pr[y], p, pr[p]))
                    return True, p

            # if not, get the one with the lowest target_label_loss
            new_prob_idx = torch.argmin(new_prob[:, y])
            if new_prob[new_prob_idx][y] < old_prob:
                print ("acc\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f)" % \
                        (self.insM.insertDict["count"], new_insertDict[new_prob_idx]["count"], 
                        y, old_prob, y, new_prob[new_prob_idx][y]))
                self.insM.insertDict = new_insertDict[new_prob_idx] # don't forget this step
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y]
            else:
                n_stop += 1
                print ("rej\t%s" % "")
            if n_stop >= len(new_x_raw):    # len(new_x) could be smaller than n_candidate
                iter = n_iter
                break
        print ("FAIL!")
        return False, y
    
    def attack_all(self, n_candidate=100, n_iter=20):

        n_succ = 0
        total_time = 0
        trues = []
        preds = []

        st_time = time.time()
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, pred = self.attack(b['raw'][0], b['y'][0], self.inss['stmt_te'][b['id'][0]], n_candidate, n_iter)
            if tag==True:
                n_succ += 1
                total_time += time.time() - start_time
            preds.append(int(pred))
            trues.append(int(b['y'][0]))
            if n_succ <= 0:
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                       % (n_succ/(i+1), total_time/n_succ), flush=True)
            precision = metrics.precision_score(trues, preds, average='macro')
            recall = metrics.recall_score(trues, preds, average='macro')
            f1 = metrics.f1_score(trues, preds, average='macro')
            print("\t(P, R, F1) = (%.3f, %.3f, %.3f)" % (precision, recall, f1))
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/self.d.test.get_size()))

class AttackerRandom(object):
    
    def __init__(self, dataset, symtab, classifier):
        
        self.txt2idx = dataset.get_txt2idx()
        self.idx2txt = dataset.get_idx2txt()
        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=self.txt2idx,
                                    idx2txt=self.idx2txt)
        self.cl = classifier
        self.d = dataset
        self.syms = symtab
    
    def attack(self, x_raw, y, uids, n_iter=20):
        
        iter = 0
        n_stop = 0
       
        batch = get_batched_data([x_raw], [y], self.txt2idx)
        old_prob = self.cl.prob(batch['x'])[0]
        if torch.argmax(old_prob) != y:
            print ("SUCC! Original mistake.")
            return True, torch.argmax(old_prob).cpu().numpy()
        old_prob = old_prob[y]
        
        while iter < n_iter:
            keys = list(uids.keys())
            for k in keys:
                if iter >= n_iter:
                    break
                if n_stop >= len(uids):
                    iter = n_iter
                    break
                if k in self.tokenM.forbidden_uid:
                    n_stop += 1
                    continue

                # don't waste iteration on the "<unk>"s 
                assert not k.startswith('Ġ')
                Gk = 'Ġ' + k
                Gk_idx = self.cl.tokenizer.convert_tokens_to_ids(Gk)
                if Gk_idx == self.cl.tokenizer.unk_token_id:
                    continue
                
                iter += 1
                new_x_raw, new_x_uid = self.tokenM.rename_uid_random(x_raw, k)
                if new_x_raw is None:
                    n_stop += 1
                    print ("skip unk\t%s" % k)
                    continue
                batch = get_batched_data(new_x_raw, [y]*len(new_x_raw), self.txt2idx)
                new_prob = self.cl.prob(batch['x'])
                new_pred = torch.argmax(new_prob, dim=-1)
                for uid, p, pr in zip(new_x_uid, new_pred, new_prob):
                    if p != y:
                        print ("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                               (k, uid, y, old_prob, y, pr[y], p, pr[p]))
                        return True, p
                new_prob_idx = torch.argmin(new_prob[:, y])
                if new_prob[new_prob_idx][y] < old_prob:
                    x_raw = new_x_raw[new_prob_idx]
                    uids[new_x_uid[new_prob_idx]] = uids.pop(k)
                    n_stop = 0
                    print ("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                           (k, new_x_uid[new_prob_idx], y, old_prob, y, new_prob[new_prob_idx][y]))
                    old_prob = new_prob[new_prob_idx][y]
                else:
                    n_stop += 1
                    print ("rej\t%s" % k)
        print ("FAIL!")
        return False, y
    
    def attack_all(self, n_iter=20):
        
        n_succ = 0
        total_time = 0
        trues = []
        preds = []

        st_time = time.time()
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, pred = self.attack(b['raw'][0], b['y'][0], self.syms['te'][b['id'][0]], n_iter)
            if tag==True:
                n_succ += 1
                total_time += time.time() - start_time
            preds.append(int(pred))
            trues.append(int(b['y'][0]))
            if n_succ <= 0:
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                       % (n_succ/(i+1), total_time/n_succ), flush=True)
            precision = metrics.precision_score(trues, preds, average='macro')
            recall = metrics.recall_score(trues, preds, average='macro')
            f1 = metrics.f1_score(trues, preds, average='macro')
            print("\t(P, R, F1) = (%.3f, %.3f, %.3f)" % (precision, recall, f1))
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/self.d.test.get_size()))

class InsAttackerRandom(object):
    
    def __init__(self, dataset, instab, classifier):
        
        self.txt2idx = dataset.get_txt2idx()
        self.idx2txt = dataset.get_idx2txt()
        self.insM = InsModifier(classifier=classifier,
                                txt2idx=self.txt2idx,
                                idx2txt=self.idx2txt,
                                poses=None) # wait to init when attack
        self.cl = classifier
        self.d = dataset
        self.inss = instab
    
    # only support single x: a token-idx list
    def attack(self, x_raw, y, poses, n_iter=20):
        
        self.insM.initInsertDict(poses)

        iter = 0
        n_stop = 0

        batch = get_batched_data([x_raw], [y], self.txt2idx) 
        old_prob = self.cl.prob(batch['x'])[0]
        if torch.argmax(old_prob) != y:
            print ("SUCC! Original mistake.")
            return True, torch.argmax(old_prob).cpu().numpy()
        old_prob = old_prob[y]
        
        while iter < n_iter:
            iter += 1

            # get insertion candidates
            new_x_raw, new_insertDict = self.insM.insert_remove_random(x_raw)
            if new_x_raw == []: # no valid candidates
                n_stop += 1
                continue

            # find if there is any candidate successful wrong classfied
            batch = get_batched_data(new_x_raw, [y]*len(new_x_raw), self.txt2idx)
            new_prob = self.cl.prob(batch['x'])
            new_pred = torch.argmax(new_prob, dim=-1)            
            for insD, p, pr in zip(new_insertDict, new_pred, new_prob):
                if p != y:
                    print ("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                            (self.insM.insertDict["count"], insD["count"], 
                                y, old_prob, y, pr[y], p, pr[p]))
                    return True, p

            # if not, get the one with the lowest target_label_loss
            new_prob_idx = torch.argmin(new_prob[:, y])
            if new_prob[new_prob_idx][y] < old_prob:
                print ("acc\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f)" % \
                        (self.insM.insertDict["count"], new_insertDict[new_prob_idx]["count"], 
                        y, old_prob, y, new_prob[new_prob_idx][y]))
                self.insM.insertDict = new_insertDict[new_prob_idx] # don't forget this step
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y]
            else:
                n_stop += 1
                print ("rej\t%s" % "")
            if n_stop >= 10:
                iter = n_iter
                break
        print ("FAIL!")
        return False, y
    
    def attack_all(self, n_iter=20):

        n_succ = 0
        total_time = 0
        trues = []
        preds = []
        
        st_time = time.time()
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, pred = self.attack(b['raw'][0], b['y'][0], self.inss['stmt_te'][b['id'][0]], n_iter)
            if tag == True:
                n_succ += 1
                total_time += time.time() - start_time
            preds.append(int(pred))
            trues.append(int(b['y'][0]))
            if n_succ <= 0:
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                       % (n_succ/(i+1), total_time/n_succ), flush=True)
            precision = metrics.precision_score(trues, preds, average='macro')
            recall = metrics.recall_score(trues, preds, average='macro')
            f1 = metrics.f1_score(trues, preds, average='macro')
            print("\t(P, R, F1) = (%.3f, %.3f, %.3f)" % (precision, recall, f1))
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/self.d.test.get_size()))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default="0")
    parser.add_argument('--data', type=str, default="../data_defect/codechef.pkl.gz")
    parser.add_argument('--model_dir', type=str, default="../model_defect/codebert/18")
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

    cc = CODECHEF(path=opt.data)
    training_set = cc.train
    valid_set = cc.dev
    test_set = cc.test

    # import transformers after gpu selection
    from codebert import CodeBERTClassifier

    with gzip.open('../data_defect/codechef_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)

    with gzip.open('../data_defect/codechef_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
        
    classifier = CodeBERTClassifier(model_path=opt.model_dir,
                                    num_labels=n_class,
                                    device=device).to(device)
    classifier.eval()

    atk = Attacker(cc, symtab, classifier)
    atk.attack_all(5, 40)
    json.dump(result, file_attack)
    file_attack.close()
    #atk = InsAttacker(cc, instab, classifier)
    #atk.attack_all(5, 40)

    #atk = AttackerRandom(cc, symtab, classifier)
    #atk.attack_all(40)

    #atk = InsAttackerRandom(cc, instab, classifier)
    #atk.attack_all(10) #(40)

