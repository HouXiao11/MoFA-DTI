import pandas as pd
import torch
import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score,accuracy_score,recall_score, matthews_corrcoef
from models import binary_cross_entropy, cross_entropy_logits
from prettytable import PrettyTable
from tqdm import tqdm
import copy

def save_model(model):
    model_path = r'../output/model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model, model_path+'model.pt')
    new_model = torch.load(model_path + 'model.pt')
    return new_model

class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, data_name, split, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.n_class = config["DECODER"]["BINARY"]

        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0

        self.lr_decay = config["SOLVER"]["LR_DECAY"]
        self.decay_interval = config["SOLVER"]["DECAY_INTERVAL"]
        self.use_ld = config['SOLVER']["USE_LD"]

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0
        self.best_auprc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"] + f'{data_name}/{split}/'

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy","MCC",
                              "Threshold", "Test_loss"]

        train_metric_header = ["# Epoch", "Train_loss"]

        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)


    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            if self.use_ld:
                if self.current_epoch % self.decay_interval == 0:
                    self.optim.param_groups[0]['lr'] *= self.lr_decay

            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))

            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc, val_loss = self.test(dataloader="val")

            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch

            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))
        
        auroc, auprc, f1, sensitivity, specificity, accuracy, mcc, test_loss, thred_optim, precision = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                            accuracy, mcc, thred_optim, test_loss]))
        self.test_table.add_row(test_lst)
        
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " mcc " + str(mcc) + " Thred_optim " + str(thred_optim))
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["mcc"] = [mcc] 
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision
        self.save_result()

        return self.test_metrics

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }

        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        val_prettytable_file = os.path.join(self.output_dir, f"valid_markdowntable_{timestamp}.txt")
        test_prettytable_file = os.path.join(self.output_dir, f"test_markdowntable_{timestamp}.txt")
        train_prettytable_file = os.path.join(self.output_dir, f"train_markdowntable_{timestamp}.txt")

        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_p, labels) in enumerate(tqdm(self.train_dataloader, ncols=80)):
            self.step += 1
            v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
            self.optim.zero_grad()
            v_d, v_p, f, score = self.model(v_d, v_p)
            if self.n_class == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()

        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch


    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        #
        df = {'drug': [], 'protein': [], 'y_pred': [], 'y_label': []}
        #
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels) in enumerate(data_loader):
                v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
                if dataloader == "val":
                    v_d, v_p, f, score = self.model(v_d, v_p)
                elif dataloader == "test":
                    v_d, v_p, f, score = self.best_model(v_d, v_p)
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()

                if dataloader == 'test':
                    df['drug'] = df['drug'] + v_d.to('cpu').tolist()
                    df['protein'] = df['protein'] + v_p.to('cpu').tolist()

        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches

        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            try:
                precision = tpr / (tpr + fpr)
            except RuntimeError:
                raise ('RuntimeError: the divide==0')
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = accuracy_score(y_label, y_pred_s)
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            mcc = matthews_corrcoef(y_label, y_pred_s)

            precision1 = precision_score(y_label, y_pred_s)
            df['y_label'] = y_label
            df['y_pred'] = y_pred
            data = pd.DataFrame(df)
            data.to_csv('../output/visualization.csv', index=False)

            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, mcc, test_loss, thred_optim, precision1
        else:
            return auroc, auprc, test_loss
