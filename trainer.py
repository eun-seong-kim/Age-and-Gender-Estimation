import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from sklearn.metrics import mean_absolute_error, f1_score
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet
from models.model import AgeGenderModel


class Trainer():
    def __init__(self, params, device, pretrained='efficientnet-b0'):
        self.params = params
        self.device = device

        self.base_model = EfficientNet.from_pretrained(pretrained)
        self.model = AgeGenderModel(self.base_model, dropout_rate=self.params['dropout_rate']).to(self.device)

        self.age_criterion = nn.MSELoss()
        self.gender_criterion = nn.BCELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'],
                                          weight_decay=self.params['l2_reg_lambda'])
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.5)

        self.log_dir = os.path.abspath(os.path.join(os.curdir, "runs"))
        self.summary_dir = os.path.join(self.log_dir, "summaries")
        self.writer = SummaryWriter(self.summary_dir)

        self.best_mae = float('inf')
        self.early_stopping_patience = self.params.get("early_stopping_patience", 15)
        self.early_stopping_counter = 0
        self.early_stop = False

    def train(self, train_loader, val_loader):
        start_time = time.time()
        global_steps = 0

        print('========================================')
        print("Start training...")

        for epoch in range(self.params['max_epochs']):
            total_age_loss = 0
            total_gender_loss = 0
            batch_cnt = 0

            self.model.train()

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                age = labels[:, 0].view(-1, 1).to(self.device) / 100.0
                gender = labels[:, 1].view(-1, 1).to(self.device)

                self.optimizer.zero_grad()
                age_outputs, gender_outputs = self.model(inputs)

                age_loss = self.age_criterion(age_outputs, age)
                gender_loss = self.gender_criterion(gender_outputs, gender)

                loss = 0.7 * age_loss + 0.3 * gender_loss
                loss.backward()
                self.optimizer.step()

                total_age_loss += age_loss.item()
                total_gender_loss += gender_loss.item()
                batch_cnt += 1

                self.writer.add_scalar("LR/Learning_rate", self.lr_scheduler.get_last_lr()[0], global_steps)
                global_steps += 1

            self.lr_scheduler.step()

            # training average loss
            age_ave_loss = total_age_loss / batch_cnt
            gender_ave_loss = total_gender_loss / batch_cnt
            training_time = (time.time() - start_time) / 60

            self.writer.add_scalar("TrainAge/Loss", age_ave_loss, epoch)
            self.writer.add_scalar("TrainGender/Loss", gender_ave_loss, epoch)

            print('========================================')
            print(f"Epoch {epoch + 1} / Global Steps: {global_steps}")
            print(f"Training Age Loss: {age_ave_loss:.4f}")
            print(f"Training Gender Loss: {gender_ave_loss:.4f}")
            print(f"Training Time: {training_time:.2f} minutes")

            self.validate(val_loader, epoch)
            if self.early_stop:
                print(f"Training stopped early at epoch {epoch+1}.")
                break

    def validate(self, val_loader, epoch):
        total_age_loss = 0
        total_gender_loss = 0
        gender_correct = 0
        batch_cnt = 0

        age_outputs_list = []
        age_labels_list = []
        gender_outputs_list = []
        gender_labels_list = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                age = labels[:, 0].view(-1, 1).to(self.device) / 100.0
                gender = labels[:, 1].view(-1, 1).to(self.device)

                age_outputs, gender_outputs = self.model(inputs)

                age_loss = self.age_criterion(age_outputs, age)
                gender_loss = self.gender_criterion(gender_outputs, gender)

                total_age_loss += age_loss.item()
                total_gender_loss += gender_loss.item()
                batch_cnt += 1

                age_outputs_list.append(age_outputs * 100)
                age_labels_list.append(age * 100)
                gender_outputs_list.append(gender_outputs)
                gender_labels_list.append(gender)

                pred_gender = (gender_outputs > 0.5).float()
                gender_correct += (pred_gender == gender).sum().item()

        all_age_outputs = torch.cat(age_outputs_list).cpu().numpy()
        all_age_labels = torch.cat(age_labels_list).cpu().numpy()
        all_gender_outputs = torch.cat(gender_outputs_list).cpu().numpy()
        all_gender_labels = torch.cat(gender_labels_list).cpu().numpy()

        age_ave_loss = total_age_loss / batch_cnt
        gender_ave_loss = total_gender_loss / batch_cnt
        gender_acc = (gender_correct / len(val_loader.dataset)) * 100
        age_mae = mean_absolute_error(all_age_labels, all_age_outputs)
        gender_f1 = f1_score(all_gender_labels.astype(int), (all_gender_outputs > 0.5).astype(int))

        print("<Validation Results>")
        print(f"Validation Age Loss: {age_ave_loss:.4f}")
        print(f"Validation Gender Loss: {gender_ave_loss:.4f}")
        print(f"Age MAE: {age_mae:.2f}")
        print(f"Gender Accuracy: {gender_acc:.2f}%")
        print(f"Gender F1 Score: {gender_f1:.4f}")

        self.writer.add_scalar("ValAge/Loss", age_ave_loss, epoch)
        self.writer.add_scalar("ValAge/MAE", age_mae, epoch)
        self.writer.add_scalar("ValGender/Loss", gender_ave_loss, epoch)
        self.writer.add_scalar("ValGender/F1", gender_f1, epoch)
        self.writer.add_scalar("ValGender/Acc", gender_acc, epoch)

        # Save best model
        if age_mae < self.best_mae - 1e-4:
            self.early_stopping_counter = 0
            save_path = os.path.join(self.log_dir, f'best_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
            }, save_path)
            self.best_mae = age_mae
            print(f"New best model saved to: {save_path}")

        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                print("Early stopping triggered.")
                self.early_stop = True
        
    def test(self, test_loader):

        checkpoint = torch.load(self.params['model_path'])
        self.model.load_state_dict(checkpoint['model_state_dict'])

        gender_classes = ('Male', 'Female')
        
        if self.params["mode"] == "test":
            age_outputs_list = []
            age_labels_list = []
            gender_outputs_list = []
            gender_labels_list = []
            gender_correct = 0
            
            self.model.eval()
            with torch.no_grad():
                for img, labels in test_loader:
                    inputs = inputs.to(self.device)
                    age = labels[:, 0].view(-1, 1).to(self.device) / 100.0
                    gender = labels[:, 1].view(-1, 1).to(self.device)

                    age_outputs, gender_outputs = self.model(inputs)

                    age_outputs_list.append(age_outputs * 100)
                    age_labels_list.append(age * 100)
                    gender_outputs_list.append(gender_outputs)
                    gender_labels_list.append(gender)

                    pred_gender = (gender_outputs > 0.5).float()
                    gender_correct += (pred_gender == gender).sum().item()

            all_age_outputs = torch.cat(age_outputs_list).cpu().numpy()
            all_age_labels = torch.cat(age_labels_list).cpu().numpy()
            all_gender_outputs = torch.cat(gender_outputs_list).cpu().numpy()
            all_gender_labels = torch.cat(gender_labels_list).cpu().numpy()


            gender_acc = (gender_correct / len(test_loader.dataset)) * 100
            age_mae = mean_absolute_error(all_age_labels, all_age_outputs)
            gender_f1 = f1_score(all_gender_labels.astype(int), (all_gender_outputs > 0.5).astype(int))

            print("<Test Results>")
            print(f"Age MAE: {age_mae:.2f}")
            print(f"Gender Accuracy: {gender_acc:.2f}%")
            print(f"Gender F1 Score: {gender_f1:.4f}")
                    
        
        elif self.params["mode"] == "inference":
            self.model.eval()
            with torch.no_grad():
                for img, labels in test_loader:
                    plt.imshow(img.squeeze().permute(1, 2, 0))
                    plt.show()

                    age = labels[:, 0].view(-1, 1).to(self.device)
                    gender = labels[:, 1].view(-1, 1).to(self.device)

                    age_output, gender_output = self.model(img.to(self.device))
                    pred_gender = (gender_output > 0.5).float()

                print("--------------------------------------")
                print('<Age inference>')
                print("truth:", int(age))
                print("model prediction:", int(age_output.item())*100)
                print('<Gender inference>')
                print("truth:", gender_classes[int(gender)])               
                print("model prediction:", gender_classes[int(pred_gender.item())])
            