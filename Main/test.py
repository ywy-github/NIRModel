import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from Main.Metrics import all_metrics
from Main.data_loader import MyData
from Main.models.Vq_VAE_Join_Classifier_multi_route import Focal_Loss, joint_loss_function

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    epochs = 500

    num_hiddens = 64
    num_residual_hiddens = 32
    num_residual_layers = 2

    embedding_dim = num_hiddens * 3
    num_embeddings = 512

    commitment_cost = 0.25

    decay = 0.99

    weight_positive = 2  # 调整这个权重以提高对灵敏度的重视

    learning_rate = 1e-5

    lambda_recon = 0.2
    lambda_vq = 0.2
    lambda_classifier = 0.6

    # 读取数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3281,), (0.2366,))  # 设置均值和标准差
    ])

    test_benign_data = MyData("../data/NIR_Wave1/test/benign", "benign", transform=transform)
    test_malignat_data = MyData("../data/NIR_Wave1/test/malignant", "malignant", transform=transform)
    test_data = test_benign_data + test_malignat_data

    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True)

    model = torch.load("../models/result/wave1_test1.pth", map_location=device)

    criterion = Focal_Loss()
    criterion.to(device)

    test_predictions = []
    test_targets = []
    test_results = []

    test_res_recon_error = []
    test_res_perplexity = []
    total_test_loss = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            data, targets, dcm_names = batch
            data = data.to(device)
            targets = targets.to(device)
            vq_loss, data_recon, perplexity, classifier_outputs = model(data)
            data_variance = torch.var(data)
            recon_loss = F.mse_loss(data_recon, data) / data_variance
            classifier_loss = criterion(classifier_outputs, targets.view(-1, 1))
            total_loss = joint_loss_function(recon_loss, vq_loss, classifier_loss, lambda_recon, lambda_vq,
                                             lambda_classifier)

            predicted_labels = (classifier_outputs >= 0.5).int().squeeze()
            # 记录每个样本的dcm_name、预测概率值和标签
            for i in range(len(dcm_names)):
                test_results.append({'dcm_name': dcm_names[i], 'prob': classifier_outputs[i].item(),
                                     'probility': predicted_labels[i].item(), 'label': targets[i].item()})
            test_predictions.extend(predicted_labels.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
            test_res_recon_error.append(recon_loss.item())
            test_res_perplexity.append(perplexity.item())
            total_test_loss.append(total_loss.item())

            # concat = torch.cat((data[0].view(128, 128),
            #                     data_recon[0].view(128, 128)), 1)
            # plt.matshow(concat.cpu().detach().numpy())
            # plt.show()

    train_acc, train_sen, train_spe = all_metrics(test_targets, test_predictions)

    print("测试集 acc: {:.4f}".format(train_acc) + "sen: {:.4f}".format(train_sen) +
          "spe: {:.4f}".format(train_spe) + "loss: {:.4f}".format(np.mean(total_test_loss[-10:])))

    df = pd.DataFrame(test_results)
    df.to_excel("../models/result/wave1_test1.xlsx", index=False)
