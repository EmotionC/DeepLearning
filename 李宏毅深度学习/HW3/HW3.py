import time
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm


sys.setrecursionlimit(1000000)  # 设置递归深度
# transform = (
#     transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转，概率0.5
#     transforms.RandomVerticalFlip(p=0.5),  # 垂直翻转，概率0.5
#     transforms.RandomRotation(degrees=90),   # 随机旋转
#     # degrees:旋转的度数范围
#     # resample:重采样过滤器
#     # expend:是否扩大矩形框，以完整容纳旋转后的图片
#     # center:旋转中心设置，远点为左上角，默认中心旋转
#     transforms.RandomApply(
#         [transforms.ColorJitter(brightness=0.8, contrast=0.5, saturation=0.5, hue=0)], p=0.5),  # 随机修改亮度，对比度，饱和度，色调
#     #     brightness:亮度
#     #     contrast:对比度
#     #     saturation:饱和度
#     #     hue:色调
#     transforms.RandomApply(
#         [transforms.RandomAffine(degrees=0, translate=None, scale=(0.5, 1), shear=45, fill=0)], p=0.5),
#     #     随机仿射变换,degrees:旋转角度范围
#     #     translate:平移范围
#     #     scale:比例因子区间
#     #     shear:仿射变换角度范围
# )
# """train_transforms:"""
# valid_tfm = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.RandomChoice(transform),     # 做两次选择
#     transforms.RandomChoice(transform),
#     transforms.ToTensor(),
# ])


def set_train_set():
    transform_1 = (
        transforms.RandomHorizontalFlip(p=1),  # 水平翻转，概率0.5
        # transforms.RandomVerticalFlip(p=0.5),  # 垂直翻转，概率0.5
        transforms.RandomRotation(degrees=90),  # 随机旋转
        # degrees:旋转的度数范围
        # resample:重采样过滤器
        # expend:是否扩大矩形框，以完整容纳旋转后的图片
        # center:旋转中心设置，远点为左上角，默认中心旋转
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.7, contrast=0.5, saturation=0.5, hue=0)], p=1),  # 随机修改亮度，对比度，饱和度，色调
        #     brightness:亮度
        #     contrast:对比度
        #     saturation:饱和度
        #     hue:色调
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=0, translate=None, scale=(0.5, 1), shear=45, fill=0)], p=1),
        #     随机仿射变换
        #     degrees:旋转角度范围
        #     translate:平移范围
        #     scale:比例因子区间
        #     shear:仿射变换角度范围
    )

    transform_2 = (
        # transforms.RandomHorizontalFlip(p=1),  # 水平翻转，概率0.5
        transforms.RandomVerticalFlip(p=1),  # 垂直翻转，概率0.5
        transforms.RandomRotation(degrees=90),  # 随机旋转
        # degrees:旋转的度数范围
        # resample:重采样过滤器
        # expend:是否扩大矩形框，以完整容纳旋转后的图片
        # center:旋转中心设置，远点为左上角，默认中心旋转
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.7, contrast=0.5, saturation=0.5, hue=0)], p=1),  # 随机修改亮度，对比度，饱和度，色调
        #     brightness:亮度
        #     contrast:对比度
        #     saturation:饱和度
        #     hue:色调
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=0, translate=None, scale=(0.5, 1), shear=45, fill=0)], p=1),
        #     随机仿射变换
        #     degrees:旋转角度范围
        #     translate:平移范围
        #     scale:比例因子区间
        #     shear:仿射变换角度范围
    )

    """train_transforms:"""
    train_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    new_train_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomChoice(transform_1),  # 做两次选择
        transforms.RandomChoice(transform_2),
        transforms.ToTensor(),
    ])

    train_set = DatasetFolder('./training/labeled', loader=lambda x: Image.open(x).convert('RGB'), extensions=('.jpg',),
                              transform=train_tfm)
    new_train_set = DatasetFolder('./training/labeled', loader=lambda x: Image.open(x).convert('RGB'),
                                  extensions=('.jpg',), transform=new_train_tfm)

    unlabeled_set = DatasetFolder('./training/unlabeled', loader=lambda x: Image.open(x).convert('RGB'),
                                  extensions=('.jpg',), transform=train_tfm)
    new_unlabeled_set = DatasetFolder('./training/unlabeled', loader=lambda x: Image.open(x).convert('RGB'),
                                      extensions=('.jpg',), transform=new_train_tfm)

    train_set = ConcatDataset([train_set, new_train_set])
    unlabeled_set = ConcatDataset([unlabeled_set, new_unlabeled_set])
    return train_set, unlabeled_set

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

batch_size = 64
test_batch_size = 512

"""
DatasetFolder：读取固定格式的文件夹（主文件夹\label文件夹\图片）路径至主文件夹即可，label会根据label文件夹名称自动生成。
loader会告诉如何读取图片；lambda函数：匿名函数，冒号前为参数列表，冒号后为参数表达式表达式中出现的参数需要在参数列表[arg......]
中有定义。extensions对应图片后缀，注意输入格式。图片的transform在getitem中进行。
"""
# train_set = DatasetFolder(r'E:\Pycharm\PyCharm Community Edition 2022.1.3\PycharmProjects\pytorch learn'
#                           r'\李宏毅深度学习\HW3\training\labeled', loader=lambda x: Image.open(x).convert('RGB'),
#                           extensions=('.jpg',), transform=train_tfm)
# valid_set = DatasetFolder(r'E:\Pycharm\PyCharm Community Edition 2022.1.3\PycharmProjects\pytorch learn'
#                           r'\李宏毅深度学习\HW3\validation', loader=lambda x: Image.open(x).convert('RGB'),
#                           extensions=('.jpg',), transform=train_tfm)
# test_set = DatasetFolder(r'E:\Pycharm\PyCharm Community Edition 2022.1.3\PycharmProjects\pytorch learn'
#                          r'\李宏毅深度学习\HW3\testing', loader=lambda x: Image.open(x).convert('RGB'),
#                          extensions=('.jpg',), transform=test_tfm)
# unlabeled_set = DatasetFolder(r'E:\Pycharm\PyCharm Community Edition 2022.1.3\PycharmProjects\pytorch learn'
#                               r'\李宏毅深度学习\HW3\training\unlabeled', loader=lambda x: Image.open(x).convert('RGB'),
#                               extensions=('.jpg',), transform=train_tfm)

train_set, unlabeled_set = set_train_set()
valid_set = DatasetFolder(r'E:\Pycharm\PyCharm Community Edition 2022.1.3\PycharmProjects\pytorch learn'
                          r'\李宏毅深度学习\HW3\food-11\validation', loader=lambda x: Image.open(x).convert('RGB'),
                          extensions=('.jpg',), transform=test_tfm)
test_set = DatasetFolder(r'E:\Pycharm\PyCharm Community Edition 2022.1.3\PycharmProjects\pytorch learn'
                         r'\李宏毅深度学习\HW3\food-11\testing', loader=lambda x: Image.open(x).convert('RGB'),
                         extensions=('.jpg',), transform=test_tfm)

"""读取数据集，由于DatasetFolder的原因，无法设置并行线程，会增加数据读取时间"""
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

"""simple:使用给定model进行训练"""


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.BatchNorm1d(256),        # 新增BN层,1d对应2,3D的输入
            nn.ReLU(),
            nn.Dropout(p=0.5),          # 新增Dropout层，p:元素被归零的概率
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x


def get_pseudo_labels(dataset, model, threshold=0.65):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # device = "cuda" if torch.cuda.is_available() else "cpu"     # 设置cuda
    global train_set
    # Make sure the model is in eval mode.
    model.eval()  # 验证模式
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)  # 设置softmax，在-1维度做softmax
    # Iterate over the dataset by batches.
    """tqdm：进度条库,传入可迭代对象即可"""
    for batch in tqdm(dataset):  # 原为dataloader
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            """通过模型获取logits数据"""
            logits = model(torch.unsqueeze(img, 0).to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)  # softmax

        # ---------- TODO ----------
        # Filter the data and construct a new dataset.
        # 过滤数据并构建新的数据集。
        if torch.max(probs).item() > threshold:
            train_set = ConcatDataset([train_set, ([(img, torch.argmax(probs).item())])])

    # # Turn off the eval mode.
    model.train()


# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
model = Classifier().to(device)
model.device = device  # 不懂

# For the classification task, we use cross-entropy as the measurement of performance.
"""交叉熵损失"""
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
"""初始化优化器，设定了权重初始值"""
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# The number of training epochs.
n_epochs = 500

# Whether to do semi-supervised learning.
"""半监督学习"""
# do_semi = False

if __name__ == '__main__':
    log_path = r'E:\Pycharm\PyCharm Community Edition 2022.1.3\PycharmProjects\pytorch learn\李宏毅深度学习\HW3\logs'
    model_path = r'E:\Pycharm\PyCharm Community Edition 2022.1.3\PycharmProjects\pytorch learn\李宏毅深度学习\HW3\model.pt'
    writer = SummaryWriter(log_dir=log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    """超参数"""
    best_acc = 0
    valid_acc_last = 0
    valid_acc_threshold = 0.6
    for epoch in range(n_epochs):
        # ---------- TODO ----------
        # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
        # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.
        # 在每个历元中，重新标记未标记的数据集以进行半监督学习。
        # 然后可以将标记数据集和伪标记数据集组合起来进行训练。
        # if do_semi:
            # Obtain pseudo-labels for unlabeled data using trained model.
            # 使用经过训练的模型获得未标记数据的伪标签。
            # pseudo_set = get_pseudo_labels(unlabeled_set, model)

            # Construct a new dataset and a data loader for training.
            # This is used in semi-supervised learning only.
            # 构建一个新的数据集和一个用于训练的数据加载器。
            # 这仅用于半监督学习。
            # concat_dataset = ConcatDataset([train_set, pseudo_set])
            # train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
            #                           pin_memory=True)
        if valid_acc_last > valid_acc_threshold:
            valid_acc_threshold = valid_acc_last

            if len(train_set) != 19732:     # 3080*2+6786*2
                get_pseudo_labels(unlabeled_set, model, threshold=0.75)

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()


        # These are used to record information in training.
        train_loss = []
        train_accs = []

        # Iterate the training set by batches.
        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            """计算交叉熵损失时无需进行softmax。因为该函数已自动进行"""
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            # 剪裁坡度标准，实现稳定训练。
            """梯度裁剪：对权重梯度向量的范数进行裁剪"""
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            """argmax：返回指定维度最大值的序号，dim指定哪个维度，即消去哪个维度，也就是把dim这个维度的，变成这个维度的最大值的index"""
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            # acc = torch.eq(logits.argmax(dim=-1), labels.squeeze().to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        writer.add_scalar('loss/train(data_add)', train_loss, epoch)
        writer.add_scalar('acc/train', train_acc, epoch)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        valid_acc_last = valid_acc

        writer.add_scalar('loss/valid', valid_loss, epoch)
        writer.add_scalar('acc/valid', valid_acc, epoch)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_path)
            print('[{:03d}/{:03d}] saving model with acc {:.3f}'.format(epoch+1, n_epochs, best_acc))
    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()

    # Initialize a list to store the predictions.
    predictions = []

    # Iterate the testing set by batches.
    for batch in tqdm(test_loader):
        # A batch consists of image data and corresponding labels.
        # But here the variable "labels" is useless since we do not have the ground-truth.
        # If printing out the labels, you will find that it is always 0.
        # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
        # so we have to create fake labels to make it work normally.
        imgs, labels = batch

        # We don't need gradient in testing, and we don't even have labels to compute loss.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # Take the class with greatest logit as prediction and record it.
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    # Save predictions into the file.
    with open(r"E:\Pycharm\PyCharm Community Edition 2022.1.3\PycharmProjects"
              r"\pytorch learn\李宏毅深度学习\HW3\predict.csv", "w") as f:

        # The first row must be "Id, Category"
        f.write("Id,Category\n")

        # For the rest of the rows, each image id corresponds to a predicted class.
        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")

    writer.close()
