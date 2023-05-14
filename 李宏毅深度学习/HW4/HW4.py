import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import random
from pathlib import Path
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        # Load the mapping from speaker name to their corresponding id.
        """mapping:字典形式，key为是speakers，value为speakers对应的数字ID"""
        mapping_path = Path(data_dir)/'mapping.json'    # 该用法相当于os.path.join，即拼接路径
        mapping = json.load(mapping_path.open())        # 读取.json文件
        self.speaker2id = mapping['speaker2id']

        # Load metadata of training data.
        """metadata:字典形式，key为不同的speakerID，value为feature_path和mel_len,同一个speaker对应多个value"""
        metadata_path = Path(data_dir)/'metadata.json'
        metadata = json.load(open(metadata_path))['speakers']   # 与上一小节作用相同，但代码形式更为简练

        # Get the total number of speaker.
        self.speaker_num = len(metadata.keys())     # 用法：dict.keys()返回字典中所有的key，返回类型为dict_key,返回顺序为key添加顺序，另外的方法还有dict.values(),dict.items()
        self.data = []
        for speaker in metadata.keys():             # 遍历列表，返回speaker
            for utterances in metadata[speaker]:    # 遍历字典，返回speaker对应的value
                self.data.append([utterances['feature_path'], self.speaker2id[speaker]])    # 在data数据集中添加元素：[feature,speaker2id]

    def __len__(self):
        return len(self.data)   # 返回数据data的长度

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # Load preprocessed mel-spectrogram.
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # Segment mel-spectrogram into "segment_len" frames.
        if len(mel) > self.segment_len:                                 # 判断从网络获取到的数据长度
            # Randomly get the starting point of the segment.
            start = random.randint(0, len(mel) - self.segment_len)      # randint返回一个从start（包含start）到stop（包含stop）之间的整数值
            # Get a segment with "segment_len" frames.
            mel = torch.FloatTensor(mel[start:start + self.segment_len])    # 选定mel长度与设定长度一致
        else:
            mel = torch.FloatTensor(mel)                                    # 如果小于，怎么说？
        # Turn the speaker id into long for computing loss later.
        speaker = torch.FloatTensor([speaker]).long()                       # 返回speaker列表，并转换为长整型
        return mel, speaker

    def get_speaker_number(self):
        """该方法返回speaker的长度"""
        return self.speaker_num


def collate_batch(batch):
    # Process features within a batch.
    """Collate a batch of data.数据排序？"""
    mel, speaker = zip(*batch)      # zip(*)意为解压，即将zip压缩成的元组列表形式解压回二维矩阵形式
    # Because we train the model batch by batch,
    # we need to pad the features in the same batch to make their lengths the same.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value, 描述与代码不符！！！
    """
    pad_sequence:以padding_value填充sequence中的各张量，使其长度相等。
    形式转换：原序列L×*，*是任意数量的尾随维度，可以为None；L为序列长度
            转换后：T×B×*，T为所有序列中最长序列的长度， B为sequence中元素个数。
            batch_first:是否将序列的维度作为第一维度，默认为False。
            padding_value:填充数值，默认为0
            example:S1:Tensor（22， 300）， S2：Tensor(15, 300), S3:Tensor(25, 300); 
            pad_sequence([a,b,c]),返回为Tensor(25, 3, 300)
            pad_sequence([a,b,c],batch_first=True),返回为Tensor(3, 25, 300)
    """
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers):
    """Generate dataloader:产生Dataloader"""
    dataset = MyDataset(data_dir)
    speaker_num = dataset.get_speaker_number()  # 单独调用类中的方法
    # Split dataset into training dataset and validation dataset
    # 确定训练集与验证集比例
    train_len = int(0.9 * len(dataset))
    lengths = [train_len, len(dataset) - train_len]
    train_set, valid_set = random_split(dataset, lengths)       # random_split:随机将一个数据集划分为给定长度的不重叠的新数据集
    """
    dataset:要划分的数据集
    lengths:要划分的长度
    generator:用于随机排列的生成器
    """

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,             # 当样本不能被batch_size整除时，舍弃最后一批数据
        num_workers=n_workers,
        pin_memory=True,            # 如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中.
        collate_fn=collate_batch,   # 重写了collate_fn函数，使其功能变为补足
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    return train_loader, valid_loader, speaker_num


class Classifier(nn.Module):
    def __init__(self, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()      # python3版本新写法，可不用在括号内输入父类名称,self
        # Project the dimension of features from that of input into d_model.
        # 将输入的特征尺寸投影到d_model中。
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=2
        )
        """自注意力机制层：
        d_model:输入预期特征的数量
        nhead:多头注意力模型中的头数
        dim_feedforward:前馈网络模型的维度，默认2048
        dropout:dropout值
        activation:中间层的激活函数，可以是字符串或一元可调用函数，默认为relu
        layer_norm-eps:层归一化组件中的eps值
        batch_first:如为True，则输入和输出张量作为(batch, seq, feature)提供，默认False(seq, batch, feature)
        norm_first:如为True，层规范分别在注意和前馈操作之前完成，默认False
        """
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Project the dimension of features from d_model into speaker nums.
        """fc层，主要有fc和ReLu层，后续可添加BN层和dropout层"""
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        args:
          mels: (batch size, length, 40)
        return:
          out: (batch size, n_spks)
        构建前馈网络：输入mels，输出分类？
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)      # permute: 对任意高维度矩阵进行转置，与transpose不同，transpose只能转置两个维度，不能转置所有的维度
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)       # transpose： 只转置两个维度
        # mean pooling
        stats = out.mean(dim=1)         # 平均池化，按length维度求平均

        # out: (batch, n_spks)
        out = self.pred_layer(stats)    # FC层
        return out


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    """
    关于学习率变化的函数
     Create a schedule with a learning rate that decreases following the values of the cosine function between the
     initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
     initial lr set in the optimizer.

     Args:
       optimizer (:class:`~torch.optim.Optimizer`):
         The optimizer for which to schedule the learning rate.
       num_warmup_steps (:obj:`int`):
         The number of steps for the warmup phase.
       num_training_steps (:obj:`int`):
         The total number of training steps.
       num_cycles (:obj:`float`, `optional`, defaults to 0.5):
         The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
         following a half-cosine).
       last_epoch (:obj:`int`, `optional`, defaults to -1):
         The index of the last epoch when resuming training.

     Return:
       :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
     """

    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step)/float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps)/float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""
    """一个batch的前馈计算，返回其loss和accuracy"""

    mels, labels = batch
    mels = mels.to(device)
    labels = labels.to(device)

    outs = model(mels)

    loss = criterion(outs, labels)

    # Get the speaker id with the highest probability.
    preds = outs.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy


def valid(dataloader, model, criterion, device):
    """Validate on validation set."""
    """整个验证集的损失及准确率计算,返回整个validation的平均准确率"""
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc='Valid', unit='uttr')
    """
    tqdm详解：
    iterable:可迭代对象，手动更新时无需设置
    desc：字符串， 进度条左端描述文字
    total：总的项目数
    leave：bool值，迭代完成后是否保留进度条
    file：输入指向位置，默认为终端
    ncols：调整进度条宽度，设为0则没有进度条，只有输出信息
    unit：描述处理文字的项目，默认为it,例：it/s
    """
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()
        # 进度条更新步长，即每次更新的数据量
        pbar.update(dataloader.batch_size)
        # 进度条右边显示信息
        pbar.set_postfix(
            loss=f'{running_loss / (i+1):.2f}',
            accuracy=f'{running_accuracy / (i+1):.2f}',
        )

    pbar.close()
    model.train()

    return running_accuracy / len(dataloader)


def parse_args():
    """arguments"""
    """
    超参数：
    data_dir：基础路径
    save_path：模型保存路径
    batch_size：训练用批次大小
    n_workers：线程数量
    valid_steps:训练模型中的batch数，事实上只有1952个，训练样本共有62494个
    warmup_steps：
    save_steps：保存模型steps，每10000steps保存一次模型
    total_steps：迭代的总的batch数
    """
    config = {
        'data_dir': './Dataset',
        'save_path': './model.ckpt',
        'batch_size': 32,
        'n_workers': 6,
        'valid_steps': 2000,
        'warmup_steps': 1000,
        'save_steps': 10000,
        'total_steps': 70000,
    }

    return config


def main(
        data_dir,
        save_path,
        batch_size,
        n_workers,
        valid_steps,
        warmup_steps,
        total_steps,
        save_steps,
):
    """main function"""
    """train主函数：自定义迭代器（原因未知）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Info]: Use {device} now!')

    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
    train_iterator = iter(train_loader)     # 创建迭代对象
    print(f'[Info]: Finish loading data!', flush=True)

    model = Classifier(n_spks=speaker_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)      # 优化器使用AdamW优化器：使用了L2正则惩罚项
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)   # 没搞清楚
    print(f'[Info]: Finish creating model!', flush=True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc='Train', unit='step')

    for step in range(total_steps):
        # Get data
        """异常处理，在该处理中实现迭代"""
        try:
            batch = next(train_iterator)
        except StopIteration:       # 意思为迭代器没有更多的值
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # 模型更新
        loss.backward()
        optimizer.step()
        scheduler.step()        # 更新学习率
        optimizer.zero_grad()

        # Log
        pbar.update()           # 手动调节进度条的情况下，更新进度条数据
        # 进度条右边显示内容
        pbar.set_postfix(
            loss=f'{batch_loss:.2f}',
            accuracy=f'{batch_accuracy:.2f}',
            step=step + 1,
        )

        # Do validation
        """每2000steps计算一次验证集准确率"""
        if (step + 1) % valid_steps == 0:
            pbar.close()    # 不使用训练进度条

            valid_accuracy = valid(valid_loader, model, criterion, device)  # 计算平均损失

            # keep the best model， 获取最佳模型
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc='Train', unit='step')

        """每10000steps保存一次模型数据"""
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f'Step {step + 1}, best model, saved. (accuracy={best_accuracy:.4f})')

    pbar.close()


class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        testdata_path = Path(data_dir) / "testdata.json"
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        return feat_path, mel


def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)

    return feat_paths, torch.stack(mels)


def parse_args_test():
    """arguments"""
    config = {
        "data_dir": "./Dataset",
        "model_path": "./model.ckpt",
        "output_path": "./output.csv",
    }

    return config


def main_test(
        data_dir,
        model_path,
        output_path,
):
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())

    dataset = InferenceDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=6,
        collate_fn=inference_collate_batch,
    )
    print(f"[Info]: Finish loading data!", flush=True)

    speaker_num = len(mapping["id2speaker"])
    model = Classifier(n_spks=speaker_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"[Info]: Finish creating model!", flush=True)

    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


if __name__ == '__main__':
    main(**parse_args())
    main_test(**parse_args_test())
