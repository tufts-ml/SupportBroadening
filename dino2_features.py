import timm
import torch
import torch.utils.data as data
from torchvision.datasets import CIFAR100


if __name__ == "__main__":
    # intialize model and set for evaluations
    model = timm.create_model(
        'vit_base_patch14_dinov2.lvd142m',
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear to get features
    )
    model = model.eval().cuda()
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # set up CIFAR-100
    train_set = CIFAR100("/cluster/tufts/hugheslab/datasets/CIFAR100",
                         True, transforms, download=True)
    test_set = CIFAR100("/cluster/tufts/hugheslab/datasets/CIFAR100",
                        False, transforms, download=True)
    train_loader = data.DataLoader(train_set, batch_size=256, shuffle=False)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False)

    # caches for outputs
    train_embeds = torch.empty((0, 768))
    train_labels = torch.empty((0,))
    test_embeds = torch.empty((0, 768))
    test_labels = torch.empty((0,))
    # loop over data and cache outputs
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.cuda()
            embeds = model(images)
            train_embeds = torch.vstack((train_embeds, embeds.cpu()))
            train_labels = torch.hstack((train_labels, labels))
        for images, labels in test_loader:
            images = images.cuda()
            embeds = model(images)
            test_embeds = torch.vstack((test_embeds, embeds.cpu()))
            test_labels = torch.hstack((test_labels, labels))
    # save caches
    torch.save(train_embeds, "train_embeds.pt")
    torch.save(train_labels, "train_labels.pt")
    torch.save(test_embeds, "test_embeds.pt")
    torch.save(test_labels, "test_labels.pt")
