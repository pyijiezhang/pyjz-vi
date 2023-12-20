from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import wandb
import numpy as np

# import scipy
# from scipy.special import softmax
import bayesian_torch.models.bayesian.simple_cnn_variational as simple_cnn

len_trainset = 60000
len_testset = 10000


def train(
    obj, model, device, train_loader, optimizer, epoch, num_mc, batch_size, log_interval
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output_ = []
        kl_ = []
        for mc_run in range(num_mc):
            output, kl = model(data)
            output_.append(output)
            kl_.append(kl)
        output = torch.stack(output_)
        kl = torch.mean(torch.stack(kl_), dim=0)
        log_p = torch.distributions.Categorical(logits=output).log_prob(target)
        # nll_loss = F.nll_loss(output, target)
        # ELBO loss

        if obj == "pacm":
            log_p_bayes = torch.logsumexp(log_p, 0) - torch.log(
                torch.tensor(log_p.shape[0])
            )
            nll_loss = -log_p_bayes.mean()
        if obj == "pac":
            nll_loss = -log_p.mean()
        loss = nll_loss + (kl / batch_size)

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

        wandb.log({f"mfvi/train/loss": loss.item()}, step=epoch)


def test(
    model,
    device,
    test_loader,
    epoch,
    batch_size,
):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, kl = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction="sum").item() + (
                kl / batch_size
            )  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    val_accuracy = correct / len(test_loader.dataset)

    wandb.log({f"mfvi/test/accuracy": val_accuracy}, step=epoch)
    wandb.log({f"mfvi/test/loss": test_loss}, step=epoch)


def evaluate(model, device, test_loader, num_monte_carlo):
    pred_probs_mc = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        pred_probs_mc = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for mc_run in range(num_monte_carlo):
                model.eval()
                output, _ = model.forward(data)
                output = F.log_softmax(output, dim=1)
                # get probabilities from log-prob
                pred_probs = torch.exp(output)
                pred_probs_mc.append(pred_probs.cpu().data.numpy())

        target_labels = target.cpu().data.numpy()
        pred_mean = np.mean(pred_probs_mc, axis=0)
        Y_pred = np.argmax(pred_mean, axis=1)
        print("Test accuracy:", (Y_pred == target_labels).mean() * 100)
        np.save("./probs_mnist_mc.npy", pred_probs_mc)
        np.save("./mnist_test_labels_mc.npy", target_labels)


def main(
    project_name=None,
    run_name=None,
    obj="pac",
    batch_size=128,
    test_batch_size=10000,
    epochs=100,
    lr=0.01,
    gamma=0.7,
    no_cuda=False,
    seed=1,
    log_interval=10,
    save_dir="./checkpoint/bayesian",
    mode="train",
    num_monte_carlo=20,
    num_mc=5,
):
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    run_name = f"mnist_{obj}"
    wandb.init(
        project=project_name,
        name=f"{run_name}",
        mode="online",
        config={
            "obj": obj,
            "batch_size": batch_size,
            "test_batch_size": test_batch_size,
            "epochs": epochs,
            "lr": lr,
            "gamma": gamma,
            "seed": seed,
            "num_monte_carlo": num_monte_carlo,
            "num_mc": num_mc,
        },
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=False,
        **kwargs,
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = simple_cnn.SCNN()
    model = model.to(device)

    print(mode)
    if mode == "train":
        for epoch in range(1, epochs + 1):
            if epoch < epochs / 2:
                optimizer = optim.Adadelta(model.parameters(), lr=lr)
            else:
                optimizer = optim.Adadelta(model.parameters(), lr=lr / 10)
            scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
            train(
                obj,
                model,
                device,
                train_loader,
                optimizer,
                epoch,
                num_mc,
                batch_size,
                log_interval,
            )
            test(model, device, test_loader, epoch, batch_size)
            scheduler.step()

            torch.save(
                model.state_dict(), save_dir + f"/mnist_bayesian_scnn_{obj}_{epoch}.pth"
            )

    elif mode == "test":
        checkpoint = save_dir + "/mnist_bayesian_scnn.pth"
        model.load_state_dict(torch.load(checkpoint))
        evaluate(model, device, test_loader)


if __name__ == "__main__":
    import fire
    import os

    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", default="dryrun")
    fire.Fire(main)
