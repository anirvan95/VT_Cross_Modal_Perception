import imageio
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, Dataset
from models.pcmc_model import PCMC

dim_z = 3
dim_x = (16, 16)
dim_u = 1
dim_a = 1
dim_beta = 15   # Fixed from before
dim_alpha = 3
batch_size = 1000
num_iterations = int(5e5)
learning_rate = 0.1


def load_data(file: str, device='cpu') -> Dataset:
    x = torch.from_numpy(np.load(file)['obs']).to(device)
    x = x.to(torch.float32) / 255
    x = x.view([1000, 15, 16*16])
    u = torch.from_numpy(np.load(file)['actions']).to(device)
    u = u.to(torch.float32) / 1.5
    return TensorDataset(x, u)


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # writer = SummaryWriter()
    datasets = dict((k, load_data(file=f'datasets/pendulum/{k}.npz', device=device)) for k in ['training_n'])

    train_loader = DataLoader(datasets['training_n'], batch_size=batch_size, shuffle=True)
    # validation_loader = DataLoader(datasets['validation_mod'], batch_size=batch_size, shuffle=False)

    pcmc = PCMC(dim_x=16*16, dim_u=dim_u, dim_z=dim_z, dim_alpha=dim_alpha, dim_beta=dim_beta, batch_size=batch_size, device=device).to(device)
    # pcmc = torch.load('pcmc.th').to(device)

    # Freeze the VAE model
    for param in pcmc.vae.parameters():
        param.requires_grad = False

    # optimizer = torch.optim.Adam(dvbf.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adadelta(pcmc.parameters(), lr=learning_rate)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    T = num_iterations
    count = 0
    for i in range(num_iterations):
        total_loss = 0
        pcmc.train(True)
        count = 0
        for batch in train_loader:
            count += 1
            if i < 5:
                pcmc.ci = 1e-2
            else:
                count += 1
                if i % 250 == 0:
                    pcmc.ci = np.minimum(1, (1e-2 + count / T))

            x, u = batch[0], batch[1]
            optimizer.zero_grad()
            loss = pcmc.loss(x, u)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # scheduler.step()
        # writer.add_scalar('loss', scalar_value=total_loss, global_step=i)
        print(f'[Epoch {i}] train_loss: {total_loss/count}')
        # writer.add_scalar('learning rate', scalar_value=scheduler.get_lr()[0], global_step=i)
        '''
        dvbf.train(False)
        total_val_loss = 0
        for batch in validation_loader:
            x, u = batch[0], batch[1]
            val_loss = dvbf.loss(x, u)
            total_val_loss += val_loss.item()
        writer.add_scalar('val_loss', scalar_value=total_val_loss, global_step=i)
        print(f'[Epoch {i}] train_loss: {total_loss}, val_loss: {total_val_loss}')
        '''
        if i % 500 == 0:
            torch.save(pcmc, 'runs/pcmc/dvbf_exp_2.th')
            generate(filename=f'dvbf-epoch-{i}')

    # torch.save(dvbf, 'dvbf.th')
    print("Model Trained")

'''
def generate(filename):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # plt.show()
    dvbf = torch.load('checkpoints/dvbf.th').to('cpu')
    dataset = load_data('datasets/pendulum/training_n.npz')
    x = dataset[0:1000][0]
    u = dataset[0:1000][1]
    # x = dataset[0][0].unsqueeze(dim=0)
    # u = dataset[0][1].unsqueeze(dim=0)
    T = u.shape[1]
    z, _ = dvbf.filter(x=x, u=u)
    reconstructed = dvbf.reconstruct(z).view(1, T, -1)

    def format(x):
        img = torch.clip(x * 255., 0, 255).to(torch.uint8)
        return img.view(-1, 16, 16).numpy()

    frames = []
    z_np = z.detach().numpy()
    colors = cm.rainbow(np.linspace(0, 1, 15))
    for i in range(T):
        print(i)
        ax.scatter(z_np[:, i, 0], z_np[:, i, 1], z_np[:, i, 2], marker='.', color=colors[i])

        # gt = format(x[:, i])
        # pred = format(reconstructed[:, i])
        # img = np.concatenate([gt, pred], axis=1).squeeze()
        # cv2.imshow(mat=img, winname='generated')
        # cv2.waitKey(5)
        # plt.imshow(img)
        # plt.show()
        # frames.append(img)

    plt.show()

'''


def generate(filename):
    dvbf = torch.load('runs/pcmc/dvbf_exp_2.th').to('cpu')
    dvbf.device = 'cpu'
    checkpoint_path = 'runs/vae/pendulum_exp_2_vae/checkpt-0000.pth'
    checkpt = torch.load(checkpoint_path)
    state_dict = checkpt['state_dict']
    dvbf.vae.load_state_dict(state_dict, strict=False)

    dataset = load_data('datasets/pendulum/training_n.npz')
    x = dataset[0:2][0]
    u = dataset[0:2][1]
    T = u.shape[1]
    z, _ = dvbf.filter(x=x, u=u)
    reconstructed = dvbf.reconstruct(z).view(2, T, -1)

    def format(x):
        img = torch.clip(x * 255., 0, 255).to(torch.uint8)
        return img.view(16, 16).numpy()

    frames = []

    for i in range(T):

        gt = format(x[0, i])
        pred = format(reconstructed[0, i])
        img = np.concatenate([gt, pred], axis=1).squeeze()
        # cv2.imshow(mat=img, winname='generated')
        # cv2.waitKey(5)
        # plt.imshow(img)
        # plt.show()
        frames.append(img)

    with imageio.get_writer(f"{filename}.mp4", mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)


if __name__ == '__main__':
    # collect_data(5, 15)
    train()
    # generate(filename=f'dvbf-trial-1')
