import torch
import torch.nn as nn
import torch.nn.functional as F

class DSAug(nn.Module):
    def __init__(self):
        super(DSAug, self).__init__()

        # layers for the localization network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # fully connected layers to regress the affine transformation parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # initialize the weights and bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # new layer to generate the rotation and scaling parameters
        self.theta_gen = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 4)  # 1 scale parameter, 1 angle, 1 tx, 1 ty
        )

    def stn(self, x):
      xs = self.localization(x)
      xs = xs.view(-1, 10 * 4 * 4)
      theta = self.fc_loc(xs)
      theta = theta.view(-1, 2, 3)

      # get the rotation and scaling parameters
      params = self.theta_gen(xs)
      s, angle, tx, ty = params.split(1, dim=1)
      s = torch.sigmoid(s)  # scale parameter should be positive
      cos_theta = torch.cos(angle)
      sin_theta = torch.sin(angle)

      # scaling by adjusting theta (with clone one)
      new_theta = theta.clone()
      new_theta[:, 0, 0] *= s.squeeze()
      new_theta[:, 1, 1] *= s.squeeze()

      # rotation
      rotation_matrices = torch.stack([torch.cat([cos_theta, -sin_theta], dim=1),torch.cat([sin_theta, cos_theta], dim=1)], dim=2)


      # Apply rotation
      new_theta[:, :2, :2] = torch.bmm(new_theta[:, :2, :2], rotation_matrices)

      # translation
      new_theta[:, 0, 2] += tx.squeeze()
      new_theta[:, 1, 2] += ty.squeeze()

      # grid generator & sampler
      grid = F.affine_grid(theta, x.size())
      x = F.grid_sample(x, grid)

      return x, new_theta


    def forward(self, x):
        x, theta = self.stn(x)
        return x, theta
