import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", message=".*xFormers.*")

def write_point_cloud(ply_filename, points):
    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3]*255, point[4]*255, point[5]*255))

    out_file = open(ply_filename, "w")
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()


class DINOFeatureExtractor(nn.Module):
    def __init__(self):
        super(DINOFeatureExtractor, self).__init__()
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.dinov2_model = self.dinov2_model.cuda()
        self.dinov2_model.eval()

    def forward(self, gt_image, feature_size=50):
        with torch.no_grad():
            feature_height = feature_width = feature_size
            gt_img = F.interpolate(gt_image.unsqueeze(0), size=(feature_height * 14, feature_width * 14), mode='bilinear', align_corners=False)
            gt_embeddings = self.dinov2_model.forward_features(gt_img)
            dino_features = gt_embeddings["x_norm_patchtokens"].reshape(1, feature_height, feature_width, -1).permute(0, 3, 1, 2)
        return dino_features.squeeze()


class DINOv3FeatureExtractor(nn.Module):
    def __init__(self, repo_dir=None, weight_path=None):
        super(DINOv3FeatureExtractor, self).__init__()
        _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        REPO_DIR = repo_dir or os.environ.get(
            "DINOV3_REPO", os.path.join(_pkg_root, "dinov3_repo")
        )
        weight_path = weight_path or os.environ.get(
            "DINOV3_WEIGHTS",
            os.path.join(_pkg_root, "utils", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
        )
        self.dinov3_model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=weight_path)
        self.dinov3_model = self.dinov3_model.cuda()
        self.dinov3_model.eval()

    @torch.no_grad()
    def forward(self, gt_image, feature_size=50):
        with torch.no_grad():
            feature_height = feature_width = feature_size
            gt_img = F.interpolate(gt_image.unsqueeze(0), size=(feature_height * 16, feature_width * 16), mode='bilinear', align_corners=False)
            gt_embeddings = self.dinov3_model.forward_features(gt_img)
            dino_features = gt_embeddings["x_norm_patchtokens"].reshape(1, feature_height, feature_width, -1).permute(0, 3, 1, 2)
        return dino_features.squeeze()


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(384, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
    
    def get_regularizer(self):
        return torch.max(abs(self.mlp[0].weight.data)) * torch.max(abs(self.mlp[2].weight.data))

    def get_residual_loss(self, mask, lower_mask, upper_mask):
        return torch.mean(nn.ReLU()(mask - upper_mask) + nn.ReLU()(lower_mask - mask))

    def forward(self, features):
        x = features.reshape(features.shape[0], -1).permute(1, 0)
        x = self.mlp(x)
        x = x.reshape(features.shape[1], features.shape[2], -1).permute(2, 0, 1)
        return x


def generate_mask(residual, threshold):
    inlier_pixel = (residual < threshold).float().unsqueeze(0).unsqueeze(0)
    window = torch.ones((1, 1, 3, 3), dtype=torch.float) / (3*3)
    if residual.is_cuda:
        window = window.cuda(residual.get_device())
    inlier_neighbors = F.conv2d(inlier_pixel, window, padding=1, groups=1)
    mask = (((inlier_neighbors > 0.5).float() + inlier_pixel) > 1e-3).float()
    return mask


def calculate_residual_mask(gt, render, cum_hist, lower_bound=0.6, upper_bound=0.8):
    residual_img = torch.abs(gt - render).clone().detach()
    residual = torch.mean(residual_img, dim=0)
    error_hist = torch.histogram(
        residual.cpu(),
        bins=10000,
        range=(0.0, 1.0)
    )[0].cuda()

    cum_hist = 0.95 * cum_hist + error_hist
    cum_error = torch.cumsum(cum_hist, dim=0)
    lower_error = torch.sum(cum_hist) * lower_bound
    upper_error = torch.sum(cum_hist) * upper_bound
    lower_threshold = torch.linspace(0, 1, 10001)[torch.where(cum_error >= lower_error)[0][0]]
    upper_threshold = torch.linspace(0, 1, 10001)[torch.where(cum_error >= upper_error)[0][0]]
    lower_mask = generate_mask(residual, lower_threshold)
    upper_mask = generate_mask(residual, upper_threshold)

    return lower_mask.squeeze(0), upper_mask.squeeze(0), cum_hist


def interpolation(tensor, height, width, type='bilinear'):
    return F.interpolate(tensor.cuda().unsqueeze(0), size=(height, width), mode=type).squeeze(0)