# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim

import math
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, modified_render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.mask_utils import DINOFeatureExtractor, MLPModel, calculate_residual_mask, interpolation, DINOv3FeatureExtractor
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
from einops import reduce, repeat, rearrange

import copy
import json
import random
import omegaconf
import numpy as np
from torch.utils.data import DataLoader
from lact_utils.data_inference import MaskNVSDataset
from lact_utils.pipe.model import LaCTLVSM

from pytorch_msssim import SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
metric_ssim = SSIM(data_range=1.0, size_average=True, channel=3)
metric_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


############### add
def capture(self):
    return (
        self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        self.xyz_gradient_accum,
        self.denom,
    )

@torch.no_grad()
def _means2D_from_cam_mats(cam, gaussians, H: int, W: int, device=None):

    if device is None:
        device = gaussians.get_xyz.device

    X = gaussians.get_xyz.detach().to(device)             # [N,3]
    ones = torch.ones((X.shape[0], 1), device=device)
    Xh = torch.cat([X, ones], dim=1)                      # [N,4]

    W2V = cam.world_view_transform.to(device)
    PROJ = cam.projection_matrix.to(device)
    M = (W2V @ PROJ)

    clip = Xh @ M
    w = clip[:, 3:4].clamp_min(1e-9)
    ndc = clip[:, :3] / w

    u = (ndc[:, 0] * 0.5 + 0.5) * (W - 1)
    # v = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (H - 1)
    v = (ndc[:, 1] * 0.5 + 0.5) * (H - 1)     # test

    uv = torch.stack([u, v], dim=-1)                      # [N,2]
    valid = (w[:,0] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return uv, valid

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, dataname=args.dataname)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    coarse_stack = scene.getTrainCameras(4).copy()

    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    mask = None

    # Prepare for mask estimation
    if args.use_dinov3:
        feature_extractor = DINOv3FeatureExtractor().cuda()
    else:
        feature_extractor = DINOFeatureExtractor().cuda()
    features_fine, features_coarse = {}, {}
    if not opt.disable_mask:
        for cam in tqdm(scene.getTrainCameras(), desc=f"DINOv2 GT Feature Extraction"):
            features_fine[cam.image_name] = feature_extractor(cam.original_image, opt.upper_feat_res).cpu()
            features_coarse[cam.image_name] = feature_extractor(cam.original_image, opt.lower_feat_res).cpu()

        mlp_model = MLPModel().to(device="cuda")
        mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=args.mask_lr)
        historical_hist = torch.zeros((10000)).cuda()

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Scheduler step
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if (iteration % 1000 == 0):
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            coarse_stack = scene.getTrainCameras(4).copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        else:
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            coarse_cam = coarse_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if (args.exp_iter_start <= iteration):
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # coarse scale rendering
        if not opt.disable_mask and iteration < opt.bootstrap_iter:
            coarse_image = render(coarse_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
            coarse_gt = coarse_cam.original_image.cuda()

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        gt_image = viewpoint_cam.original_image.cuda()

        # MLP eval for masked loss calculation
        if opt.disable_mask or iteration < opt.mask_beginning:
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                Lssim = 1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                Lssim = 1.0 - ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim

        elif args.use_inpaint and (args.use_inpaint_iter < iteration <= args.use_inpaint_iter_end):
            inpaint_img = inpainted_dict.get(viewpoint_cam.image_name, None)
            mlp_model.eval()
            upsample_feature = interpolation(features_fine[viewpoint_cam.image_name], image.shape[1], image.shape[2])
            mask = mlp_model(upsample_feature)      # 1: known, 0: unknown
            loss_mask = mask.clone().detach() > 0.25
            loss_mask = -F.max_pool2d(-(loss_mask.float().unsqueeze(0)), kernel_size=7, stride=1, padding=3).squeeze(0)

            if inpaint_img is not None:
                warmup_steps = 2000
                current_offset = iteration - args.use_inpaint_iter
                inpaint_weight = min(1.0, 0.2 + (0.8 * current_offset / warmup_steps))
                Ll1 = inpaint_weight * (loss_mask * torch.abs((image - inpaint_img))).mean()
                Lssim = inpaint_weight * (1.0 - ssim((loss_mask * image), (loss_mask * inpaint_img), size_average=False)).mean()
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
                mask = None
            else:
                Ll1 = (loss_mask * torch.abs((image - gt_image))).mean()
                Lssim = (1.0 - ssim((loss_mask * image), (loss_mask * gt_image), size_average=False)).mean()
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim

        elif args.use_inpaint and (iteration > args.use_inpaint_iter_end):
            mlp_model.eval()
            upsample_feature = interpolation(features_fine[viewpoint_cam.image_name], image.shape[1], image.shape[2])
            mask = mlp_model(upsample_feature)
            loss_mask = mask.clone().detach() > 0.25
            loss_mask = -F.max_pool2d(-(loss_mask.float().unsqueeze(0)), kernel_size=7, stride=1, padding=3).squeeze(0)
            
            inpaint_img = inpainted_dict.get(viewpoint_cam.image_name, None)
            if inpaint_img is not None:
                inp_mask = inpaint_mask_dict[viewpoint_cam.image_name]
                loss_mask *= inp_mask
                mask = None

            Ll1 = (loss_mask * torch.abs((image - gt_image))).mean()
            Lssim = (1.0 - ssim((loss_mask * image), (loss_mask * gt_image), size_average=False)).mean()
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
            
        else:
            mlp_model.eval()
            upsample_feature = interpolation(features_fine[viewpoint_cam.image_name], image.shape[1], image.shape[2])
            mask = mlp_model(upsample_feature)

            loss_mask = mask.clone().detach() > 0.25
            loss_mask = -F.max_pool2d(-(loss_mask.float().unsqueeze(0)), kernel_size=7, stride=1, padding=3).squeeze(0)

            Ll1 = (loss_mask * torch.abs((image - gt_image))).mean()
            Lssim = (1.0 - ssim((loss_mask * image), (loss_mask * gt_image), size_average=False)).mean()
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        # MLP training
        reset_start = iteration // opt.opacity_reset_interval * opt.opacity_reset_interval
        reset_end = reset_start + 300
        if (mask is not None) and not opt.disable_mask and iteration >= opt.mask_beginning and (not((iteration>reset_start) and (iteration<reset_end) and (iteration>=opt.reset_iter))):
            mlp_model.train()
            if iteration < opt.bootstrap_iter:
                gt_feature = features_coarse[viewpoint_cam.image_name].cuda()
                render_feature = feature_extractor(image.detach(),opt.lower_feat_res) 
                lower_mask, upper_mask, historical_hist = calculate_residual_mask(coarse_gt, coarse_image, historical_hist)
            else:
                gt_feature = features_fine[viewpoint_cam.image_name].cuda()
                render_feature = feature_extractor(image.detach(),opt.upper_feat_res) 
                lower_mask, upper_mask, historical_hist = calculate_residual_mask(gt_image, image, historical_hist)
            lower_mask = interpolation(lower_mask, image.shape[1], image.shape[2])
            upper_mask = interpolation(upper_mask, image.shape[1], image.shape[2])

            cosine = (1.- F.cosine_similarity(gt_feature, render_feature, dim=0).unsqueeze(0).sub(0.5).div(0.5)).clip(0.,1.)
            cosine = 1. - interpolation(cosine, image.shape[1], image.shape[2])

            reg_loss = 0.5 * mlp_model.get_regularizer()
            reg_loss += 2.0 * ((1-mask) * math.exp(-iteration / opt.beta_reg)).mean()
            residual_loss = mlp_model.get_residual_loss(mask.flatten(), lower_mask.flatten(), upper_mask.flatten())

            mask_loss = args.w_cos * torch.abs(mask - cosine).mean() + args.w_res * residual_loss + reg_loss 
            mask_loss.backward()
            
            mlp_optimizer.step()
            mlp_optimizer.zero_grad(set_to_none=True)
        iter_end.record()

        # Selection based FisherRF
        if (args.use_inpaint and (iteration == args.use_inpaint_iter)):

            original_viewpoint_cam = scene.getTrainCameras().copy()
            original_gaussians = copy.deepcopy(gaussians)

            # FisherRF
            params = capture(original_gaussians)[1:7]
            name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}

            filter_out_idx = [name2idx[k] for k in ["rotation", "scale", "xyz", "opacity"]]
            params = [p.requires_grad_(True) for i, p in enumerate(params) if i not in filter_out_idx]
            _optim = torch.optim.SGD(params, 0.)
            original_gaussians.optimizer = _optim
            device = params[0].device

            H_per_gaussian = torch.zeros(params[0].shape[0], device=device, dtype=params[0].dtype)
            per_view_influence = []   # list of tensors shaped [N]
            per_view_names = []       # list of view image names
            inpaint_mask_dict = dict()

            mlp_model.eval()
            for idx, cam in enumerate(tqdm(original_viewpoint_cam, desc="Accumulating Fisher across views")):
                mod_render_pkg = modified_render(cam, original_gaussians, pipe, background)
                pred_img = mod_render_pkg["render"]

                # MLP eval & Save inpainting mask
                upsample_feature = interpolation(features_fine[cam.image_name], image.shape[1], image.shape[2])
                mask = mlp_model(upsample_feature)
                mask = mask.clone().detach() > 0.25
                mask = -F.max_pool2d(-(mask.float().unsqueeze(0)), kernel_size=7, stride=1, padding=3).squeeze(0)
                inpaint_mask_dict[cam.image_name] = mask

                pred_img *= mask
                pred_img.backward(gradient=torch.ones_like(pred_img))

                with torch.no_grad():
                    g_per_gauss_abs = sum([reduce(p.grad.detach().abs(), "n ... -> n", "sum") for p in params])
                    per_view_influence.append(g_per_gauss_abs)
                    per_view_names.append(cam.image_name)
                H_per_gaussian += sum([reduce(p.grad.detach(), "n ... -> n", "sum") for p in params])
                _optim.zero_grad(set_to_none = True)
                
            if len(per_view_influence) > 0:
                clean_influence_mat = torch.stack(per_view_influence, dim=0)  # [V, N]

            per_view_influence = []   # list of tensors shaped [N]
            per_view_names = []       # list of view image names
            for idx, cam in enumerate(tqdm(original_viewpoint_cam, desc="Accumulating Fisher across views")):
                mod_render_pkg = modified_render(cam, original_gaussians, pipe, background)
                pred_img = mod_render_pkg["render"]
                
                pred_img *= (1-inpaint_mask_dict[cam.image_name])
                pred_img.backward(gradient=torch.ones_like(pred_img))

                with torch.no_grad():
                    g_per_gauss_abs = sum([reduce(p.grad.detach().abs(), "n ... -> n", "sum") for p in params])
                    per_view_influence.append(g_per_gauss_abs)
                    per_view_names.append(cam.image_name)
                H_per_gaussian += sum([reduce(p.grad.detach(), "n ... -> n", "sum") for p in params])
                _optim.zero_grad(set_to_none = True)
                
            if len(per_view_influence) > 0:
                distract_influence_mat = torch.stack(per_view_influence, dim=0)  # [V, N]

            valid_mat = distract_influence_mat > args.tau_min

            scene_scale = scene.cameras_extent
            dynamic_sigma = scene_scale * 0.2
            min_fisher_ratio = 0.01
            relative_threshold = args.fisherrf_thr

            selection_dict = dict()
            N = gaussians.get_xyz.shape[0]
            per_view_names = [cam.image_name for cam in original_viewpoint_cam]

            for idx, cam in enumerate(tqdm(original_viewpoint_cam, desc="Scoring Views")):
                target_name = cam.image_name
                inpainted_mask = inpaint_mask_dict[target_name]
                H, W = inpainted_mask.shape[-2:]
                
                c_i = cam.camera_center.to(device)
                v_i = torch.from_numpy(cam.R[2, :]).to(device) # Z-axis (Forward vector)

                means2D, _ = _means2D_from_cam_mats(cam, gaussians, H, W, device=inpainted_mask.device)
                u1, v1 = means2D[:, 0], means2D[:, 1]
                valid = torch.isfinite(u1) & torch.isfinite(v1) & (u1 >= -0.5) & (u1 <= W - 0.5) & (v1 >= -0.5) & (v1 <= H - 0.5)
                
                xi = u1[valid].round().clamp(0, W - 1).long()
                yi = v1[valid].round().clamp(0, H - 1).long()
                
                mask_inp = (1 - inpainted_mask).float()
                dil_flat = mask_inp.view(-1)
                sel = (dil_flat[yi * W + xi] > 0.5)
                
                S_mask = torch.zeros(N, dtype=torch.bool, device=u1.device)
                S_mask[valid] = sel
                val_gs_mask = (valid_mat[idx] * S_mask)

                val_gs_bool = clean_influence_mat[:, val_gs_mask] > 0 

                all_scores = []
                for j, ref_cam in enumerate(original_viewpoint_cam):
                    if idx == j: continue
                    
                    intersection_count = val_gs_bool[j].sum()
                    fisher_ratio = intersection_count / (val_gs_bool.shape[1] + 1e-8)
                    
                    if fisher_ratio < min_fisher_ratio:
                        continue

                    c_j = ref_cam.camera_center.to(device)
                    dist = torch.norm(c_i - c_j)
                    spatial_weight = torch.exp(-dist / dynamic_sigma)
                    
                    v_j = torch.from_numpy(ref_cam.R[2, :]).to(device)
                    angular_sim = torch.clamp(torch.dot(v_i, v_j), min=0.0)
                    angular_weight = angular_sim ** 2 
                    
                    final_score = (fisher_ratio * spatial_weight * angular_weight).item()
                    
                    if final_score > args.final_score:
                        all_scores.append({'name': per_view_names[j], 'score': final_score})

                if len(all_scores) > 0:
                    all_scores = sorted(all_scores, key=lambda x: x['score'], reverse=True)
                    max_score = all_scores[0]['score']
                    reliable_views = [
                        item for item in all_scores 
                        if item['score'] >= (max_score * relative_threshold)
                    ]
                    top_k = min(len(reliable_views), 10) 
                    selection_dict[target_name] = {
                        item['name']: item['score'] for item in reliable_views[:top_k]
                    }
                else:
                    selection_dict[target_name] = {}

            with open(f"{scene.model_path}/selection_all.txt", "w", encoding="utf-8") as f:
                import json
                json.dump(selection_dict, f, ensure_ascii=False, indent=2)

            # LaCT eval
            # Model configuration
            model_config = omegaconf.OmegaConf.load(args.lact_config)
                
            # Seed everything
            seed = 42
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # LaCT model
            lact_model = LaCTLVSM(**model_config).cuda()
            lact_checkpoint = torch.load(args.lact_ckpt, map_location="cpu")
            lact_model.load_state_dict(lact_checkpoint["model"])

            # Dataset
            lact_dataset = MaskNVSDataset(
                args.source_path,
                args.num_views,
                mask_dict=inpaint_mask_dict,
                mask_dilate=args.mask_dilate,
                select_strategy=args.select_strategy,
                selection_dict=selection_dict,
                seed=seed,)

            if lact_dataset is None:
                break

            # Dataloader
            dataloader_seed_generator = torch.Generator()
            dataloader_seed_generator.manual_seed(seed)
            lact_dataloader = DataLoader(
                lact_dataset,
                batch_size=1,
                shuffle=False,
                generator=dataloader_seed_generator,    # This ensures deterministic dataloader
            )

            # Eval
            inpainted_dict = dict()
            for data_dict in lact_dataloader:
                name_list = data_dict["name"]
                data_dict = {key: value.cuda() for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
                for idx in tqdm(range(data_dict["image"].shape[1] // args.num_views), desc="LaCT inpainting:"):
                    input_indices = torch.arange(idx * args.num_views, (idx + 1) * args.num_views)
                    target_data_dict = {k: v[:, input_indices[0]:input_indices[0]+1] for k, v in data_dict.items()}
                    input_masked_data_dict = {key: value[:, input_indices] for key, value in data_dict.items()}
                    
                    with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True) and torch.no_grad():
                        rendering = lact_model(input_masked_data_dict, target_data_dict, args.lact_original)
                    out = F.interpolate(rendering.squeeze(0), size=(H,W), mode='bicubic', align_corners=False).squeeze(0)
                    inpainted_dict[name_list[input_indices[0].item()][0]] = out.clamp(0,1)

                    np_out = out.permute(1,2,0).detach().cpu().numpy().clip(0,1)
                    save_path = f"{scene.model_path}/lact_results"
                    os.makedirs(save_path, exist_ok=True)
                    plt.imsave(f"{save_path}/{name_list[input_indices[0].item()][0]}", np_out)                    

                    # Feature update
                    features_fine[name_list[input_indices[0].item()][0]] = feature_extractor(out.clip(0,1), opt.upper_feat_res).cpu()
                    features_coarse[name_list[input_indices[0].item()][0]] = feature_extractor(out.clip(0,1), opt.lower_feat_res).cpu()

            historical_hist = torch.zeros((10000)).cuda()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, Lssim,
                            iter_start.elapsed_time(iter_end), testing_iterations, scene, render,
                            (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if args.use_inpaint_iter < iteration < args.use_inpaint_iter + 5000:
                        current_threshold = opt.densify_grad_threshold * 0.5
                        current_min_opacity = 0.01 
                    else:
                        current_threshold = opt.densify_grad_threshold
                        current_min_opacity = 0.005
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(current_threshold, current_min_opacity, scene.cameras_extent, None, radii)

                # Reset
                if iteration >= opt.reset_iter and iteration < opt.iterations and iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            if iteration == args.use_inpaint_iter:
                gaussians.reset_opacity()

            if args.use_inpaint_iter < iteration < args.use_inpaint_iter + 5000:
                if iteration % 1000 == 0:
                    gaussians.reset_opacity()

            if iteration == opt.iterations - 3000:
                gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if args.save_mask and (iteration % 1000 == 0):
                mask_viewpoint_cam = scene.getTrainCameras().copy()
                print("\n[ITER {}] Saving Mask Image".format(iteration))
                for mask_cam in mask_viewpoint_cam:
                    
                    mlp_model.eval()
                    save_mask_fname = mask_cam.image_name.replace('.JPG','.png').replace('.jpg','.png')
                    upsample_feature = interpolation(features_fine[mask_cam.image_name], mask_cam.original_image.shape[1], mask_cam.original_image.shape[2])
                    save_mask = mlp_model(upsample_feature)
                    save_mask = save_mask.clone().detach() > 0.25
                    save_mask = -F.max_pool2d(-(save_mask.float().unsqueeze(0)), kernel_size=7, stride=1, padding=3).squeeze(0)

                    image_with_mask = torch.cat([(mask_cam.original_image*255).permute(1, 2, 0), (save_mask*255).permute(1, 2, 0)], dim=2).cpu().numpy().astype(np.uint8)
                    mask_save_path = scene.model_path + f"/mask_{iteration}/{save_mask_fname}"
                    os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                    plt.imsave(mask_save_path, image_with_mask)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, Lssim,
                    elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', Lssim.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += metric_ssim(image[None,...], gt_image[None,...]).mean().double()
                    lpips_test += metric_lpips(image[None,...], gt_image[None,...]).mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000, 40_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--test_iterations", nargs="+", type=int,
        default=[1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000,
                11_000, 12_000, 13_000, 14_000, 15_000, 16_000, 17_000, 18_000, 19_000, 20_000,
                21_000, 22_000, 23_000, 24_000, 25_000, 26_000, 27_000, 28_000, 29_000, 30_000,
                31_000, 32_000, 33_000, 34_000, 35_000, 36_000, 37_000, 38_000, 39_000, 40_000])
                
    # general
    parser.add_argument("--use_inpaint", action='store_true')
    parser.add_argument("--use_inpaint_iter", type=int, default=10_000)
    parser.add_argument("--use_inpaint_iter_end", type=int, default=90_000)
    parser.add_argument("--save_mask", action='store_true')

    # FisherRF
    parser.add_argument("--fisherrf_thr", type=float, default=0.0)
    parser.add_argument("--use_rel_fisherrf", action='store_true')

    # LaCT / TTT
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--mask_dilate", type=int, default=15)
    parser.add_argument("--select_strategy", type=str, default="select_knn")
    parser.add_argument("--lact_config", type=str, default="lact_utils/config/lact_l24_d768_ttt2x.yaml")
    parser.add_argument("--lact_ckpt", type=str, default="lact_utils/ckpts/model_0003000.pth")
    parser.add_argument("--lact_original", action='store_true')

    # Exposure
    parser.add_argument("--exp_iter_start", type=int, default=3000)

    # Loss weights
    parser.add_argument("--w_cos", type=float, default=0.5)
    parser.add_argument("--w_res", type=float, default=0.5)

    # DINO backbone
    parser.add_argument("--use_dinov3", action='store_true')

    # Mask MLP
    parser.add_argument("--mask_lr", type=float, default=0.001)
    parser.add_argument("--mask_val_num", type=float, default=10.0)

    # Dataset variant
    parser.add_argument("--dataname", type=str, default='wg')

    # Fisher-RF thresholds
    parser.add_argument("--tau_min", type=float, default=0.1)
    parser.add_argument("--final_score", type=float, default=0.3)

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")