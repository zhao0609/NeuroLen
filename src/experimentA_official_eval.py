#!/usr/bin/env python
"""MindEye2-style quantitative evaluation for Experiment A variants.

This keeps the metrics used in MindEye2 `final_evaluations.py`, but adds
explicit paths, JSON outputs, optional metric groups, and exits before the
notebook-only UMAP plotting section.
"""

import argparse
import json
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR / "generative_models"))

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import webdataset as wds
from accelerate import Accelerator
from scipy import stats
from sentence_transformers import SentenceTransformer, util
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim_fn
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from models import GNet8_Encoder

import utils


torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser("Experiment A official-style evaluation")
    parser.add_argument("--eval_root", default=os.environ.get("EVAL_ROOT", "outputs/evals"))
    parser.add_argument("--tables_root", default=os.environ.get("TABLES_ROOT", "outputs/tables"))
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--all_recons_path", default=None)
    parser.add_argument("--data_path", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--cache_dir", default=os.environ.get("HF_HOME", os.environ.get("DATA_ROOT", "data")))
    parser.add_argument("--subj", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_eval", type=int, default=0, help="<=0 means all.")
    parser.add_argument("--skip_braincorr", action="store_true")
    parser.add_argument("--skip_caption", action="store_true")
    parser.add_argument("--skip_swav", action="store_true")
    parser.add_argument("--sentence_model", default=os.environ.get("SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument(
        "--clip_b_text_model",
        default=os.environ.get("CLIP_B_PATH", "openai/clip-vit-base-patch32"),
    )
    parser.add_argument(
        "--clip_l_text_model",
        default=os.environ.get("CLIP_L_PATH", "openai/clip-vit-large-patch14"),
    )
    parser.add_argument(
        "--openclip_bigg_ckpt",
        default=os.environ.get("BIGG_CKPT", ""),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def topk_acc(sim, labels, k=1):
    return utils.topk(sim, labels, k=k).item()


@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, device, feature_layer=None):
    preds = model(torch.stack([preprocess(recon) for recon in all_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).detach().cpu().numpy()
        reals = reals.float().flatten(1).detach().cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).detach().cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).detach().cpu().numpy()
    r = np.corrcoef(reals, preds)
    r = r[: len(all_images), len(all_images) :]
    congruents = np.diag(r)
    success = r < congruents
    success_cnt = np.sum(success, 0)
    return float(np.mean(success_cnt) / (len(all_images) - 1))


def test_count(subj):
    if subj == 3:
        return 2371
    if subj == 4:
        return 2188
    if subj == 6:
        return 2371
    if subj == 8:
        return 2188
    return 3000


def load_test_voxels(data_path, subj):
    with h5py.File(f"{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5", "r") as f:
        betas = torch.tensor(f["betas"][:], dtype=torch.float32)
    num_voxels = betas.shape[-1]
    num_test = test_count(subj)
    test_url = f"{data_path}/wds/subj0{subj}/new_test/0.tar"

    def my_split_by_node(urls):
        return urls

    test_data = (
        wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node)
        .decode("torch")
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    )
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
    test_images_idx = []
    test_voxels = None
    for test_i, (behav, _past, _future, _old) in enumerate(test_dl):
        test_voxels = betas[behav[:, 0, 5].cpu().long()]
        test_images_idx = np.append(test_images_idx, behav[:, 0, 0].cpu().numpy())
    assert test_i == 0
    return num_voxels, test_voxels, test_images_idx.astype(int)


def brain_corr_metrics(args, all_recons, device, n):
    masks = {}
    with h5py.File(f"{args.cache_dir}/brain_region_masks.hdf5", "r") as f:
        subject_group = f[f"subj0{args.subj}"]
        for name in ["nsd_general", "V1", "V2", "V3", "V4", "higher_vis"]:
            masks[name] = subject_group[name][:]

    num_voxels, test_voxels, test_images_idx = load_test_voxels(args.data_path, args.subj)
    uniq_imgs = np.unique(test_images_idx)[:n]
    averaged = torch.zeros((len(uniq_imgs), num_voxels))
    for i, uniq_img in enumerate(uniq_imgs):
        locs = np.where(test_images_idx == uniq_img)[0]
        if len(locs) == 1:
            locs = locs.repeat(3)
        elif len(locs) == 2:
            locs = locs.repeat(2)[:3]
        averaged[i] = torch.mean(test_voxels[None, locs], dim=1)

    recon_list = [transforms.ToPILImage()(all_recons[i].detach().cpu()) for i in range(n)]
    from torchmetrics import PearsonCorrCoef

    gnet = GNet8_Encoder(device=device, subject=args.subj, model_path=f"{args.cache_dir}/gnet_multisubject.pt")
    pec = PearsonCorrCoef(num_outputs=len(recon_list))
    beta_primes = gnet.predict(recon_list)
    out = {}
    for region, mask in masks.items():
        score = pec(averaged[:, mask].moveaxis(0, 1), beta_primes[:, mask].moveaxis(0, 1))
        out[f"Brain Corr. {region}"] = float(torch.mean(score).item())
    return out


def image_metrics(args, all_images, all_recons, all_clipvoxels, all_blurryrecons, device):
    metrics = {}
    n = len(all_images)

    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version=args.openclip_bigg_ckpt
        if Path(args.openclip_bigg_ckpt).exists()
        else "laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
        cache_dir=args.cache_dir,
    ).to(device)
    clip_img_embedder.eval().requires_grad_(False)

    percent_correct_fwds, percent_correct_bwds = [], []
    rng = np.random.default_rng(args.seed)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for loop in tqdm(range(30), desc="retrieval"):
            size = min(300, n)
            random_samps = rng.choice(np.arange(n), size=size, replace=False)
            emb = clip_img_embedder(all_images[random_samps].to(device)).float().reshape(size, -1)
            emb_ = all_clipvoxels[random_samps].to(device).float().reshape(size, -1)
            emb = nn.functional.normalize(emb, dim=-1)
            emb_ = nn.functional.normalize(emb_, dim=-1)
            labels = torch.arange(size, device=device)
            bwd_sim = utils.batchwise_cosine_similarity(emb, emb_)
            fwd_sim = utils.batchwise_cosine_similarity(emb_, emb)
            percent_correct_fwds.append(topk_acc(fwd_sim, labels, k=1))
            percent_correct_bwds.append(topk_acc(bwd_sim, labels, k=1))
    metrics["FwdRetrieval"] = float(np.mean(percent_correct_fwds))
    metrics["BwdRetrieval"] = float(np.mean(percent_correct_bwds))
    metrics["FwdRetrieval_ci95_low"] = float(stats.norm.interval(0.95, loc=np.mean(percent_correct_fwds), scale=np.std(percent_correct_fwds) / np.sqrt(len(percent_correct_fwds)))[0])
    metrics["FwdRetrieval_ci95_high"] = float(stats.norm.interval(0.95, loc=np.mean(percent_correct_fwds), scale=np.std(percent_correct_fwds) / np.sqrt(len(percent_correct_fwds)))[1])
    metrics["BwdRetrieval_ci95_low"] = float(stats.norm.interval(0.95, loc=np.mean(percent_correct_bwds), scale=np.std(percent_correct_bwds) / np.sqrt(len(percent_correct_bwds)))[0])
    metrics["BwdRetrieval_ci95_high"] = float(stats.norm.interval(0.95, loc=np.mean(percent_correct_bwds), scale=np.std(percent_correct_bwds) / np.sqrt(len(percent_correct_bwds)))[1])

    pix_preprocess = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    img_flat = pix_preprocess(all_images).reshape(n, -1).cpu()
    rec_flat = pix_preprocess(all_recons).reshape(n, -1).cpu()
    metrics["PixCorr"] = float(np.mean([np.corrcoef(img_flat[i], rec_flat[i])[0][1] for i in range(n)]))

    img_gray = rgb2gray(pix_preprocess(all_images).permute((0, 2, 3, 1)).cpu())
    rec_gray = rgb2gray(pix_preprocess(all_recons).permute((0, 2, 3, 1)).cpu())
    metrics["SSIM"] = float(
        np.mean(
            [
                ssim_fn(rec, im, channel_axis=None, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
                for im, rec in tqdm(zip(img_gray, rec_gray), total=n, desc="ssim")
            ]
        )
    )

    from torchvision.models import AlexNet_Weights, Inception_V3_Weights, alexnet, efficientnet_b1, EfficientNet_B1_Weights, inception_v3

    alex_weights = AlexNet_Weights.IMAGENET1K_V1
    alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=["features.4", "features.11"]).to(device)
    alex_model.eval().requires_grad_(False)
    alex_preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    metrics["AlexNet(2)"] = two_way_identification(all_recons, all_images, alex_model, alex_preprocess, device, "features.4")
    metrics["AlexNet(5)"] = two_way_identification(all_recons, all_images, alex_model, alex_preprocess, device, "features.11")
    del alex_model
    torch.cuda.empty_cache()

    weights = Inception_V3_Weights.DEFAULT
    inc_model = create_feature_extractor(inception_v3(weights=weights), return_nodes=["avgpool"]).to(device)
    inc_model.eval().requires_grad_(False)
    inc_preprocess = transforms.Compose(
        [
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    metrics["InceptionV3"] = two_way_identification(all_recons, all_images, inc_model, inc_preprocess, device, "avgpool")
    del inc_model
    torch.cuda.empty_cache()

    import clip

    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ]
    )
    metrics["CLIP"] = two_way_identification(all_recons, all_images, clip_model.encode_image, clip_preprocess, device)
    del clip_model
    torch.cuda.empty_cache()

    import scipy as sp

    eff_model = create_feature_extractor(efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT), return_nodes=["avgpool"])
    eff_model.eval().requires_grad_(False)
    eff_preprocess = transforms.Compose(
        [
            transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    gt = eff_model(eff_preprocess(all_images))["avgpool"].reshape(n, -1).detach().cpu().numpy()
    fake = eff_model(eff_preprocess(all_recons))["avgpool"].reshape(n, -1).detach().cpu().numpy()
    metrics["EffNet-B"] = float(np.array([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(n)]).mean())
    del eff_model

    if not args.skip_swav:
        swav_model = torch.hub.load("facebookresearch/swav:main", "resnet50")
        swav_model = create_feature_extractor(swav_model, return_nodes=["avgpool"])
        swav_model.eval().requires_grad_(False)
        swav_preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        gt = swav_model(swav_preprocess(all_images))["avgpool"].reshape(n, -1).detach().cpu().numpy()
        fake = swav_model(swav_preprocess(all_recons))["avgpool"].reshape(n, -1).detach().cpu().numpy()
        metrics["SwAV"] = float(np.array([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(n)]).mean())

    metrics["BlurryPixCorr"] = float(utils.pixcorr(all_images, all_blurryrecons).item())
    return metrics


def caption_metrics(args, all_captions, all_predcaptions, all_git_generated_captions, device):
    metrics = {}
    n = min(len(all_captions), len(all_predcaptions), len(all_git_generated_captions))
    refs = [str(x) for x in all_captions[:n]]
    brain = [str(x) for x in all_predcaptions[:n]]
    img = [str(x) for x in all_git_generated_captions[:n]]

    try:
        import evaluate

        meteor = evaluate.load("meteor")
        rouge = evaluate.load("rouge")
        meteor_img_ref = meteor.compute(predictions=img, references=refs)
        meteor_brain_ref = meteor.compute(predictions=brain, references=refs)
        meteor_brain_img = meteor.compute(predictions=brain, references=img)
        rouge_img_ref = rouge.compute(predictions=img, references=refs)
        rouge_brain_ref = rouge.compute(predictions=brain, references=refs)
        rouge_brain_img = rouge.compute(predictions=brain, references=img)
        metrics.update(
            {
                "Meteor_img_ref": float(meteor_img_ref["meteor"]),
                "Meteor_brain_ref": float(meteor_brain_ref["meteor"]),
                "Meteor_brain_img": float(meteor_brain_img["meteor"]),
                "Meteor_relative": float(meteor_brain_img["meteor"] / meteor_img_ref["meteor"]),
                "Rouge1_img_ref": float(rouge_img_ref["rouge1"]),
                "Rouge1_brain_ref": float(rouge_brain_ref["rouge1"]),
                "Rouge1_brain_img": float(rouge_brain_img["rouge1"]),
                "Rouge1_relative": float(rouge_brain_img["rouge1"] / rouge_img_ref["rouge1"]),
                "RougeL_img_ref": float(rouge_img_ref["rougeL"]),
                "RougeL_brain_ref": float(rouge_brain_ref["rougeL"]),
                "RougeL_brain_img": float(rouge_brain_img["rougeL"]),
                "RougeL_relative": float(rouge_brain_img["rougeL"] / rouge_img_ref["rougeL"]),
            }
        )
    except Exception as exc:
        metrics["evaluate_package_error"] = repr(exc)

    sent_model = SentenceTransformer(args.sentence_model, device=device)
    with torch.no_grad():
        emb_brain = sent_model.encode(brain, convert_to_tensor=True)
        emb_refs = sent_model.encode(refs, convert_to_tensor=True)
        emb_img = sent_model.encode(img, convert_to_tensor=True)
    ss_brain_img = util.pytorch_cos_sim(emb_brain, emb_img).detach().cpu()
    ss_brain_ref = util.pytorch_cos_sim(emb_brain, emb_refs).detach().cpu()
    ss_img_ref = util.pytorch_cos_sim(emb_img, emb_refs).detach().cpu()
    metrics.update(
        {
            "Sentence_img_ref": float(ss_img_ref.diag().mean().item()),
            "Sentence_brain_ref": float(ss_brain_ref.diag().mean().item()),
            "Sentence_brain_img": float(ss_brain_img.diag().mean().item()),
            "Sentence_relative": float((ss_brain_img.diag().mean() / ss_img_ref.diag().mean()).item()),
        }
    )

    clip_text_models = [
        ("CLIP-B", Path(args.clip_b_text_model)),
        ("CLIP-L", Path(args.clip_l_text_model)),
    ]
    for name, model_path in clip_text_models:
        try:
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            model_clip = CLIPModel.from_pretrained(str(model_path), local_files_only=True).to(device)
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            with torch.no_grad():
                input_ids = tokenizer(brain, return_tensors="pt", padding=True, truncation=True).to(device)
                emb_brain = model_clip.get_text_features(**input_ids)
                input_ids = tokenizer(refs, return_tensors="pt", padding=True, truncation=True).to(device)
                emb_refs = model_clip.get_text_features(**input_ids)
                input_ids = tokenizer(img, return_tensors="pt", padding=True, truncation=True).to(device)
                emb_img = model_clip.get_text_features(**input_ids)
            sim_brain_img = util.pytorch_cos_sim(emb_brain, emb_img).detach().cpu()
            sim_brain_ref = util.pytorch_cos_sim(emb_brain, emb_refs).detach().cpu()
            sim_img_ref = util.pytorch_cos_sim(emb_img, emb_refs).detach().cpu()
            metrics.update(
                {
                    f"{name}_img_ref": float(sim_img_ref.diag().mean().item()),
                    f"{name}_brain_ref": float(sim_brain_ref.diag().mean().item()),
                    f"{name}_brain_img": float(sim_brain_img.diag().mean().item()),
                    f"{name}_relative": float((sim_brain_img.diag().mean() / sim_img_ref.diag().mean()).item()),
                }
            )
            del model_clip
            torch.cuda.empty_cache()
        except Exception as exc:
            metrics[f"{name}_error"] = repr(exc)
    return metrics


def main():
    args = parse_args()
    utils.seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    Accelerator(split_batches=False, mixed_precision="fp16")

    eval_root = Path(args.eval_root)
    tables_root = Path(args.tables_root)
    tables_root.mkdir(parents=True, exist_ok=True)
    model_dir = eval_root / args.model_name
    recons_path = Path(args.all_recons_path) if args.all_recons_path else model_dir / f"{args.model_name}_all_enhancedrecons.pt"

    all_images = torch.load(eval_root / "all_images.pt", map_location="cpu").float()
    all_captions = torch.load(eval_root / "all_captions.pt", map_location="cpu")
    all_recons = torch.load(recons_path, map_location="cpu").float()
    all_clipvoxels = torch.load(model_dir / f"{args.model_name}_all_clipvoxels.pt", map_location="cpu").float()
    all_blurryrecons = torch.load(model_dir / f"{args.model_name}_all_blurryrecons.pt", map_location="cpu").float()
    all_predcaptions = torch.load(model_dir / f"{args.model_name}_all_predcaptions.pt", map_location="cpu")

    n = min(len(all_images), len(all_recons), len(all_clipvoxels), len(all_blurryrecons), len(all_predcaptions))
    if args.num_eval > 0:
        n = min(n, args.num_eval)
    all_images = all_images[:n]
    all_recons = all_recons[:n]
    all_clipvoxels = all_clipvoxels[:n]
    all_blurryrecons = all_blurryrecons[:n]
    all_predcaptions = all_predcaptions[:n]

    imsize = 256
    if all_images.shape[-1] != imsize:
        all_images = transforms.Resize((imsize, imsize), antialias=True)(all_images).float()
    if all_recons.shape[-1] != imsize:
        all_recons = transforms.Resize((imsize, imsize), antialias=True)(all_recons).float()
    if all_blurryrecons.shape[-1] != imsize:
        all_blurryrecons = transforms.Resize((imsize, imsize), antialias=True)(all_blurryrecons).float()

    model_name_plus_suffix = f"{args.model_name}_all_enhancedrecons"
    if "enhanced" in model_name_plus_suffix:
        all_recons = all_recons * 0.75 + all_blurryrecons * 0.25

    img_metrics = image_metrics(args, all_images, all_recons, all_clipvoxels, all_blurryrecons, device)
    if not args.skip_braincorr:
        img_metrics.update(brain_corr_metrics(args, all_recons, device, n))

    metric_order = [
        "PixCorr",
        "SSIM",
        "AlexNet(2)",
        "AlexNet(5)",
        "InceptionV3",
        "CLIP",
        "EffNet-B",
        "SwAV",
        "FwdRetrieval",
        "BwdRetrieval",
        "Brain Corr. nsd_general",
        "Brain Corr. V1",
        "Brain Corr. V2",
        "Brain Corr. V3",
        "Brain Corr. V4",
        "Brain Corr. higher_vis",
        "BlurryPixCorr",
    ]
    image_rows = [{"Metric": k, "Value": img_metrics[k]} for k in metric_order if k in img_metrics]
    image_df = pd.DataFrame(image_rows)
    image_df.to_csv(tables_root / f"{model_name_plus_suffix}.csv", sep="\t", index=False)
    (tables_root / f"{model_name_plus_suffix}.json").write_text(json.dumps(img_metrics, indent=2), encoding="utf-8")
    print(image_df.to_string(index=False))

    if not args.skip_caption:
        git_caps_path = eval_root / "all_git_generated_captions.pt"
        if git_caps_path.exists():
            all_git_generated_captions = torch.load(git_caps_path, map_location="cpu")
            cap_metrics = caption_metrics(args, all_captions[:n], all_predcaptions[:n], all_git_generated_captions[:n], device)
            cap_df = pd.DataFrame.from_dict(cap_metrics, orient="index", columns=["Value"])
            cap_df.to_csv(tables_root / f"{model_name_plus_suffix}_caption_metrics.csv", sep="\t", index=True)
            (tables_root / f"{model_name_plus_suffix}_caption_metrics.json").write_text(
                json.dumps(cap_metrics, indent=2), encoding="utf-8"
            )
            print(cap_df.to_string())
        else:
            print(f"caption evaluation skipped; missing {git_caps_path}")


if __name__ == "__main__":
    main()
