import logging
import os
import numpy as np
import random
import torch
import argparse
from config import cfg
from datasets.make_dataloader import make_eval_dataloader
from processor.processor_spd_reid import do_inference
from utils.metrics import R1_mAP_eval

os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)

from utils.logger import setup_logger
from model.make_model_SOCSNet import make_model
import tqdm
import warnings
warnings.filterwarnings("ignore")


def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

def Inference(cfg,
            model,
            val_loader,
            num_query,
			logger):

	device = "cuda"
	logger = logging.getLogger("SOCSNet.test")
	logger.info("Enter inferencing")

	evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

	evaluator.reset()

	if device:
		if torch.cuda.device_count() > 1:
			print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
			# model = nn.DataParallel(model)
		model.to(device)

	model.eval()
	img_path_list = []


	for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(tqdm.tqdm(val_loader)):
		flag_n = None
		if len(img.shape) == 6:
			n, B, T, C, H, W = img.shape
			flag_n = True
			img = img.view(-1, *img.shape[2:])
			camids = camids.repeat(1, B).view(B)
		with torch.no_grad():
			img = img.to(device)
			if cfg.MODEL.SIE_CAMERA:
				camids = camids.to(device)
			else:
				camids = None
			if cfg.MODEL.SIE_VIEW:
				target_view = target_view.to(device)
			else:
				target_view = None
			feat = model(img, cam_label=camids, view_label=target_view)
			if flag_n:
				feat = feat.mean(0, keepdim=True)
			evaluator.update((feat, pid, camid))

	cmc, mAP, _, _, _, _, _ = evaluator.compute()
	logger.info("Validation Results ")
	logger.info("mAP: {:.1%}".format(mAP))
	for r in [1, 5, 10, 20]:
		logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
	return cmc[0], cmc[4]

if __name__ == '__main__':

	#############################################
	# --> param load and init
	#############################################
	parser = argparse.ArgumentParser(description="ReID Baseline Training")
	parser.add_argument(
		"--config_file", default="./configs/spd_reid_lsvid.yml", help="path to config file", type=str
	)

	parser.add_argument("opts", help="Modify config options using the command-line", default=None,
	                    nargs=argparse.REMAINDER)
	# parser.add_argument("--local_rank", default=3, type=int)
	args = parser.parse_args()

	if args.config_file != "":
		cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	cfg.freeze()

	set_seed(cfg.SOLVER.SEED)

	if cfg.MODEL.DIST_TRAIN:
		torch.cuda.set_device(args.local_rank)

	output_dir = cfg.OUTPUT_DIR
	if output_dir and not os.path.exists(output_dir):
		os.makedirs(output_dir)

	logger = setup_logger("SOCSNet", output_dir, if_train=True)
	logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
	logger.info(args)

	if args.config_file != "":
		logger.info("Loaded configuration file {}".format(args.config_file))
		with open(args.config_file, 'r') as cf:
			config_str = "\n" + cf.read()
			logger.info(config_str)
	logger.info("Running with config:\n{}".format(cfg))

	if cfg.MODEL.DIST_TRAIN:
		torch.distributed.init_process_group(backend='nccl', init_method='env://')

	#############################################
	# --> 数据加载
	#############################################
	val_loader, num_query, num_classes, camera_num, view_num = make_eval_dataloader(
		cfg)

	model_path = cfg.TEST.WEIGHT
	model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

	# model.load_param(model_path)#cfg.TEST.WEIGHT
	model.load_state_dict(torch.load(model_path))

	Inference(cfg, model, val_loader, num_query, logger)
