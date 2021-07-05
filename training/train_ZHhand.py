import numpy as np
import torch
from tqdm import tqdm

from datasets.HumanPoseEstimation import HumanPoseEstimationDataset
from misc.utils import flip_tensor, flip_back, get_final_preds
from misc.visualization import save_images
from training.base_model import Train


class ZHHandTrain(Train):
    """
    The class provides a basic tool for training PoseResNet
    Most of the traning options are customizable.

    Extension of the base_Train_PoseResNet for the ZHhand dataset.
    The only method supposed to be directly called is `run()`.
    """

    def __init__(self,
                 exp_name,
                 ds_train,
                 ds_val,
                 epochs=210,
                 batch_size=32,
                 num_workers=4,
                 loss='JointsMSELoss',
                 lr=0.01,
                 lr_decay=True,
                 lr_decay_steps=(170, 200),
                 lr_decay_gamma=0.1,
                 optimizer='Adam',
                 weight_decay=0.,
                 momentum=0.9,
                 nesterov=False,
                 pretrained_weight_path=None,  # '../models/weights/mmPose_state_dict.pth'
                 checkpoint_path=None,
                 log_path='./logs',
                 use_tensorboard=True,
                 model_c=50,
                 model_nof_joints=21,
                 model_bn_momentum=0.1,
                 flip_test_images=False,
                 device=None):
        """
        Initializes a new Train object, if you want to know the means of those parameters,
        you can see the comments in the same part of ./training/Train.py

        the 'flip_test_image' is False, because the keypoints of hand not use the same flip rule
        just like body.
        """
        super(ZHHandTrain, self).__init__(
            exp_name=exp_name,
            ds_train=ds_train,
            ds_val=ds_val,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            loss=loss,
            lr=lr,
            lr_decay=lr_decay,
            lr_decay_steps=lr_decay_steps,
            lr_decay_gamma=lr_decay_gamma,
            optimizer=optimizer,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            pretrained_weight_path=pretrained_weight_path,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            use_tensorboard=use_tensorboard,
            model_c=model_c,
            model_nof_joints=model_nof_joints,
            model_bn_momentum=model_bn_momentum,
            flip_test_images=flip_test_images,
            device=device
        )

    def _train(self):
        num_samples = self.len_dl_train * self.batch_size
        all_preds = np.zeros((num_samples, self.model_nof_joints, 3), dtype=np.float32)
        all_boxes = np.zeros((num_samples, 6), dtype=np.float32)
        image_paths = []
        image_ids = []
        idx = 0

        self.model.train()
        for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_train, desc='Training')):
            image = image.to(self.device)
            target = target.to(self.device)
            target_weight = target_weight.to(self.device)

            self.optim.zero_grad()
            output = self.model(image)
            loss = self.loss_fn(output, target, target_weight)

            loss.backward()
            self.optim.step()

            # Evaluate accuracy
            accs, avg_acc, cnt, joints_preds, joints_target = \
                self.ds_train.evaluate_accuracy(output, target)

            # Original
            num_images = image.shape[0]

            # measure elapsed time
            c = joints_data['center'].numpy()
            s = joints_data['scale'].numpy()
            score = joints_data['score'].numpy()
            pixel_std = 200

            # Get predictions on the original images
            preds, maxvals = get_final_preds(True, output.detach(), c, s, pixel_std)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2].detach().cpu().numpy()
            all_preds[idx:idx + num_images, :, 2] = maxvals.squeeze(-1).detach().cpu().numpy()
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * pixel_std, 1)
            all_boxes[idx:idx + num_images, 5] = score
            # image_paths.extend(joints_data['imgPath'])  # todo use id to replace path
            image_ids.extend(joints_data['imgId'].tolist())

            idx += num_images

            self.mean_loss_train += loss.item()
            if self.use_tensorboard:
                self.summary_writer.add_scalar('train_loss', loss.item(),
                                               global_step=step + self.epoch * self.len_dl_train)
                if step == 0:
                    save_images(image, target, joints_target, output, joints_preds,
                                joints_data['joints_visibility'], self.summary_writer,
                                step=step + self.epoch * self.len_dl_train, prefix='train_')

        self.mean_loss_train /= len(self.dl_train)

        # COCO evaluation
        print('\n image_ids:\n', image_ids)
        print('\n Train AP/AR')
        self.train_accs, self.mean_mAP_train = self.ds_train.evaluate_overall_accuracy(
            all_preds, all_boxes, image_ids, output_dir=self.log_path)

    def _val(self):
        num_samples = len(self.ds_val)
        all_preds = np.zeros((num_samples, self.model_nof_joints, 3), dtype=np.float32)
        all_boxes = np.zeros((num_samples, 6), dtype=np.float32)
        image_paths = []
        image_ids = []
        idx = 0
        self.model.eval()
        with torch.no_grad():
            for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_val, desc='Validating')):
                image = image.to(self.device)
                target = target.to(self.device)
                target_weight = target_weight.to(self.device)

                output = self.model(image)

                if self.flip_test_images:
                    pass

                loss = self.loss_fn(output, target, target_weight)

                # Evaluate accuracy
                accs, avg_acc, cnt, joints_preds, joints_target = \
                    self.ds_val.evaluate_accuracy(output, target)

                num_images = image.shape[0]
                # measure elapsed time
                c = joints_data['center'].numpy()
                s = joints_data['scale'].numpy()
                score = joints_data['score'].numpy()
                pixel_std = 200

                # Get predictions on the original images
                preds, maxvals = get_final_preds(True, output.detach(), c, s, pixel_std)

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2].detach().cpu().numpy()
                all_preds[idx:idx + num_images, :, 2] = maxvals.squeeze(-1).detach().cpu().numpy()

                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s * pixel_std, 1)
                all_boxes[idx:idx + num_images, 5] = score
                # image_paths.extend(joints_data['imgPath'])
                image_ids.extend(joints_data['imgId'].tolist())

                idx += num_images

                self.mean_loss_val += loss.item()
                self.mean_acc_val += avg_acc.item()
                if self.use_tensorboard:
                    global_step = step + self.epoch * self.len_dl_val
                    self.summary_writer.add_scalar('val_loss', loss.item(), global_step=global_step)
                    self.summary_writer.add_scalar('val_acc', avg_acc.item(), global_step=global_step)

                    if step == 0:
                        save_images(image, target, joints_target, output, joints_preds,
                                    joints_data['joints_visibility'], self.summary_writer,
                                    step=global_step, prefix='val_')

        self.mean_loss_val /= len(self.dl_val)
        self.mean_acc_val /= len(self.dl_val)

        # COCO evaluation
        print('\n image_ids:\n', image_ids)
        print('\n Val AP/AR')
        self.val_accs, self.mean_mAP_val = self.ds_val.evaluate_overall_accuracy(
            all_preds, all_boxes, image_ids, output_dir=self.log_path
        )
