import torch
import torch.nn.functional as F
import torch.distributions as dist
import math
import time
import random
import copy
import numpy as np
import cv2
import torchvision
from collections import OrderedDict
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
import ltr.data.bounding_box_utils as bbutils
from ltr.models.target_classifier.initializer import FilterInitializerZero
from ltr.models.layers import activation

from ..dimplt import DiMPLT


class MotionDiMPLT(DiMPLT):
    def track(self, image, info: dict = None):
        """
        info: {'lost': bool, 'motion_feedback'}
        """
        torch.cuda.synchronize()
        t_first = time.time()
        if not hasattr(self, '_cam_K') or not hasattr(self, '_cam_D'):
            if 'cam_params' in info:
                self._cam_K = info['cam_params']['K']
                self._cam_D = info['cam_params']['D']

        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        if self.frame_num == 2:
            self.search_global = False
            self.search_random = False
            self.redetection = False
            self.cnt_track = 0
            self.cnt_empty = 0

        # Convert image
        img_arr = image.copy()
        im = numpy_to_torch(image)
        # print('.')

        # set to re-detection mode, if external lost decider is True
        if info is not None and 'lost' in info and not self.redetection:
            if self.params.get('redetection_now', False) and info['lost']:
                self.search_random = True
                self.search_global = False
                self.redetection = True

        # [---------------- enhanced short-term tracking using random erasing (new) ----------------]
        if (self.search_global == False) and (self.search_random == False):  # previous frame -> found
            print('normal tracking')

            if self.params.get('erasing_mode', False) and self.cnt_track % self.params.get('erasing_cnt', 1) == 0:
                t_s = time.time()
                backbone_feat, sample_coords, im_patches = self.extract_backbone_features_with_erasing(im,
                                                                                                       self.get_centered_sample_pos(),
                                                                                                       self.target_scale * self.params.scale_factors,
                                                                                                       self.img_sample_sz)
                test_x = self.get_classification_features(
                    backbone_feat)  # Batch x 512 x 18 x 18
                sample_pos, sample_scales = self.get_sample_location(
                    sample_coords)
                scores_raw = self.classify_target(test_x)  # Batch x 19 x 19
                average_scores_raw = scores_raw.mean(dim=0).unsqueeze(0)
                translation_vec, scale_ind, s, flag = self.localize_target(
                    average_scores_raw, sample_pos, sample_scales)
                if self.params.get('use_original_pos', False):
                    original_scores_raw = scores_raw[-2:-1, :, :, :]
                    translation_vec, scale_ind, s, _ = self.localize_target(
                        original_scores_raw, sample_pos, sample_scales)
                new_pos = sample_pos[scale_ind, :] + translation_vec

                t_e = time.time()
                print('enhanced short-term tracking costs {} sec'.format(t_e - t_s))
            else:
                # original
                # Extract backbone features
                t_s = time.time()
                
                # time bottleneck
                tic = time.time()
                backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                                          self.target_scale * self.params.scale_factors,
                                                                                          self.img_sample_sz)
                toc = time.time()
                print('extract backbone feat cost {} sec'.format(toc - tic))
                
                # Extract classification features
                tic = time.time()
                test_x = self.get_classification_features(
                    backbone_feat)  # 512 x 18 x 18
                toc = time.time()
                print('extract cls feat cost {} sec'.format(toc - tic))

                # Location of sample
                tic = time.time()
                sample_pos, sample_scales = self.get_sample_location(
                    sample_coords)
                toc = time.time()
                print('loc sampling cost {} sec'.format(toc - tic))

                # Compute classification scores
                tic = time.time()
                scores_raw = self.classify_target(test_x)  # 19 x 19
                toc = time.time()
                print('cls target cost {} sec'.format(toc - tic))

                # Localize the target
                tic = time.time()
                translation_vec, scale_ind, s, flag = self.localize_target(
                    scores_raw, sample_pos, sample_scales)
                new_pos = sample_pos[scale_ind, :] + translation_vec
                toc = time.time()
                print('locate target cost {} sec'.format(toc - tic))

                t_e = time.time()
                print('original short-term tracking costs {} sec'.format(t_e - t_s))

            if self.params.get('redetection_now', False) and flag == 'not_found':
                self.search_global = False
                self.search_random = True
                self.redetection = True
                # self.cnt_empty = 1

        # [---------------- Global re-detection with random property (new) ----------------]
        t_s = time.time()
        if self.search_global or self.search_random:
            # find candidate (global: sliding window, random: random window
            print('re-detection')

            # print('random search')
            search_pos = self.pos + ((self.feature_sz + self.kernel_size) % 2) * \
                self.target_scale * self.img_support_sz / (2 * self.feature_sz)
            search_pos_sample = search_pos.clone()
            list_search_pos = [search_pos]

            # all_pos = self.find_all_index(search_pos.clone(), self.target_scale * self.img_sample_sz, self.image_sz,
            #                               self.params.get('redetection_global_search_flag', 1), 50, False)

            # find candidates with rearlight segmentation
            all_pos = self._sample_redetection_pos(
                img_arr, self.image_sz, num_limit=500,
                is_night=info['is_night'] if 'is_night' in info else False,
                vh_boxes=info['vh_boxes'] if 'vh_boxes' in info else None)
            if all_pos is None:
                all_pos = self.find_all_index(search_pos.clone(), self.target_scale * self.img_sample_sz, self.image_sz,
                                              self.params.get('redetection_global_search_flag', 1), 50, False)
            # all_pos_arr = np.array([pos.cpu().numpy() for pos in all_pos])
            # samples_3d = self._map_2d_to_3d(all_pos_arr)

            if self.search_random:
                print('random search')
                if self.params.get('additional_candidate_adaptive', False):
                    num_add = int(self.params.get(
                        'additional_candidate_adaptive_ratio', 0.33) * len(all_pos))
                    if num_add < self.params.get('additional_candidate_adaptive_min', 1):
                        num_add = self.params.get(
                            'additional_candidate_adaptive_min', 1)
                    if num_add > self.params.get('additional_candidate_adaptive_max', 10):
                        num_add = self.params.get(
                            'additional_candidate_adaptive_max', 10)

                else:
                    num_add = self.params.get('additional_candidate_random', 0)
                idx_remain = [x for i, x in enumerate(np.random.permutation(
                    len(all_pos))) if i < num_add]  # random n value
                idx_remain = sorted(idx_remain)
            else:
                print('global search')
                if self.params.get('global_search_memory_limit', 10000) < len(all_pos):
                    num_add = self.params.get(
                        'global_search_memory_limit', 10000)
                    idx_remain = [x for i, x in enumerate(np.random.permutation(
                        len(all_pos))) if i < num_add]  # random n value
                    idx_remain = sorted(idx_remain)
                else:
                    idx_remain = [x for x in range(len(all_pos))]

            for i in idx_remain:
                list_search_pos.append(all_pos[i])

            flag_batch = True

            if flag_batch:  # batch version
                backbone_feat, sample_coords, im_patches = self.extract_backbone_features_with_multiple_search(
                    im, list_search_pos, self.target_scale * self.params.scale_factors, self.img_sample_sz)
                test_x = self.get_classification_features(
                    backbone_feat)  # 512 x 18 x 18
                sample_pos, sample_scales = self.get_sample_location(
                    sample_coords)
                scores_raw = self.classify_target(test_x)  # 19 x 19

                if self.params.get('redetection_score_penalty', False):
                    # dist1 = np.zeros(len(list_search_pos))
                    # dist_max = math.sqrt(sum(self.image_sz ** 2))
                    # for i in range(len(list_search_pos)):
                    #     dist1[i] = math.sqrt(
                    #         sum((list_search_pos[i] - list_search_pos[0]) ** 2))
                    # weight_penalty = 1.0 - self.params.get('redetection_score_penalty_alpha', 0.5) * (dist1 / dist_max) * math.exp(
                    #     - self.params.get('redetection_score_penalty_beta', 0.5) * (self.cnt_empty - 1))
                    # for i in range(len(scores_raw)):
                    #     scores_raw[i] *= weight_penalty[i]
                    if 'motion_feedback' in info and info['motion_feedback'] is not None:
                        feedback_2d = self._map_3d_to_2d(np.expand_dims(
                            info['motion_feedback'], axis=0))[0]   # (2,)
                        dist_max = math.sqrt(sum(self.image_sz ** 2))
                        penalty_normal_dist = dist.Normal(torch.Tensor(
                            [0.0]), torch.Tensor([300.0]))
                        norm_scale = 1.0 / \
                            (penalty_normal_dist.log_prob(
                                torch.Tensor([0.0])).exp())
                        u_diff = torch.Tensor(
                            [pos[1] - feedback_2d[0] for pos in list_search_pos])
                        weight_penalty = norm_scale * \
                            penalty_normal_dist.log_prob(u_diff).exp()
                        # print('u_diff: ', u_diff)
                        # print('weight_penalty: ', weight_penalty)
                        for i in range(len(scores_raw)):
                            scores_raw[i] *= weight_penalty[i]

                        # # plot
                        # for i in range(len(list_search_pos)):
                        #     cv2.circle(img_arr, (int(list_search_pos[i][1].item()), int(list_search_pos[i][0].item())), radius=3,
                        #                color=(255, 0, 0), thickness=4)
                        # cv2.circle(img_arr, (int(feedback_2d[0]), int(feedback_2d[1])), radius=3, color=(0, 0, 255),
                        #            thickness=4)
                        # cv2.imshow('debug', img_arr)
                        # cv2.waitKey(1)

                if self.redetection:
                    for i in range(len(scores_raw)):
                        if i != 0:  # not original
                            scores_raw[i] *= self.params.get(
                                'redetection_basic_penalty', 1.0)

                translation_vec, scale_ind, s, flag = self.localize_target(
                    scores_raw, sample_pos, sample_scales)
                new_pos = sample_pos[scale_ind, :] + translation_vec
                # print('flag: {}'.format(flag))

                s = torch.unsqueeze(s[scale_ind], 0)
                sample_pos = torch.unsqueeze(sample_pos[scale_ind], 0)
                scores_raw = torch.unsqueeze(scores_raw[scale_ind], 0)
                sample_scales = torch.unsqueeze(sample_scales[scale_ind], 0)
                sample_coords = torch.unsqueeze(sample_coords[scale_ind], 0)
                test_x = torch.unsqueeze(test_x[scale_ind], 0)

                backbone_feat_new = OrderedDict()
                for key, value in backbone_feat.items():
                    backbone_feat_new[key] = torch.unsqueeze(
                        value[scale_ind], 0)
                backbone_feat = backbone_feat_new

                if scale_ind.is_cuda:
                    scale_ind = torch.tensor(0).cuda()
                else:
                    scale_ind = torch.tensor(0)

            else:  # not batch version
                for i in range(len(list_search_pos)):
                    if i == 0:  # original search
                        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(
                            im, list_search_pos[i], self.target_scale * self.params.scale_factors, self.img_sample_sz)
                        test_x = self.get_classification_features(
                            backbone_feat)  # 512 x 18 x 18
                        sample_pos, sample_scales = self.get_sample_location(
                            sample_coords)
                        scores_raw = self.classify_target(test_x)  # 19 x 19
                        translation_vec, scale_ind, s, flag = self.localize_target(
                            scores_raw, sample_pos, sample_scales)
                        new_pos = sample_pos[scale_ind, :] + translation_vec

                        list_backbone_feat = [backbone_feat]
                        list_sample_coords = [sample_coords]
                        list_test_x = [test_x]
                        list_sample_pos = [sample_pos]
                        list_sample_scales = [sample_scales]
                        list_scale_ind = [scale_ind]
                        list_s = [s]
                        list_flag = [flag]
                        list_new_pos = [new_pos]

                    else:  # random search
                        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(
                            im, list_search_pos[i], self.target_scale * self.params.scale_factors, self.img_sample_sz)
                        test_x = self.get_classification_features(
                            backbone_feat)  # 512 x 18 x 18
                        sample_pos, sample_scales = self.get_sample_location(
                            sample_coords)
                        scores_raw = self.classify_target(test_x)  # 19 x 19

                        if self.params.get('redetection_score_penalty', False):
                            dist1 = math.sqrt(
                                sum((list_search_pos[i] - list_search_pos[0])**2))
                            dist_max = math.sqrt(sum(self.image_sz**2))
                            weight_penalty = 1.0 - self.params.get('redetection_score_penalty_alpha', 0.5) * (
                                dist1 / dist_max) * math.exp(- self.params.get('redetection_score_penalty_beta', 0.5) * (self.cnt_empty - 1))
                            scores_raw *= weight_penalty
                            # print(weight_penalty)

                        if self.redetection:
                            scores_raw *= self.params.get(
                                'redetection_basic_penalty', 1.0)

                        translation_vec, scale_ind, s, flag = self.localize_target(
                            scores_raw, sample_pos, sample_scales)
                        new_pos = sample_pos[scale_ind, :] + translation_vec

                        list_backbone_feat.append(backbone_feat)
                        list_sample_coords.append(sample_coords)
                        list_test_x.append(test_x)
                        list_sample_pos.append(sample_pos)
                        list_sample_scales.append(sample_scales)
                        list_scale_ind.append(scale_ind)
                        list_s.append(s)
                        list_flag.append(flag)
                        list_new_pos.append(new_pos)

                # find max s
                for i in range(len(list_search_pos)):
                    local_s = list_s[i]
                    local_scale_ind = list_scale_ind[i]
                    score_map = local_s[local_scale_ind, ...]
                    max_score = torch.max(score_map).item()  # confidence
                    if i == 0:
                        list_max_score = [max_score]
                    else:
                        list_max_score.append(max_score)
                find_i = list_max_score.index(max(list_max_score))

                backbone_feat = list_backbone_feat[find_i]
                sample_coords = list_sample_coords[find_i]
                test_x = list_test_x[find_i]
                sample_pos = list_sample_pos[find_i]
                sample_scales = list_sample_scales[find_i]
                scale_ind = list_scale_ind[find_i]
                s = list_s[find_i]
                flag = list_flag[find_i]
                new_pos = list_new_pos[find_i]
        t_e = time.time()
        print('re-detection costs {} sec'.format(t_e - t_s))

        # Update position and scale
        t_s = time.time()
        if flag != 'not_found':
            if self.params.get('use_iou_net', True):
                update_scale_flag = self.params.get(
                    'update_scale_when_uncertain', True) or flag != 'uncertain'
                if self.params.get('use_classifier', True):
                    self.update_state(new_pos)
                self.refine_target_box(
                    backbone_feat, sample_pos[scale_ind, :], sample_scales[scale_ind], scale_ind, update_scale_flag)
            elif self.params.get('use_classifier', True):
                self.update_state(new_pos, sample_scales[scale_ind])
        t_e = time.time()
        print('update position and scale costs {} sec'.format(t_e - t_s))

        # ------- UPDATE ------- #
        t_s = time.time()
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get(
            'hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False):
            # early sample (redetected) is not updated
            if self.cnt_track >= self.params.get('no_update_early_redetection', 0):
                # <Original>
                # Get train sample
                train_x = test_x[scale_ind:scale_ind+1, ...]

                # Create target_box and label for spatial sample
                target_box = self.get_iounet_box(
                    self.pos, self.target_sz, sample_pos[scale_ind, :], sample_scales[scale_ind])

                # <Additional>
                # [------------------------ more discriminative feature learning (new)-------------------------]
                if self.params.get('track_net_more_learn', False) and s[scale_ind, ...].max().item() > self.params.get('track_net_more_learn_score', True) and \
                        (self.params.get('track_net_more_learn_not_save', True) or (self.cnt_track % self.params.get('track_net_more_learn_cnt', 1) == 0) and self.cnt_track > 0):  # first track (not used)
                    # conditions
                    # 1) score should be bigger than track_net_more_learn_score
                    # 2) track_net_more_learn is True
                    # 3) track_net_more_learn_not_save is True or track_net_more_learn_cnt condition is satisfied

                    more_search_pos = self.pos + ((self.feature_sz + self.kernel_size) %
                                                  2) * self.target_scale * self.img_support_sz / (2 * self.feature_sz)
                    list_more_search_pos = []
                    all_pos = self.find_all_index(more_search_pos.clone(), self.target_scale * self.img_sample_sz, self.image_sz,
                                                  self.params.get('track_net_more_learn_search_flag', 1), self.params.get('train_more_sample_limit', 5), False)
                    for i in reversed(range(len(all_pos))):

                        row_min = int(all_pos[i][0]) - \
                            int((self.target_sz[0] - 1) / 2)
                        row_max = int(all_pos[i][0]) + \
                            int((self.target_sz[0] - 1) / 2)
                        col_min = int(all_pos[i][1]) - \
                            int((self.target_sz[1] - 1) / 2)
                        col_max = int(all_pos[i][1]) + \
                            int((self.target_sz[1] - 1) / 2)

                        if (row_min < 0) or (col_min < 0) or (row_max >= int(self.image_sz[0])) or (col_max >= int(self.image_sz[1])):
                            del(all_pos[i])

                    num_add = self.params.get('additional_train_candidate', 0)
                    idx_remain = [x for i, x in enumerate(np.random.permutation(
                        len(all_pos))) if i < num_add]  # random n value
                    idx_remain = sorted(idx_remain)

                    for i in idx_remain:
                        list_more_search_pos.append(all_pos[i])

                    if len(list_more_search_pos) > 0:
                        # image masking
                        row_min = int(self.pos[0]) - \
                            int((self.target_sz[0] - 1) / 2)
                        row_max = int(self.pos[0]) + \
                            int((self.target_sz[0] - 1) / 2)
                        col_min = int(self.pos[1]) - \
                            int((self.target_sz[1] - 1) / 2)
                        col_max = int(self.pos[1]) + \
                            int((self.target_sz[1] - 1) / 2)
                        im_mask = im.clone()
                        im_target = im_mask[:, :, row_min:row_max +
                                            1, col_min:col_max + 1].clone()
                        im_mask[:, :, row_min:row_max +
                                1, col_min:col_max + 1] = 0
                        # self.tensor_image_save(im_mask[0], "im")

                        flag_contour_all = []
                        for i in range(len(list_more_search_pos)):
                            im_mask, flag_contour = self.im_masking(im_mask, im_target, list_more_search_pos[i], self.image_sz, [
                                                                    im_target.shape[2], im_target.shape[3]])
                            flag_contour_all.append(flag_contour)
                        # self.tensor_image_save(im_mask[0], "im_after")

                        backbone_feat_more, sample_coords_more, _ = self.extract_backbone_features_with_multiple_search(
                            im, list_more_search_pos, self.target_scale * self.params.scale_factors, self.img_sample_sz)
                        train_x_more = self.get_classification_features(
                            backbone_feat_more)
                        sample_pos_more, sample_scales_more = self.get_sample_location(
                            sample_coords_more)
                        target_box_more = []
                        for i in range(len(sample_pos_more)):
                            target_box_more_local = self.get_iounet_box(
                                list_more_search_pos[i], self.target_sz, sample_pos_more[i], sample_scales_more[i])
                            target_box_more_local = torch.unsqueeze(
                                target_box_more_local, dim=0)
                            if i == 0:
                                target_box_more = target_box_more_local
                            else:
                                target_box_more = torch.cat(
                                    (target_box_more, target_box_more_local), dim=0)

                        # Update the classifier model
                        self.update_classifier_more(
                            train_x, target_box, learning_rate, s[scale_ind, ...], train_x_more, target_box_more)
                    else:
                        self.update_classifier(
                            train_x, target_box, learning_rate, s[scale_ind, ...])
                else:
                    self.update_classifier(
                        train_x, target_box, learning_rate, s[scale_ind, ...])
        t_e = time.time()
        print('update classifier cost {} sec'.format(t_e - t_s))

        # Set the pos of the tracker to iounet pos
        if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()

        score_map = s[scale_ind, ...]
        max_score = torch.max(score_map).item()  # confidence

        # Visualize and set debug info
        self.search_area_box = torch.cat((sample_coords[scale_ind, [
                                         1, 0]], sample_coords[scale_ind, [3, 2]] - sample_coords[scale_ind, [1, 0]] - 1))
        self.debug_info['flag' + self.id_str] = flag
        self.debug_info['max_score' + self.id_str] = max_score
        if self.visdom is not None:
            self.visdom.register(score_map, 'heatmap', 2,
                                 'Score Map' + self.id_str)
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')
        elif self.params.debug >= 2:
            show_tensor(
                score_map, 5, title='Max score = {:.2f}'.format(max_score))

        # Compute output bounding box
        new_state = torch.cat(
            (self.pos[[1, 0]] - (self.target_sz[[1, 0]]-1)/2, self.target_sz[[1, 0]]))

        if self.params.get('output_not_found_box', False) and flag == 'not_found':
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        # ---------------- [re-detection flag] ----------------#
        if flag == 'not_found' and self.params.get('re_detection', False):
            self.cnt_track = 0
            if self.old_flag == 'not_found':  # continuous
                self.cnt_empty += 1
                self.redetection = True
            else:  # first not_found
                self.cnt_empty = 1
                self.redetection = True
                # self.terminate_pos =
            # print('self.cnt_empty: {}, self.pos: {}, self.feature_sz: {}, self.kernel_size: {}, self.target_scale: {}, self.img_support_sz: {}'.format(self.cnt_empty, self.pos, self.feature_sz, self.kernel_size, self.target_scale, self.img_support_sz))
            # print('.')
            if self.cnt_empty == 1:
                self.search_global = False
                self.search_random = True
            elif self.cnt_empty % self.params.get('cnt_global', 10000000) == 0:
                self.search_global = True
                self.search_random = False
            elif self.cnt_empty % self.params.get('cnt_random', 10000000) == 0:
                self.search_global = False
                self.search_random = True
            else:
                self.search_global = False
                self.search_random = False
        else:
            self.cnt_empty = 0
            self.search_global = False
            self.search_random = False
            self.redetection = False
            if self.old_flag == 'not_found':
                self.cnt_track = 1
            else:
                self.cnt_track += 1
        # print('[Frame: {}] previous state: {} / present state: {} / empty count: {} / track count: {}'.format(self.frame_num, self.old_flag, flag, self.cnt_empty, self.cnt_track))
        self.old_flag = flag

        # # ---------------- [confidence] ----------------#
        # flag_confidence = self.params.get('flag_confidence', 'none')
        # if flag_confidence == 1:  # basic
        #     confidence = max_score
        # elif flag_confidence == 2:  # upper/lower limit
        #     if max_score > 1:
        #         confidence = 1.0
        #     elif max_score < 0:
        #         confidence = 0.0
        #     else:
        #         confidence = max_score
        # elif flag_confidence == 3:  # consider not_found
        #     if max_score > 1:
        #         confidence = 1.0
        #     elif flag == 'not_found' or max_score < 0:
        #         confidence = 0.0
        #     else:
        #         confidence = max_score
        # elif flag_confidence == 4:  # 1.0 fix
        #     confidence = 1.0
        # elif flag_confidence == 5:  # 1 or 0
        #     if flag == 'not_found':
        #         confidence = 0.0
        #     else:
        #         confidence = 1.0
        # elif flag_confidence == 6:  # 1 or 0
        #     if flag == 'not_found':
        #         confidence = 0.0
        #     elif flag == 'uncertain':
        #         confidence = 0.5
        #     else:
        #         confidence = 1.0
        # elif flag_confidence == 7:  # 1 or 0
        #     if flag == 'not_found':
        #         confidence = 0.0
        #     elif flag == 'uncertain':
        #         confidence = 0.66
        #     elif flag == 'hard_negative':
        #         confidence = 0.33
        #     else:
        #         confidence = 1.0
        # elif flag_confidence == 8:  # 1 or 0
        #     if flag == 'not_found':
        #         confidence = 0.0
        #     elif flag == 'hard_negative':
        #         confidence = 0.66
        #     elif flag == 'uncertain':
        #         confidence = 0.33
        #     else:
        #         confidence = 1.0
        # elif flag_confidence == 9:
        #     if flag == 'not_found':
        #         confidence = 0.0
        #     elif flag == 'hard_negative':
        #         confidence = 0.5
        #     else:
        #         confidence = 1.0
        # else:
        #     confidence = max_score

        # print(flag)

        torch.cuda.synchronize()
        t_last = time.time()
        out = {'target_bbox': output_state,
               #    'confidence': confidence,
               'score': max_score,
               'time': t_last - t_first}

        return out

    def _sample_redetection_pos(self, image, image_size, num_limit=200, is_night=False, vh_boxes=None):
        seg_candidates_stats, seg_candidates_centroids = self._rearlight_seg(
            image, is_night=is_night, vh_boxes=vh_boxes)
        n_comp = seg_candidates_stats.shape[0]
        if n_comp == 0:  # sampling fail
            return None

        mix = dist.Categorical(torch.ones((n_comp,)))
        seg_candidates_stats = seg_candidates_stats.astype(float)
        seg_candidates_centroids = seg_candidates_centroids.astype(float)
        # (y, x) in OpenCV, (x, y) in torch
        means = torch.from_numpy(seg_candidates_centroids[:, (1, 0)])
        hw = seg_candidates_stats[:, (3, 2)]
        covs = torch.from_numpy(hw / 6)  # 3-sigma
        comp = dist.Independent(dist.Normal(means, covs), 1)
        gmm = dist.MixtureSameFamily(mix, comp)

        # sample candidates
        sample_pos = gmm.sample((num_limit,)).type(torch.IntTensor)
        valid_sample_mask_1 = torch.logical_and(
            sample_pos[:, 0] >= 0, sample_pos[:, 0] < image_size[0])
        valid_sample_mask_2 = torch.logical_and(
            sample_pos[:, 1] >= 0, sample_pos[:, 1] < image_size[1])
        valid_sample_mask = torch.logical_and(
            valid_sample_mask_1, valid_sample_mask_2)
        sample_pos = sample_pos[valid_sample_mask]

        return list(sample_pos)

    def _map_3d_to_2d(self, points_3d: np.ndarray):
        """_summary_
        Args:
            points_3d (np.ndarray): shape (N, 3), in camera coordinates
                    Z^
                     |    <
                     |   + X
                     |  +
                     | +
                     |+
            <--------+
            Y         O
        return: np.ndarray, shape (N, 2), in image coordinates
        """
        # convert to OpenCV camera coordinate
        points_3d_cv = np.empty_like(points_3d)
        points_3d_cv[:, 0] = -points_3d[:, 1]
        points_3d_cv[:, 1] = -points_3d[:, 2]
        points_3d_cv[:, 2] = points_3d[:, 0]
        points_3d_cv = points_3d_cv.reshape((1, -1, 3))

        rvec = tvec = np.zeros((1, 1, 3))
        points_2d, _ = cv2.fisheye.projectPoints(
            points_3d_cv, rvec, tvec, self._cam_K, self._cam_D)

        return points_2d[0]

    def _map_2d_to_3d(self, points_2d: np.ndarray):
        """
        points_2d: point in image coordinates. x as row index and y as column index
        """
        # convert to OpenCV image coordinates
        points_2d_cv = np.empty_like(points_2d)
        points_2d_cv[:, 0] = points_2d[:, 1]
        points_2d_cv[:, 1] = points_2d[:, 0]

        points_undistort = cv2.fisheye.undistortPoints(
            points_2d_cv, self._cam_K, self._cam_D)
        f_x = self._cam_K[0, 0]
        f_y = self._cam_K[1, 1]
        c_x = self._cam_K[0, 2]
        c_y = self._cam_K[1, 2]
        points_undistort[:, 0] = (points_undistort[:, 0] - c_x) / f_x
        points_undistort[:, 1] = (points_undistort[:, 1] - c_y) / f_y

        points_3d = np.empty((points_undistort.shape[0], 3))
        points_3d[:, 0] = 1.0  # X = 1.0
        points_3d[:, 1] = -points_undistort[:, 0]  # Y = -x
        points_3d[:, 2] = -points_undistort[:, 1]  # Z = -Y

        return points_3d

    def _rearlight_seg(self, image, vh_boxes=None, is_night: bool = False):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask = None
        if is_night:
            red_mask_1 = cv2.inRange(img_hsv, (0, 50, 150), (15, 255, 255))
            red_mask_2 = cv2.inRange(img_hsv, (165, 50, 150), (180, 255, 255))
            red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        else:
            red_mask_1 = cv2.inRange(img_hsv, (0, 70, 20), (15, 255, 255))
            red_mask_2 = cv2.inRange(img_hsv, (165, 70, 20), (180, 255, 255))
            red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(
            red_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
        mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        if vh_boxes is not None:
            for box in vh_boxes:
                vh_mask = np.zeros(
                    (image.shape[0], image.shape[1]), dtype=np.uint8)
                vh_mask[int(box[1]): int(box[1] + box[3]),
                        int(box[0]): int(box[0] + box[2])] = 255
                mask = cv2.bitwise_and(mask, vh_mask)

        # img_disp = image.copy()
        # red = cv2.bitwise_and(img_disp, img_disp, mask=mask)
        # name = np.random.rand()
        # cv2.imwrite(
        #     '/media/wuhr/data/platoon_dataset/2022_10_20/evaluation/2d/vis/{}.png'.format(
        #         name), img_disp)
        # cv2.imwrite(
        #     '/media/wuhr/data/platoon_dataset/2022_10_20/evaluation/2d/vis/red_{}.png'.format(
        #         name), red)
        # cv2.imshow('red', red)
        # cv2.waitKey(1)

        connected_commponents = cv2.connectedComponentsWithStats(
            mask, connectivity=8, ltype=cv2.CV_32S)
        stats = connected_commponents[2]  # (x, y, w, h, area)
        centroids = connected_commponents[3].astype(np.int)
        keep_idx = np.where((stats[:, 4] > 40) & (stats[:, 4] < 5000))
        keep_idx = list(set(keep_idx[0]))
        if 0 in keep_idx:
            keep_idx.remove(0)

        return stats[keep_idx, :], centroids[keep_idx, :]

    def localize_target_redet(self, scores, sample_pos, sample_scales, weight_penalty=None):
        """Run the target localization."""

        scores = scores.squeeze(1)

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(
                self.net.classifier.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = activation.softmax_reg(
                scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('score_filter_ksz', 1)
        if score_filter_ksz > 1:
            assert score_filter_ksz % 2 == 1
            kernel = scores.new_ones(1, 1, score_filter_ksz, score_filter_ksz)
            scores = F.conv2d(
                scores.view(-1, 1, *scores.shape[-2:]), kernel, padding=score_filter_ksz//2).view(scores.shape)

        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind, ...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * \
            (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None

    def localize_advanced_redet(self, scores, sample_pos, sample_scales, weight_penalty=None):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind, ...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * \
            (self.img_support_sz / output_sz) * sample_scale

        # print('maxscore: {}'.format(max_score1.item()))
        if max_score1.item() < self.params.target_not_found_threshold:  # 0.25
            return translation_vec1, scale_ind, scores_hn, 'not_found'
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):  # x
            return translation_vec1, scale_ind, scores_hn, 'uncertain'
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):  # x
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * \
            (self.target_sz / sample_scale) * (output_sz / self.img_support_sz)

        tneigh_top = max(
            round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(
            round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(
            round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(
            round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[..., tneigh_top:tneigh_bottom,
                      tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * \
            (self.img_support_sz / output_sz) * sample_scale

        prev_target_vec = (self.pos - sample_pos[scale_ind, :]) / (
            (self.img_support_sz / output_sz) * sample_scale)

        # Handle the different cases
        # similar sample (0.8 * max1)
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(
                torch.sum((target_disp1-prev_target_vec)**2))
            disp_norm2 = torch.sqrt(
                torch.sum((target_disp2-prev_target_vec)**2))
            disp_threshold = self.params.dispalcement_scale * \
                math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain'

        # similar sample (0.5 * max1) and bigger than not_found_th(0.25)
        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        return translation_vec1, scale_ind, scores_hn, 'normal'
