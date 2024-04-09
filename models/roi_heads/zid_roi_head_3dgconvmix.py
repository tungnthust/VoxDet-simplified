import torch

from ..utils.det_utils import bbox2result, bbox2roi
from .standard_roi_head import StandardRoIHead
from torch.autograd import grad
from .relation_3dtraj import Relate3DMix
from .discriminator import NetD
from .submodules import VGGPerceptualLoss
from models.utils.rpn_utils import BboxOverlaps2D
from torch.nn import functional as F
import torch.nn as nn

class ROIFeatureExtraction(nn.Module):
    def __init__(self, channels, spatial_size):
        super(ROIFeatureExtraction, self).__init__()
        self.extract = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * spatial_size ** 2, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024)
        )
    def forward(self, x):
        return F.normalize(self.extract(x), dim=1)
        
class ZidRoIHead3DGConvMix(StandardRoIHead):
    """OLN Box head.
    
    We take the top-scoring (e.g., well-centered) proposals from OLN-RPN and
    perform RoIAlign to extract the region features from each feature pyramid
    level. Then we linearize each region features and feed it through two fc
    layers, followed by two separate fc layers, one for bbox regression and the
    other for localization quality prediction. It is recommended to use IoU as
    the localization quality target in this stage. 
    """

    def __init__(self,
                 mode='det',
                 save_p1=False,
                 support_guidance=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(ZidRoIHead3DGConvMix, self).__init__(
                 bbox_roi_extractor=bbox_roi_extractor,
                 bbox_head=bbox_head,
                 mask_roi_extractor=mask_roi_extractor,
                 mask_head=mask_head,
                 shared_head=shared_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg)
        self.relate_3d = Relate3DMix(support_guidance, mode)
        if mode != 'det':
            self.netd = NetD()
            self.perceptual_loss = VGGPerceptualLoss()
            self.optimizer_d = torch.optim.Adam(self.netd.parameters(), lr=5e-5, betas=(0,0.999))
        self.save_p1 = save_p1
        if self.save_p1:
            self.p1_info = {}
        self.rot_su = ("rot_weight" in support_guidance.keys())
        self.roi_feat_extract = ROIFeatureExtraction(channels=256, spatial_size=7)

        if "rot_weight" in support_guidance.keys():
            self.rot_weight = support_guidance['rot_weight']
            self.rot_mode = support_guidance['rot_mode']
    def supp_query_contrastive_loss(self, supp_features, roi_features, labels, ious, temperature=0.2):
        labels = labels.T
        ious = ious.T
        similarity = torch.div(
            torch.matmul(roi_features, supp_features.T), temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()
        
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        num_valid = torch.sum(torch.max(labels, dim=1)[0] > 0)
        per_label_log_prob = (log_prob * labels * ious)[:num_valid].sum(1) / labels[:num_valid].sum(1)

        # keep = ious >= self.iou_threshold
        # per_label_log_prob = per_label_log_prob[keep]
        loss = -per_label_log_prob

        # coef = self._get_reweight_func(self.reweight_func)(ious)
        # coef = coef[keep]

        # loss = loss * coef
        return loss.mean()
        
    def constrastive_learning(self, roi_feats, supp_feats, rois, gt_bboxes, gt_categories):
        """
        Args:
            roi_feats (Tensor): [batch_size * num_rpn_per_img, 1000] 
            agg_supp_feats (list[Tensors]): list of support features [batch_size, #supports, 1000]
            rois (Tensor): [batch_size * num_rpn_per_img, 5] #5: img_id, x, y, xx, yy
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        Returns:
            dict[str, Tensor]: a dictionary of components
        """
        num_imgs = len(gt_bboxes)
        rpn = rois[:, 1:].reshape(num_imgs, -1, 4)
        roi_feats = roi_feats.reshape(num_imgs, -1, roi_feats.shape[1])
        iou_calculator = BboxOverlaps2D()
        similarities = []
        gt_similarities = []
        assigned_gt_bboxes = []
        loss_supp_query = 0
        gt_similarites = []
        gt_overlaps = []
        for i in range(num_imgs):
            overlaps = iou_calculator(gt_bboxes[i], rpn[i])
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)
            num_gts, num_bboxes = overlaps.shape[0], overlaps.shape[1]
            assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                                         -1,
                                                         dtype=torch.long)
            pos_inds = max_overlaps >= 0.7
            assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds]
            
            gt_similarity = torch.zeros_like(overlaps)
            for j, gt_idx in enumerate(assigned_gt_inds):
                if gt_idx >= 0:
                    gt_similarity[gt_idx, j] = 1.0
            neg_supp_feats = supp_feats.copy()
            neg_gt_categories = gt_categories.copy()
            neg_gt_categories.pop(i)
            neg_supp_feats.pop(i)
            for neg_idx in range(len(neg_gt_categories)):
                exclude_idx = list(range(len(neg_gt_categories[neg_idx])))
                
                for j in range(len(neg_gt_categories[neg_idx])):
                    if neg_gt_categories[neg_idx][j] in gt_categories[i]:
                        exclude_idx.remove(j)
                neg_supp_feats[neg_idx] = neg_supp_feats[neg_idx][exclude_idx]
            neg_supp_feats = torch.cat(neg_supp_feats)
            num_neg_supp = neg_supp_feats.shape[0]
            neg_overlaps = torch.zeros(num_neg_supp, overlaps.shape[1]).to(overlaps.device)
            neg_gt_similarity = torch.zeros(num_neg_supp, overlaps.shape[1]).to(overlaps.device)

            support_features = torch.cat([supp_feats[i], neg_supp_feats])
            gt_similarity = torch.cat([gt_similarity, neg_gt_similarity])
            overlaps = torch.cat([overlaps, neg_overlaps])
            
            # output_dict = dict(support_features=support_features,
            #                 roi_feats=roi_feats[i],
            #                 gt_similarity=gt_similarity,
            #                 overlaps=overlaps)

            # torch.save(output_dict, 'output.pt')
            loss_supp_query += self.supp_query_contrastive_loss(support_features, roi_feats[i], gt_similarity, overlaps)
            gt_similarites.append(gt_similarity)
            gt_overlaps.append(overlaps)
        # torch.save({'supp_feats': supp_feats,
        #             'roi_feats': roi_feats,
        #             'gt_similarites': gt_similarites,
        #             'gt_overlaps': gt_overlaps}, 'supquery.pt')
                   
        #     similarity = F.cosine_similarity(supp_feats[i].unsqueeze(1), roi_feats[i], dim=-1)
        #     gt_similarities.append(gt_similarity)
        #     similarities.append(similarity)
        #     assigned_gt_bboxes.append(assigned_gt_inds)
        # return dict(
        #     gt_similarities=gt_similarities,
        #     pred_similarities=similarities,
        #     assigned_gt_bboxes=assigned_gt_bboxes
        # )
        return loss_supp_query.mean()

    
    def supervised_contrastive_learning(self, features, labels, temperature=0.2):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [50, 1024]
            labels (tensor): shape of [M].  e.g., [50]
        """
        assert features.shape[0] == labels.shape[0]
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float()

        similarity = torch.div(
            torch.matmul(features, features.T), temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)

        # keep = ious >= self.iou_threshold
        # per_label_log_prob = per_label_log_prob[keep]
        loss = -per_label_log_prob

        # coef = self._get_reweight_func(self.reweight_func)(ious)
        # coef = coef[keep]

        # loss = loss * coef
        return loss.mean()
        
    
    def _bbox_forward(self, x, rois, p1_feats=None, p1_traj=None, p1_id=None, gt_bboxes=None, supp_feats=None,\
                      supp_roi_feats_contrastive=None, supp_labels=None, gt_categories=None):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        B = x[0].shape[0]
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        # test will pre save p1
        # sample = dict(bbox_feats=bbox_feats, support=support, rois=rois, gt_bboxes=gt_bboxes)
        
        roi_feats = self.roi_feat_extract(bbox_feats)
        # for i in range(B):
        #     supp_feats[i] = self.roi_feat_extract(supp_feats[i])
        # supp_roi_feats_contrastive = self.roi_feat_extract(supp_roi_feats_contrastive) 
        loss_supcon = self.supervised_contrastive_learning(supp_roi_feats_contrastive, supp_labels)

        loss_supp_query = self.constrastive_learning(roi_feats, supp_feats, rois, gt_bboxes, gt_categories)
        # torch.save({'supp_roi_feats_contrastive': supp_roi_feats_contrastive,
        #             'supp_labels': supp_labels}, 'supcon.pt')
        # torch.save(out, "out.pt")
        # cls_score = out['pred_similarities']
        # gt_cls_score = out['gt_similarities']
        # assigned_gt_bboxes = out['assigned_gt_bboxes']
        
        # print("BBOX_FEATS", bbox_feats.shape)
        # print("P1_FEATS", p1_feats.shape)
        # if not self.save_p1:
        #     rela, rot_vec = self.relate_3d(bbox_feats, p1_feats, p1_traj)
        # elif p1_id not in self.p1_info.keys():
        #     rela, voxel_support, rot_vec = self.relate_3d(bbox_feats, p1_feats, p1_traj, return_support=True)
        #     self.p1_info[p1_id] = {
        #         'p1_feats': p1_feats,
        #         'voxel_support': voxel_support
        #     }
        # else:
        #     p1_feats = self.p1_info[p1_id]['p1_feats']
        #     rela, rot_vec = self.relate_3d(bbox_feats, self.p1_info[p1_id]['voxel_support'])
        bbox_feat = {"ori": bbox_feats, 'support': p1_feats}
        bbox_pred, bbox_score, contra_logits = self.bbox_head(bbox_feat)

        bbox_results = dict(
            loss_supcon=loss_supcon, loss_supp_query=loss_supp_query, bbox_pred=bbox_pred, bbox_feats=bbox_feats,
            bbox_score=bbox_score, contra_logits=contra_logits)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, query_pose,
                            img_metas, p1_2D, supp_feats, gt_categories):
        """Run forward function and calculate loss for box head in training."""
        B = x[0].shape[0]
        rois = bbox2roi([res.bboxes for res in sampling_results])
        # print("ROI", rois.shape)
        # print("ROI", rois[0], rois[255], rois[256], rois[512])

        p1_rois = bbox2roi([b.unsqueeze(0) for b in p1_2D['box']])
        # print("P1 ROI", p1_rois.shape)
        p1_x = p1_2D['feat']
        p1_feats = self.bbox_roi_extractor(
            p1_x[:self.bbox_roi_extractor.num_inputs], p1_rois)
        p1_feats = p1_feats.reshape(B, -1, p1_feats.shape[1], p1_feats.shape[2], p1_feats.shape[3])
        supp_roi_feats = []
        supp_roi_feats_contrastive = []
        supp_labels = []
        for i in range(B):
            supp_rois = bbox2roi([b.unsqueeze(0) for b in supp_feats[i]['support_bbox']])
            # print("P1 ROI", p1_rois.shape)
            support_x = supp_feats[i]['support_feat']
            support_roi_feat = self.bbox_roi_extractor(
                support_x[:self.bbox_roi_extractor.num_inputs], supp_rois)
            num_gts = gt_bboxes[i].shape[0]
            support_roi_feat = self.roi_feat_extract(support_roi_feat)
            # support_roi_feat = support_roi_feat.reshape(num_gts, -1, support_roi_feat.shape[1], support_roi_feat.shape[2], support_roi_feat.shape[3])
            
            
            # support_roi_feat_flatten = self.roi_feat_extract(support_roi_feat.flatten(0,1))
            support_roi_feat_agg = F.normalize(torch.max(support_roi_feat.reshape(num_gts, -1, support_roi_feat.shape[1]), dim=1)[0], dim=1)
            num_views = 10
            supp_roi_feats.append(support_roi_feat_agg)
            supp_roi_feats_contrastive.append(torch.cat([support_roi_feat.reshape(num_gts, -1, support_roi_feat.shape[1]), support_roi_feat_agg.unsqueeze(1)], dim=1).flatten(0,1))
            supp_labels.append(gt_categories[i].repeat(num_views + 1, 1).transpose(1, 0).flatten())
        supp_roi_feats_contrastive = torch.cat(supp_roi_feats_contrastive)
        supp_labels = torch.cat(supp_labels)
        # torch.save(dict(supp_roi_feats_contrastive=supp_roi_feats_contrastive, supp_roi_feats=supp_roi_feats, supp_labels=supp_labels), "outt.pt")
            # print("SUPP FEAT", support_roi_feat.shape)
        bbox_results = self._bbox_forward(x, rois, p1_feats, p1_2D['traj'], gt_bboxes=gt_bboxes, supp_feats=supp_roi_feats,\
                                          supp_roi_feats_contrastive=supp_roi_feats_contrastive, supp_labels=supp_labels, gt_categories=gt_categories)
        # print("bbox_results", bbox_results['bbox_pred'].shape, bbox_results['bbox_score'].shape)
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        # print("Bbox target", bbox_targets[2].shape)

        loss_bbox = self.bbox_head.loss(bbox_results['loss_supcon'],
                                        bbox_results['loss_supp_query'],
                                        bbox_results['bbox_pred'], 
                                        bbox_results['bbox_score'],
                                        bbox_results['contra_logits'],
                                        rois,
                                        *bbox_targets)
        
        if self.rot_su:
            labels = bbox_targets[0].reshape(B, -1)==0
            assert torch.all(labels[:, 0])
            pose_m = bbox_results['rot_vec'].reshape(B, -1, 3, 4)[:, 0, :, :3]
            pose_gt = query_pose[:, 0, :, :3]
            if self.rot_mode == '6d':
                loss_rot = torch.abs(self.matrix_to_rotation_6d(pose_gt) - self.matrix_to_rotation_6d(pose_m))
                loss_rot = torch.mean(loss_rot)
            else:
                res = pose_gt @ pose_m.permute(0, 2, 1)
                loss_rot = torch.abs(torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(res.device)-res)
                loss_rot = torch.mean(loss_rot)
            loss_rot *= self.rot_weight
            loss_bbox.update(loss_rot=loss_rot)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           p1_id=None,
                           p1_2D=None,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        # RPN score
        B = x[0].shape[0]
        rpn_score = torch.cat([p[:, -1:] for p in proposals], 0)
        # topk
        ind = torch.topk(rpn_score, 500, dim=0)[1].squeeze()
        origin_proposals = proposals[0]
        new_proposals = origin_proposals[ind]
        proposals = [new_proposals]
        rpn_score = rpn_score[ind]
        rois = bbox2roi(proposals)
        
        if not self.save_p1:
            p1_rois = bbox2roi([b.unsqueeze(0) for b in p1_2D['box']])
            p1_x = p1_2D['feat']
            p1_feats = self.bbox_roi_extractor(
                p1_x[:self.bbox_roi_extractor.num_inputs], p1_rois)
            p1_feats = p1_feats.reshape(B, -1, p1_feats.shape[1], p1_feats.shape[2], p1_feats.shape[3])

            bbox_results = self._bbox_forward(x, rois, p1_feats, p1_2D['traj'])
        elif p1_id not in self.p1_info.keys():
            p1_rois = bbox2roi([b.unsqueeze(0) for b in p1_2D['box']])
            p1_x = p1_2D['feat']
            p1_feats = self.bbox_roi_extractor(
                p1_x[:self.bbox_roi_extractor.num_inputs], p1_rois)
            p1_feats = p1_feats.reshape(B, -1, p1_feats.shape[1], p1_feats.shape[2], p1_feats.shape[3])

            bbox_results = self._bbox_forward(x, rois, p1_feats, p1_2D['traj'], p1_id)
        else:
            bbox_results = self._bbox_forward(x, rois, p1_id=p1_id)

        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        bbox_score = bbox_results['bbox_score']
        contra_logits = bbox_results['contra_logits']

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_score = bbox_score.split(num_proposals_per_img, 0)          
        rpn_score = rpn_score.split(num_proposals_per_img, 0)
        contra_logits = contra_logits.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        inds = []
        raw_score = []
        for i in range(len(proposals)):
            det_bbox, det_label, ind = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                bbox_score[i],
                contra_logits[i],
                rpn_score[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            inds.append(ind)
            raw_score.append(cls_score[i])
        return det_bboxes, det_labels

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      query_pose,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      p1_2D=None,
                      supp_feats=None,
                      gt_categories=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i][:, :4], gt_bboxes[i], gt_bboxes_ignore[i],
                    torch.zeros_like(gt_labels[i]))
                # if (torch.sum(assign_result.gt_inds > 0) > 0):
                #     print(f"ROI Num assign {torch.sum(assign_result.gt_inds > 0)}")
                # # print(f"Proposal {proposal_list[i][:, :4][:50]}")
                # print(f"Assign_result: {str(assign_result)} {assign_result.gt_inds[:50]} {assign_result.labels[:50]} {assign_result.max_overlaps[:50]}") 
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
                # print(f"POS bbox {sampling_result.pos_bboxes.shape}, {sampling_result.pos_bboxes}")
                # print("GT bbox", sampling_result.pos_gt_bboxes)
                # print("Sampling results ROI", sampling_result.bboxes.shape)
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels, query_pose,
                                                    img_metas, p1_2D, supp_feats, gt_categories)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    p1_2D=None,
                    obj_id=None,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, obj_id, p1_2D, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        1)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
            # return det_bboxes[0][:, -1], bbox_results
            # for cam
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def matrix_to_rotation_6d(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
        by dropping the last row. Note that 6D representation is not unique.
        Args:
            matrix: batch of rotation matrices of size (*, 3, 3)

        Returns:
            6D rotation representation, of size (*, 6)

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """
        batch_dim = matrix.size()[:-2]
        return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

    def forward_train_recon(self,
                        p1_2D=None):
        losses = dict()
        # all_data, only use -3 layer from FPN
        all_imgs = p1_2D['img']
        all_feat = p1_2D['feat'][-3]
        N = 32 # for training this is hard-coded
        B, _, W, H = all_imgs.shape[0], all_imgs.shape[1], all_imgs.shape[3], all_imgs.shape[4]
        C, w, h = all_feat.shape[1], all_feat.shape[2], all_feat.shape[3]
        all_feat = all_feat.reshape(B, N, C, w, h)
        all_traj = p1_2D['traj']
        # sample recon id
        input_imgs = all_imgs[:, 0:32]
        input_feat = all_feat
        input_traj = all_traj[:, 0:32]
        output_imgs = all_imgs[:, 32:]
        total_n = all_imgs.shape[1]
        output_traj = all_traj[:, total_n:]
        output_traj = output_traj[:, 32:]
        # recon using voxels
        recon_img = self.relate_3d.recon(input_feat, input_traj, output_traj)
        res = output_imgs.flatten(0,1)-recon_img
        # mask = output_imgs!=0
        recon_loss = torch.abs(res)*10
        percep_loss = self.perceptual_loss(recon_img, output_imgs.flatten(0,1))*0.1
        # GAN loss
        # first train discriminator
        self.optimizer_d.zero_grad()
        l_d = self.train_netd(output_imgs, recon_img)
        self.netd.zero_grad()
        # then train generator
        output = self.netd(recon_img)
        gan_loss = -output.mean()*0.01
        losses = {
            'recon_loss': recon_loss.mean(),
            'percep_loss': percep_loss.mean(),
            'gan_loss': gan_loss
        }

        return losses

    def forward_test_recon(self,
                        p1_2D=None):
        output = dict()
        # all_data, only use -3 layer from FPN
        all_imgs = p1_2D['img']
        all_feat = p1_2D['feat'][-3]
        N = 32
        B, _, W, H = all_imgs.shape[0], all_imgs.shape[1], all_imgs.shape[3], all_imgs.shape[4]
        C, w, h = all_feat.shape[1], all_feat.shape[2], all_feat.shape[3]
        all_feat = all_feat.reshape(B, N, C, w, h)
        all_traj = p1_2D['traj']
        # sample recon id
        input_imgs = all_imgs[:, 0:32]
        input_feat = all_feat
        input_traj = all_traj[:, 0:32]
        output_imgs = all_imgs[:, 32:]
        total_n = all_imgs.shape[1]
        output_traj = all_traj[:, total_n:]
        output_traj = output_traj[:, 32:]
        # recon using voxels
        recon_img = self.relate_3d.recon(input_feat, input_traj, output_traj)
        output = {
            'input_imgs': input_imgs,
            'output_imgs': output_imgs,
            'recon_img': recon_img
        }

        return output

    def train_netd(self, real_images, fake_images):
        b, t, c, h, w = real_images.size()
        self.netd.zero_grad()
        ## train netd with real poses
        img_tensor = real_images.reshape(b*t, c, h, w)
        real = img_tensor.detach()  # only use n_gan_angles images
        output = self.netd(real)
        real_predict1 = output.mean() - 0.001 * (output ** 2).mean()
        error_real = -real_predict1
        error_real.backward()
        ## train netd with fake poses
        img_tensor = fake_images.view(b*t, c, h, w)
        fake = img_tensor.detach()  # only use n_gan_angles images
        output2 = self.netd(fake)
        error_fake = output2.mean()
        error_fake.backward()
        # calculate gradient penalty
        eps = torch.rand(b*t, 1, 1, 1).to(real.device)
        x_hat = eps * real.data + (1 - eps) * fake.data
        x_hat.requires_grad = True
        hat_predict = self.netd(x_hat)
        grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        ################################
        error_D = error_real + error_fake
        self.optimizer_d.step()
        return error_D

