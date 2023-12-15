import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import interpolate_dense_features, upscale_positions

def process_multiscale(image_input, computing_device, model, scales=[.5, 1, 2]):
    b, _, height_init, width_init = image_input.size()
    assert(b == 1)

    all_keypoints_data = torch.zeros([3, 0])
    all_descriptors_data = torch.zeros([
        model.dense_feature_extraction.num_channels, 0
    ])
    all_scores = torch.zeros(0)

    prev_dense_features = None
    ban = None
    for index, custom_scale in enumerate(scales):
        current_img = F.interpolate(
            image_input, scale_factor=custom_scale,
            mode='bilinear', align_corners=True
        )
        _, _, height_level, width_level = current_img.size()

        dense_feats = model.dense_feature_extraction(current_img)
        del current_img

        _, _, height, width = dense_feats.size()

        if prev_dense_features is not None:
            dense_feats += F.interpolate(
                prev_dense_features, size=[height, width],
                mode='bilinear', align_corners=True
            )
            del prev_dense_features

        detections = model.detection(dense_feats)
        if ban is not None:
            ban = F.interpolate(ban.float(), size=[height, width]).bool()
            detections = torch.min(detections, ~ban)
            ban = torch.max(
                torch.max(detections, dim=1)[0].unsqueeze(1), ban
            )
        else:
            ban = torch.max(detections, dim=1)[0].unsqueeze(1)
        fmap_positions = torch.nonzero(detections[0].cpu()).t()
        del detections

        displacements = model.localization(dense_feats)[0].cpu()
        displacements_i = displacements[
            0, fmap_positions[0, :], fmap_positions[1, :], fmap_positions[2, :]
        ]
        displacements_j = displacements[
            1, fmap_positions[0, :], fmap_positions[1, :], fmap_positions[2, :]
        ]
        del displacements

        mask = torch.min(
            torch.abs(displacements_i) < 0.5,
            torch.abs(displacements_j) < 0.5
        )
        fmap_positions = fmap_positions[:, mask]
        valid_displacements = torch.stack([
            displacements_i[mask],
            displacements_j[mask]
        ], dim=0)
        del mask, displacements_i, displacements_j

        fmap_keypoints = fmap_positions[1 :, :].float() + valid_displacements
        del valid_displacements

        raw_descs, _, ids = interpolate_dense_features(
            fmap_keypoints.to(computing_device),
            dense_feats[0]
        )

        fmap_positions = fmap_positions.to(computing_device)
        fmap_keypoints.to(computing_device)
        ids = ids.to(torch.device("cpu"))

        fmap_positions = fmap_positions[:, ids]
        fmap_keypoints = fmap_keypoints[:, ids]
        del ids

        keypoints_data = upscale_positions(fmap_keypoints, scaling_steps=2)
        del fmap_keypoints

        descriptors_data = F.normalize(raw_descs, dim=0).cpu()
        del raw_descs

        keypoints_data[0, :] *= height_init / height_level
        keypoints_data[1, :] *= width_init / width_level

        fmap_positions = fmap_positions.cpu()
        keypoints_data = keypoints_data.cpu()

        keypoints_data = torch.cat([
            keypoints_data,
            torch.ones([1, keypoints_data.size(1)]) * 1 / custom_scale,
        ], dim=0)

        scores_data = dense_feats[
            0, fmap_positions[0, :], fmap_positions[1, :], fmap_positions[2, :]
        ].cpu() / (index + 1)
        del fmap_positions

        all_keypoints_data = torch.cat([all_keypoints_data, keypoints_data], dim=1)
        all_descriptors_data = torch.cat([all_descriptors_data, descriptors_data], dim=1)
        all_scores = torch.cat([all_scores, scores_data], dim=0)
        del keypoints_data, descriptors_data

        prev_dense_features = dense_feats
        del dense_feats
    del prev_dense_features, ban

    keypoints_data = all_keypoints_data.t().numpy()
    del all_keypoints_data
    scores_data = all_scores.numpy()
    del all_scores
    descriptors_data = all_descriptors_data.t().numpy()
    del all_descriptors_data
    return keypoints_data, scores_data, descriptors_data
