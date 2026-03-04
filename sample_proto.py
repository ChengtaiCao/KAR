import torch
import time
import math
import numpy as np
from sklearn.cluster import KMeans
from src.helper_functions.IncrementalDataset import ModifiableSubset


def _forward_features(model, train_loader, buffer_loader, low_range, high_range, buffer_threshold=0.9):
    """
    Calculate Features for Sampling
    """
    features = []
    targets = []
    predictions = []
    model.eval()
    for _, (image, target) in enumerate(train_loader):
        image = image.cuda(non_blocking=True)
        with torch.no_grad():
            feature = model(image)['pool_embeddings'].cpu()
            features.append(feature)
            targets.append(target)
            prediction = torch.sigmoid(model(image)['logits']).cpu()
            predictions.append(prediction)
            torch.cuda.empty_cache()
    
    if buffer_loader:
        for _, (image, target) in enumerate(buffer_loader):
            image = image.cuda(non_blocking=True)
            with torch.no_grad():
                feature = model(image)['pool_embeddings'].cpu()
                logit = torch.sigmoid(model(image)['logits']).cpu()
                buffer_data_mask = (target[:, low_range:high_range] < 1).all(dim=1)
                pseudo_buffer_mask = (logit[:, low_range:high_range] > buffer_threshold) & buffer_data_mask.unsqueeze(1)
                target[:, low_range:high_range][pseudo_buffer_mask] = 1
                features.append(feature)
                targets.append(target)
                prediction = torch.sigmoid(model(image)['logits']).cpu()
                predictions.append(prediction)
                torch.cuda.empty_cache()

    features = torch.cat(features).numpy()
    targets = torch.cat(targets).numpy()
    predictions = torch.cat(predictions).numpy()

    return features, targets, predictions



def _icarl_selection(features, nb_examplars, class_idx, split_index):
    features = np.array(features)
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0],))

    w_t = mu
    iter_herding, iter_herding_eff = 0, 0

    selected_indices = []

    while len(selected_indices) < nb_examplars and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            if class_idx[ind_max] < split_index:
                herding_matrix[ind_max] = 1 + iter_herding
                selected_indices.append(ind_max)
                iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]
    return np.array(selected_indices[:nb_examplars])


def _nearest_to_mean_selection(features, nb_examplars, class_idx, split_index):
    features = np.array(features)
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    mu = np.mean(D, axis=1)

    selected_indices = []
    
    distances = np.linalg.norm(D - mu[:, np.newaxis], axis=0)
    
    sorted_indices = np.argsort(distances)
    
    for idx in sorted_indices:
        if len(selected_indices) >= nb_examplars:
            break
        
        if class_idx[idx] < split_index:
            selected_indices.append(idx)

    return np.array(selected_indices)


def _farthest_selection(features, nb_examplars, class_idx, split_index):
    features = np.array(features)
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    mu = np.mean(D, axis=1)

    selected_indices = []
    
    distances = np.linalg.norm(D - mu[:, np.newaxis], axis=0)
    
    sorted_indices = np.argsort(distances)[::-1]
    
    for idx in sorted_indices:
        if len(selected_indices) >= nb_examplars:
            break
        
        if class_idx[idx] < split_index:
            selected_indices.append(idx)

    return np.array(selected_indices)


def _kmeans_selection(features, nb_examplars, class_idx, split_index):
    features = np.array(features)
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)

    K = 3
    
    kmeans = KMeans(n_clusters=K, random_state=42)
    cluster_labels = kmeans.fit_predict(D.T)

    selected_indices = []
    cluster_centers = kmeans.cluster_centers_

    samples_per_cluster = math.ceil(nb_examplars / K)
    remaining_samples = nb_examplars
    for i in range(K):
        cluster_samples = np.where(cluster_labels == i)[0]
        distances_to_center = np.linalg.norm(D[:, cluster_samples] - cluster_centers[i][:, np.newaxis], axis=0)
        
        sorted_indices = np.argsort(distances_to_center)
        
        cluster_selected = []
        for idx in sorted_indices:
            if class_idx[cluster_samples[idx]] < split_index:
                cluster_selected.append(cluster_samples[idx])
                if len(cluster_selected) == samples_per_cluster:
                    break
        
        selected_indices.extend(cluster_selected)
        remaining_samples -= len(cluster_selected)
    
    if remaining_samples > 0:
        all_samples = set(range(len(class_idx)))
        remaining_valid_samples = list(all_samples - set(selected_indices))
        remaining_valid_samples = [idx for idx in remaining_valid_samples if class_idx[idx] < split_index]
        
        if remaining_valid_samples:
            additional_samples = np.random.choice(remaining_valid_samples, 
                                                  size=min(remaining_samples, len(remaining_valid_samples)), 
                                                  replace=False)
            selected_indices.extend(additional_samples)

    selected_indices = selected_indices[:nb_examplars]

    return np.array(selected_indices)


def _uncertainty_sampling_multilabel(model_predictions, nb_examplars, class_idx, split_index):
    uncertainties = _compute_uncertainty_multilabel(model_predictions)

    selected_indices = []
    
    sorted_indices = np.argsort(uncertainties)[::-1]
    
    for idx in sorted_indices:
        if len(selected_indices) >= nb_examplars:
            break
        
        if class_idx[idx] < split_index:
            selected_indices.append(idx)

    return np.array(selected_indices)


def _compute_uncertainty_multilabel(predictions):
    uncertainties = np.mean(np.abs(predictions - 0.5), axis=1)
    return uncertainties


def _kis_selection(features, predictions, nb_examplars, low_range, high_range, alpha=0.5):
    """
    Knowledge-Informed Selection (KIS)
    Selects exemplars by balancing representativeness and informativeness gain.

    :param features: Features of all candidate samples for a class.
    :param predictions: Model predictions (logits) for all candidate samples.
    :param nb_examplars: The number of exemplars to select.
    :param low_range: The starting index of the new classes.
    :param high_range: The ending index of the new classes.
    :param alpha: The weight to balance representativeness and informativeness.
                  final_score = alpha * rep_score + (1 - alpha) * info_score
    :return: An array of selected local indices.
    """
    if features.shape[0] == 0:
        return np.array([], dtype=int)

    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    mu = np.mean(D, axis=1, keepdims=True)
    
    distances = np.linalg.norm(D - mu, axis=0)
    
    if np.max(distances) == np.min(distances):
        rep_scores = np.ones_like(distances)
    else:
        normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        rep_scores = torch.exp(-normalized_distances)

    info_scores = np.mean(predictions[:, low_range:high_range], axis=1)

    if np.max(info_scores) == np.min(info_scores):
        normalized_info_scores = np.ones_like(info_scores)
    else:
        normalized_info_scores = (info_scores - np.min(info_scores)) / (np.max(info_scores) - np.min(info_scores))

    final_scores = alpha * rep_scores + (1 - alpha) * normalized_info_scores

    sorted_indices = np.argsort(final_scores)[::-1] 
    selected_local_indices = sorted_indices[:nb_examplars]
    
    return selected_local_indices


def sample_protos_buffer(model, 
                        low_range, 
                        high_range, 
                        train_dataset, 
                        train_loader,
                        buffer_loader, 
                        num_protos, 
                        buffer_threshold,
                        alpha=0.5,
                        logger=None):
    """
    Sample with Buffer
    """
    if logger:
        logger.info('Sample with Buffer Limit')

    assert num_protos >= 1
    print(f'Save {num_protos} protos per class')

    split_index = len(train_dataset)
    # Create Indices Matrix
    all_indices = []
    per_cls_index=[]
    for _ in range(low_range, high_range):
        class_indices = []
        all_indices.append(class_indices)
    
    # Forward to Get Features
    start_time = time.time()
    features, targets, predictions = _forward_features(model, train_loader, buffer_loader, low_range, high_range, buffer_threshold)
    indices = np.arange(len(features))
    if logger:
        logger.info(f'Forward Feature Time: {(time.time()-start_time):.2f}')

    # Assign Features to Regarding Labels
    start_time = time.time()
    for i, target in enumerate(targets):
        for label in target.nonzero()[0]:
            if low_range <= label < high_range:
                all_indices[label-low_range].append(indices[i])
    if logger:
        logger.info(f'Assign Features to Regarding Labels Time: {(time.time()-start_time):.2f}')

    # Select Samples for Each Classes
    start_time = time.time()
    for i, label in enumerate(range(low_range, high_range)):
        class_indices_global = np.array(all_indices[i])
        
        if len(class_indices_global) == 0:
            continue

        current_task_mask = class_indices_global < split_index
        eligible_indices_global = class_indices_global[current_task_mask]

        if len(eligible_indices_global) == 0:
            if logger:
                logger.warning(f"No samples from current task for class {label}. Skipping selection.")
            continue
        
        eligible_features = features[eligible_indices_global]
        eligible_predictions = predictions[eligible_indices_global]
        
        num_per_cls = min(num_protos, len(eligible_indices_global))

        selected_local_indices = _kis_selection(
            features=eligible_features,
            predictions=eligible_predictions,
            nb_examplars=num_per_cls,
            low_range=low_range,
            high_range=high_range,
            alpha=alpha
        )
        
        sample_index = eligible_indices_global[selected_local_indices]
        per_cls_index.append(sample_index)

    all_sample_index = np.concatenate(per_cls_index)
    if logger:
        logger.info(f'Select Samples for Each Class Time:{(time.time()-start_time):.2f}')

    sample_ds = ModifiableSubset(train_dataset, all_sample_index)

    return sample_ds, per_cls_index


def sample_protos(model, 
                  low_range, 
                  high_range, 
                  train_dataset, 
                  train_loader,
                  buffer_loader,
                  num_protos,
                  buffer_threshold,
                  alpha=0.5,
                  logger=None):
    """
    Sample without Buffer
    """
    if logger:
        logger.info('Sample -- No Buffer Limit')
    
    return sample_protos_buffer(
        model, low_range, high_range, train_dataset, train_loader,
        buffer_loader, num_protos, buffer_threshold, alpha, logger
    )