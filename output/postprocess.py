def unnormalize(feature, normalization_param):
    """
    Unnormalize a feature dataset, to make it visible for the analysis.

    Args:
        feature (np arr): Feature of the dataset
        normalization_param (list): Normalization parameter, obtained from preprocess.normalize

    Returns:
        unnormalized_feature (np arr): Unnormalize feature array

    """
    mean, std = normalization_param
    unnormalized_feature = feature * std + mean

    return unnormalized_feature
