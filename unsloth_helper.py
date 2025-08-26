def apply_unsloth_augmentations(texts):
    """Try to apply Unsloth augmentations if the package is available.

    This wrapper keeps the main training script optional dependency-free.
    If Unsloth is not present it raises an ImportError.
    """
    try:
        import unsloth
    except Exception as e:
        raise ImportError('unsloth not available') from e

    # Example usage - adapt to actual Unsloth API
    aug_texts = []
    for t in texts:
        try:
            aug = unsloth.augment(t)
            if isinstance(aug, list):
                aug_texts.extend(aug)
            else:
                aug_texts.append(str(aug))
        except Exception:
            aug_texts.append(t)
    return aug_texts
