import numpy as np
import cv2


def perform_omission(mask, omission_rate, seed=None):
    """
    Performs the omission perturbation on the cell mask.
    :param mask: A 2D image or a 3D volume containing the cell labels.
    :param omission_rate: The proportion of the cells to be dropped.
    :param seed: Random seed.

    :return: The mask with the applied perturbation.
    """

    if seed:
        np.random.seed(seed)

    cell_labels = np.unique(mask)
    cell_labels = cell_labels[np.where(cell_labels != 0)]
    no_cells = len(cell_labels)

    omitted_cells = np.random.choice(cell_labels, max(1, int(omission_rate * no_cells)), replace=False)

    for oc in omitted_cells:
        mask[np.where(mask == oc)] = 0

    return mask


def merge_cell_images(image_main, mask_main, image_extra, mask_extra, threshold=0.8):
    """
    Merges an image belonging to the initial data set with an additional image with different cells
    in order to artificially perform the inclusion perturbation.
    :param image_main: The 2D image or 3D volume of the main data set.
    :param mask_main: The 2D mask or the 3D volume mask of the main data set.
    :param image_extra: The 2D image or 3D volume of the additional data set.
    :param mask_extra: The 2D mask or the 3D volume mask of the additional data set.
    :param threshold: The intensity threshold by which we include the cells from the additional image.

    :return: Merged image, merged mask.
    """
    # First scale the images
    image_main = image_main / np.max(image_main)
    image_extra = image_extra / np.max(image_extra)

    matching_indices = np.where(((mask_extra > 0) | (image_extra > threshold)) & (mask_main == 0))
    image_main[matching_indices] = image_extra[matching_indices]

    cell_labels_main = np.unique(mask_main)
    cell_labels_main = cell_labels_main[np.where(cell_labels_main != 0)]
    cell_labels_extra = np.unique(mask_extra)
    cell_labels_extra = cell_labels_extra[np.where(cell_labels_extra != 0)]

    no_cells_main = len(cell_labels_main)

    # Assuming the cell mask values are from 1 to no_cells in both the main and the additional mask
    for extra_cell in cell_labels_extra:
        mask_main[np.where((mask_main == 0) & (mask_extra == extra_cell))] = no_cells_main + extra_cell

    return image_main, mask_main


def perform_inclusion(mask, no_main_cells, inclusion_rate, seed=None):
    """
    Perform the inclusion perturbation. We assume two types of cells are present in the mask ordered from
    1 to the number of main cells; and the number of main cells + 1 to the number of main + additional cells.
    :param mask:  A 2D image or a 3D volume containing the cell labels.
    :param no_main_cells: The number of cells belonging to the main data set (these cells remain fixed).
    :param inclusion_rate: The proportion of the cells we keep from the additional category.
    :param seed: Random seed.

    :return: The mask containing all main cells and a percentage of the additional cells.
    """

    cell_labels = np.unique(mask)
    additional_cell_labels = cell_labels[np.where(cell_labels > no_main_cells)]

    if seed:
        np.random.seed(seed)

    discarded_cells = np.random.choice(additional_cell_labels,
                                       max(1, int((1 - inclusion_rate) * len(additional_cell_labels))), replace=False)

    for dc in discarded_cells:
        mask[np.where(mask == dc)] = 0

    return mask


def perform_bias(mask, qmax, seed=None):
    """
    Add enlarging or shrinking bias to cell masks.
    :param mask: A 2D image or a 3D volume containing the cell labels.
    :param qmax: The maximum number of iterations for which we perform the enlargement or shrinkage of the cells.
    :param seed: Random seed.

    :return: A binary label with the cells expanded or shrunk.
    """

    kernel = np.ones((3, 3), np.uint8)

    # Binarize the mask
    mask[np.where(mask != 0)] = 1

    if seed:
        np.random.seed(seed)

    operation = cv2.erode if np.random.rand() > 0.5 else cv2.dilate
    iterations = np.random.randint(1, qmax + 1)

    mask = np.squeeze(mask)     # Make sure the mask does not have too many dimensions
    if mask.shape == 2:
        mask = operation(mask, kernel, iterations=iterations)
    else:
        for i in range(mask.shape[0]):
            mask[i] = operation(mask[i], kernel, iterations=iterations)

    return mask
