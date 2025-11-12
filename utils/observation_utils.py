import numpy as np
from typing import Tuple, List, Union, Literal, Callable
from POLARIScore.config import LOGGER
from POLARIScore.utils.batch_utils import compute_smoothness
from skimage import measure


def count_nonan(matrix:np.ndarray)->int:
    """Returns the number of non nan in a matrix"""
    return np.count_nonzero(~np.isnan(matrix))

def distance_to_center(matrix:np.ndarray)->np.ndarray:
    """Returns the distance to center of the matrix normalized between 0. to 1. where 1. is on center"""
    h, w = matrix.shape
    y, x = np.indices((h, w))    
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0    
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)    
    dist_normalized = 1.0 - dist / dist.max()
    print(dist_normalized)
    return dist_normalized

def find_context(canvas:np.ndarray, region:Tuple[int,int,int,int], context_size:int, score_methods:Union[List[Callable[[np.ndarray],float]],Callable[[np.ndarray],float]]=[count_nonan], method:Literal["order","mean"]="order")->Tuple[int,int,int,int]:
    """
    Find the best context(matrix) for a region(matrix) in a large matrix (named canvas).
    The best matrix is choosen by finding the maximum of a score function.
    Args:
        canvas: the large matrix which contains the region and context
        region: region defined by bouding box (square) where each integer is the indice of the corner [x1,y1,x2,y2].
        context_size(int): context size, need to be higher than region size.
        score_methods: score functions
        method: method to combine the score functions.
    Returns:
        context: bouding box of the context
    """

    assert len(canvas.shape) == 2, LOGGER.error("Can't find context because canvas is not 2D (i.e not a matrix).")

    assert region[2]-region[0]==region[3]-region[1], LOGGER.error(f"Region need to be a square: ({region[2]-region[0]},{region[3]-region[1]})")
    region_size:int = region[2]-region[0]
    assert context_size > region_size, LOGGER.error(f"Context size need to higher than the region size: (cont:{context_size},reg:{region_size})")

    if callable(score_methods):
        score_methods = [score_methods]

    steps = context_size-region_size
    scores = np.zeros((steps,steps,len(score_methods)))

    for j in range(steps):
        for i in range(steps):
            #if(all(scores[k] > 0 for k in scores)):
            #    continue
            x1 = max(region[0] - i, 0)
            y1 = max(region[1] - j, 0)
            x2 = min(x1 + context_size, canvas.shape[0])
            y2 = min(y1 + context_size, canvas.shape[1])

            if x2 - x1 < context_size or y2 - y1 < context_size:
                continue

            context = canvas[x1:x2, y1:y2]
            for k, s_method in enumerate(score_methods):
                try:
                    scores[i, j, k] = s_method(context)
                except Exception as e:
                    LOGGER.warn(f"Error in score method {k}: {e}")
                    scores[i, j, k] = 0.
    
    if method == "mean":
        combined_score = np.nanmean(scores, axis=2)
    else:  # "order"
        ranks = np.argsort(np.argsort(scores, axis=None)).reshape(scores.shape)
        combined_score = np.mean(ranks, axis=2)

    #TODO random choise between all image with maximum score
    best_idx = np.unravel_index(np.nanargmax(combined_score), combined_score.shape)
    best_i, best_j = best_idx

    context_x1 = max(region[0] - best_i, 0)
    context_y1 = max(region[1] - best_j, 0)
    context_x2 = min(context_x1 + context_size, canvas.shape[0])
    context_y2 = min(context_y1 + context_size, canvas.shape[1])

    return context_x1, context_y1, context_x2, context_y2

def get_clumps(array, threshold=0.85):
    """Return regions/clumps perimeters and areas in an image. (Unit: px) """
    threshold = np.nanpercentile(array, threshold*100) if threshold < 1 else threshold 
    mask = array > threshold

    labels = measure.label(mask, connectivity=2)
    props = measure.regionprops(labels)

    areas = []
    perimeters = []
    for region in props:
        A = region.area
        P = measure.perimeter(region.image)
        if A > 5:
            areas.append(A)
            perimeters.append(P)

    areas = np.array(areas)
    perimeters = np.array(perimeters)
    
    return perimeters, areas
    
