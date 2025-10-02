from typing import Dict, List, Tuple
from .utils import iou

class FaceTracker:
    def __init__(self, iou_thresh: float):
        self.iou_thresh = iou_thresh
        self.face_map: Dict[int, Tuple[int,int,int,int]] = {}
        self.next_id = 1

    def assign_ids(self, new_boxes: List[Tuple[int,int,int,int]]):
        old_map = self.face_map
        used = set()
        assigned = {}
        next_id = max([*old_map.keys(), self.next_id - 1]) + 1 if old_map else self.next_id

        for nb in new_boxes:
            best, best_id = 0.0, None
            for fid, ob in old_map.items():
                if fid in used:
                    continue
                score = iou(nb, ob)
                if score > best:
                    best, best_id = score, fid
            if best >= self.iou_thresh and best_id is not None:
                assigned[best_id] = nb
                used.add(best_id)
            else:
                assigned[next_id] = nb
                next_id += 1

        self.face_map = assigned
        self.next_id = next_id
        return self.face_map
