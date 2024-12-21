from dataclasses import dataclass
import numpy as np

@dataclass
class BestMatch:        
    id: int
    name: str
    img_path: str
    distance: float        
    histogram: np.ndarray
    radius: float
    pca_comp: np.ndarray
    kp_center: np.ndarray
    width: int = 0
    height: int = 0
    t_nn_score: float = 0.0

    def __repr__(self):
        return f"""id: {self.id}
                name:{self.name}
                img_path: {self.img_path}
                distance: {self.distance}
                histogram: {self.histogram}
                radius: {self.radius}
                t_nn_score: {self.t_nn_score}"""
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "img_path": self.img_path,
            "distance": self.distance,
            "histogram": self.histogram,
            "radius": self.radius,
        }
    
    