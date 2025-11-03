class Config:
    # Model parameters
    POINT_FEATURE_DIM = 128
    MESH_FEATURE_DIM = 256
    GRAPH_HIDDEN_DIM = 64
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-4
    
    # Point cloud processing
    MAX_POINTS = 1024
    POINT_SAMPLING = "farthest"
    
    # Mesh processing
    MAX_VERTICES = 5000
    MAX_FACES = 10000
    
    # Reconstruction parameters
    POISSON_DEPTH = 9
    ALPHA_SHAPE_ALPHA = 0.03
    
    # Loss weights
    CHAMFER_WEIGHT = 1.0
    NORMAL_WEIGHT = 0.1
    SEGMENTATION_WEIGHT = 0.5
    
    @classmethod
    def to_dict(cls):
        return {key: value for key, value in cls.__dict__.items() 
                if not key.startswith('_') and not callable(value)}