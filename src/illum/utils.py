import torch
import cv2


class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.best_model_state = None  # Store the best model
        self.best_model_epoch = None  # Store the best model epoch

    def early_stop(self, monitor_metric, model, epoch):
        if monitor_metric < self.min_validation_loss:
            self.min_validation_loss = monitor_metric
            self.counter = 0
            self.best_model_state = model.state_dict()  # Save the best model state
            self.best_model_epoch = epoch
        elif monitor_metric > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class EarlyStoppingMultiModel:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.best_model_1_state = None
        self.best_model_2_state = None
        self.best_model_epoch = None

    def early_stop(self, monitor_metric, model_1, model_2, epoch):
        if monitor_metric < self.min_validation_loss:
            self.min_validation_loss = monitor_metric
            self.counter = 0
            self.best_model_1_state = model_1.state_dict()
            self.best_model_2_state = model_2.state_dict()
            self.best_model_epoch = epoch
        elif monitor_metric > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def seconds_to_minutes_str(seconds):
    return f"{seconds//60}m {(seconds%60):3f}s"


def pad_to_stride(img, stride=32):
    """Pads a NumPy image array so its height and width are multiples of stride."""
    h, w = img.shape[:2]  # Assuming shape is (H, W, C)

    padding_h = (stride - h % stride) % stride
    padding_w = (stride - w % stride) % stride

    # Pad the bottom and right using border replication (similar to PyTorch 'replicate' mode)
    img_padded = cv2.copyMakeBorder(
        img, 0, padding_h, 0, padding_w, cv2.BORDER_REPLICATE
    )

    return img_padded, padding_h, padding_w


def remove_padding(img: torch.Tensor, padding_h: int, padding_w: int):
    """Removes the added padding from an image tensor."""
    if padding_h > 0:
        img = img[..., :-padding_h, :]  # Remove rows from bottom
    if padding_w > 0:
        img = img[..., :, :-padding_w]  # Remove columns from right
    return img
