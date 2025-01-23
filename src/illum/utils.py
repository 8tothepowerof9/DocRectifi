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
