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
