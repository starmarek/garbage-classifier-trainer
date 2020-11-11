import logging
from functools import wraps

log = logging.getLogger(__name__)


def tweaking_loop(
    model_structures,
    optimizers,
    dense_layers_quantities,
    dl_neuron_quantities,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log.debug("Starting training loop")
            for model_structure in model_structures:
                log.debug(f"Model structure change to `{model_structure}`")
                for optimizer in optimizers:
                    log.debug(f"Optimizer change to `{optimizer}`")
                    for dense_layers_quantity in dense_layers_quantities:
                        log.debug(
                            "Dense layer quantity "
                            f"change to `{dense_layers_quantity}`"
                        )
                        for dl_neuron_quantity in dl_neuron_quantities:
                            log.debug(
                                "Dense layer neuron quantity change "
                                f"to `{dl_neuron_quantity}`"
                            )
                            log.debug(
                                f"Running train function with arguments = "
                                f"model_structure: {model_structure}, "
                                f"optimizer: {optimizer}, "
                                f"dense_layers_quantity: {dense_layers_quantity}, "
                                f"dense_layer_neuron_quantity: {dl_neuron_quantity}"
                            )
                            func(
                                model_structure=model_structure,
                                dense_layers_quantity=dense_layers_quantity,
                                dl_neuron_quantity=dl_neuron_quantity,
                                optimizer=optimizer,
                            )
                            if dense_layers_quantity == 0:
                                break

        return wrapper

    return decorator
