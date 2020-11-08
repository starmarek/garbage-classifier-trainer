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
            for model_structure in model_structures:
                for optimizer in optimizers:
                    for dense_layers_quantity in dense_layers_quantities:
                        for dl_neuron_quantity in dl_neuron_quantities:
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
