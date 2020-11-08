import logging
from functools import wraps

log = logging.getLogger(__name__)


def tweaking_loop(
    model_structures_with_image_sizes,
    optimizers,
    dense_layers_quantities,
    dl_neuron_quantities,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for model_structure_with_image_size in model_structures_with_image_sizes:
                model_structure = model_structure_with_image_size[0]
                image_size = model_structure_with_image_size[1]
                for optimizer in optimizers:
                    for dense_layers_quantity in dense_layers_quantities:
                        for dl_neuron_quantity in dl_neuron_quantities:
                            func(
                                model_structure=model_structure,
                                image_size=image_size,
                                dense_layers_quantity=dense_layers_quantity,
                                dl_neuron_quantity=dl_neuron_quantity,
                                optimizer=optimizer,
                            )
                            if dense_layers_quantity == 0:
                                break

        return wrapper

    return decorator
