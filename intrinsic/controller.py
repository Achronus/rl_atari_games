from core.enums import IMType, ValidIMMethods
from intrinsic.parameters import IMParameters


class IMController:
    """
    A controller that retrieves functionality for a given intrinsic motivation (IM) method.
    Acts as a form of storage to house the selected IM method's module and model, allowing an
    effective method for calling its methods without creating each all IM methods at once.

    :param im_type (str) - name of the type of intrinsic motivation to use
    :param params (IMParameters) - hyperparameters for the specified intrinsic method
    :param optim_params (dict) - optimizer hyperparameters (learning rate and epsilon)
    :param device (str) - CUDA device name
    """
    def __init__(self, im_type: str, params: IMParameters, optim_params: dict, device: str) -> None:
        valid_types = list(IMType.__members__.keys())
        if im_type not in valid_types:
            raise ValueError(f"'{im_type}' does not exist! Must be one of: '{valid_types}'.")

        self.im_type = im_type
        self.im_method = IMType[im_type].value

        self.params = params
        self.model = self.im_method['model'](params.input_shape, params.n_actions, device)
        self.module = self.im_method['module'](params, self.model, device)

        # Create empowerment optimizers
        if im_type == ValidIMMethods.EMPOWERMENT.value:
            self.module.source_optim = self.im_method['source_optim'](
                self.model.source_net.parameters(),
                lr=optim_params['lr'],
                eps=optim_params['eps']
            )
            self.module.forward_optim = self.im_method['forward_optim'](
                self.model.forward_net.parameters(),
                lr=optim_params['lr'],
                eps=optim_params['eps']
            )
        else:
            self.model = self.model.to(device)
