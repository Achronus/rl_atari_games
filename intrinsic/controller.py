from core.enums import IMType
from intrinsic.parameters import IMParameters


class IMController:
    """
    A controller that retrieves functionality for a given intrinsic motivation (IM) method.
    Acts as a form of storage to house the selected IM method's module and model, allowing an
    effective method for calling its methods without creating each all IM methods at once.
    """
    def __init__(self, im_type: str, params: IMParameters, device: str) -> None:
        valid_types = list(IMType.__members__.keys())
        if im_type not in valid_types:
            raise ValueError(f"'{im_type}' does not exist! Must be one of: '{valid_types}'.")

        self.im_type = im_type
        self.im_method = IMType[im_type].value

        self.params = params
        self.model = self.im_method['model'](params.input_shape, params.n_actions, device).to(device)
        self.module = self.im_method['module'](params, self.model, device)
