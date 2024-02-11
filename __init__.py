"""Albumentations image augmentation plugin.
"""


"""Stable Diffusion Inpainting augmentation plugin.
"""

from bson import json_util
import inspect
import os

import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
import pandas as pd
import cv2
from PIL import Image
import torch
import numpy as np
from diffusers.utils import load_image, make_image_grid
import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np
import random
import hashlib
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone.core.utils import add_sys_path


from bson import json_util
import inspect
import os

import albumentations as A
import cv2
import json
import numpy as np
from PIL import Image

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone.core.utils import add_sys_path

LAST_ALBUMENTATIONS_RUN_KEY = "_last_albumentations_run"
ALBUMENTATIONS_RUN_INDICATOR = "albumentations_transform_"

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    # pylint: disable=no-name-in-module,import-error
    from supported_transforms import (
        SUPPORTED_BLUR_TRANSFORMS,
        SUPPORTED_CROP_TRANSFORMS,
        SUPPORTED_DROPOUT_TRANSFORMS,
        SUPPORTED_GEOMETRIC_TRANSFORMS,
        SUPPORTED_FUNCTIONAL_TRANSFORMS,
    )
    from utils import (
        _camel_to_snake,
        _create_hash,
        _execution_mode,
        _get_image_size,
        _get_target_view,
        _convert_bbox_to_albumentations,
        _convert_bbox_from_albumentations,
        _convert_keypoint_to_albumentations,
        _convert_keypoint_from_albumentations,
        _enforce_mask_size,
        _get_label_fields,
        _get_detections_fields,
        _get_keypoints_fields,
        _get_mask_fields,
        _join_lines_with_indentation,
        _list_target_views,
    )


NAME_TO_TYPE = {
    "scale": "float",
    "slant_lower": "int",
    "slant_upper": "int",
    "drop_length": "int",
    "drop_width": "int",
    "translate_percent": "float",
    "translate_px": "int",
    "rotate": "float",
    "shear": "float",
}

SUPPORTED_TRANSFORMS = (
    *SUPPORTED_BLUR_TRANSFORMS,
    *SUPPORTED_CROP_TRANSFORMS,
    *SUPPORTED_GEOMETRIC_TRANSFORMS,
    *SUPPORTED_DROPOUT_TRANSFORMS,
    *SUPPORTED_FUNCTIONAL_TRANSFORMS,
)


def serialize_view(view):
    return json.loads(json_util.dumps(view._serialize()))


def _collect_bboxes(sample, label_fields):
    """Collect all bounding boxes from the given sample.

    Args:
        sample: a fiftyone.core.sample.Sample
        label_fields: a list of label field names to collect bounding boxes from

    Returns:
        a list of [x, y, width, height, id] lists representing bounding boxes
        for the given sample, where id is the id of the detection in the sample
    """
    boxes_list = []
    for field in label_fields:
        detections = sample[field].detections
        for det in detections:
            bbox = _convert_bbox_to_albumentations(det.bounding_box)
            boxes_list.append(bbox + [str(det._id)])

    return boxes_list


def _collect_masks(sample, mask_fields, detections_fields):
    """Collect all masks from the given sample.

    Args:
        sample: a fiftyone.core.sample.Sample
        mask_fields: a list of label field names to collect full masks from
        detections_fields: a list of label field names to collect instance masks from

    Returns:
        a dict of {id: mask} representing masks
        for the given sample, where id is the id of the label in the sample
    """

    width, height = _get_image_size(sample)

    masks_dict = {}

    ## Add semantic segmentation masks and heatmaps to the masks dict
    for field in mask_fields:
        mask_label = sample[field]
        if isinstance(mask_label, fo.Segmentation):
            if mask_label.mask_path is not None:
                mask = np.array(Image.open(mask_label.mask_path))
            else:
                mask = mask_label.mask
        elif isinstance(mask_label, fo.Heatmap):
            if mask_label.map_path is not None:
                mask = np.array(Image.open(mask_label.map_path))
            else:
                mask = mask_label.map

        mask = _enforce_mask_size(mask, width, height)

        masks_dict[str(mask_label._id)] = mask

    ## Add instance masks to the masks dict
    for field in detections_fields:
        detections = sample[field].detections
        for det in detections:
            mask = det.mask
            if mask is None:
                continue
            mask = _project_instance_mask(mask, det.bounding_box, (width, height))
            masks_dict[str(det._id)] = mask

    return masks_dict


def _project_instance_mask(mask, bbox, frame_size):
    det = fo.Detection(mask=mask, bounding_box=bbox)
    seg = det.to_segmentation(frame_size=frame_size)
    return seg.mask.astype(np.uint8) / 255


def _crop_instance_mask(mask, bbox):
    alb_bbox = _convert_bbox_to_albumentations(bbox)
    x1, y1, x2, y2 = alb_bbox
    img_size = mask.shape
    im = Image.fromarray(mask)
    im_cropped = im.crop(
        (x1 * img_size[1], y1 * img_size[0], x2 * img_size[1], y2 * img_size[0])
    )
    return np.array(im_cropped)


def _collect_keypoints(sample, keypoints_fields):
    width, height = _get_image_size(sample)

    keypoints_dict = {}
    for field in keypoints_fields:
        keypoints = sample[field].keypoints
        for kp in keypoints:
            kp_id = str(kp._id)
            kp_points = kp.points
            for i, point in enumerate(kp_points):
                kpp_id = f"{kp_id}_{i}"
                keypoints_dict[kpp_id] = _convert_keypoint_to_albumentations(
                    point, [width, height]
                )

    return keypoints_dict


def _update_detection_field(
    original_sample,
    new_sample,
    detection_field,
    transformed_boxes,
    transformed_masks_dict,
):
    detections = original_sample[detection_field].detections
    new_detections = []
    for det in detections:
        if str(det._id) in transformed_boxes:
            new_det = det.copy()
            new_det.bounding_box = transformed_boxes[str(det._id)]

            ## Remove instance masks from the new detections
            ##  Until we can figure out how to crop them properly
            if new_det.mask is not None:
                new_det.mask = None
            ## Add instance masks to the new detections
            if str(det._id) in transformed_masks_dict:
                full_mask = transformed_masks_dict[str(det._id)]
                instance_mask = _crop_instance_mask(full_mask, new_det.bounding_box)
                new_det.mask = instance_mask

            new_detections.append(new_det)
    new_sample[detection_field] = fo.Detections(detections=new_detections)
    return new_sample


def _update_mask_field(original_sample, new_sample, mask_field, transformed_masks_dict):
    mid = str(original_sample[mask_field]._id)
    new_mask_label = original_sample[mask_field].copy()
    if isinstance(new_mask_label, fo.Segmentation):
        if new_mask_label.mask_path is not None:
            new_mask_label["mask_path"] = None
        else:
            new_mask_label["mask"] = transformed_masks_dict[mid]
    elif isinstance(new_mask_label, fo.Heatmap):
        if new_mask_label.map_path is not None:
            new_mask_label["map_path"] = None
        else:
            new_mask_label["map"] = transformed_masks_dict[mid]
    new_sample[mask_field] = new_mask_label
    return new_sample


def _update_keypoints_field(
    original_sample, new_sample, keypoints_field, transformed_keypoints_dict
):
    new_kps = []
    for kp in original_sample[keypoints_field].keypoints:
        kp_id = str(kp._id)
        new_kp = kp.copy()
        for i in range(len(kp.points)):
            kpp_id = f"{kp_id}_{i}"
            if kpp_id in transformed_keypoints_dict:
                new_kp.points[i] = transformed_keypoints_dict[kpp_id]
            else:
                new_kp.points[i] = [float("nan"), float("nan")]
        new_kps.append(new_kp)
    kps = fo.Keypoints(keypoints=new_kps)
    new_sample[keypoints_field] = kps
    return new_sample


def transform_sample(sample, transforms, label_fields=False, new_filepath=None):
    """Apply an Albumentations transform to the image
    and all label fields listed for the sample.

    Args:
        sample: a fiftyone.core.sample.Sample
        transform: an Albumentations transform
        label_fields (False): a field or list of field names to transform.
            If label_fields=False, no label fields are transformed.
            If label_fields=True, all label fields are transformed.

    Currently does not handle:
    - instance segmentation masks.
    """
    if new_filepath is None:
        hash = _create_hash()
        new_filepath = f"/tmp/{hash}.jpg"

    if not label_fields:
        label_fields = []
    if label_fields is True:
        label_fields = _get_label_fields(sample)

    image = cv2.cvtColor(cv2.imread(sample.filepath), cv2.COLOR_BGR2RGB)

    detection_fields = _get_detections_fields(sample, label_fields)
    boxes_list = _collect_bboxes(sample, detection_fields)

    mask_fields = _get_mask_fields(sample, label_fields)

    masks_dict = _collect_masks(sample, mask_fields, detection_fields)
    masks = list(masks_dict.values())

    keypoint_fields = _get_keypoints_fields(sample, label_fields)
    keypoints_dict = _collect_keypoints(sample, keypoint_fields)
    keypoints = list(keypoints_dict.values())
    keypoint_labels = list(keypoints_dict.keys())

    has_boxes = len(boxes_list) > 0
    has_masks = len(masks) > 0
    has_keypoints = len(keypoints) > 0

    kwargs = {"image": image}
    if has_boxes:
        kwargs["bboxes"] = boxes_list
    if has_masks:
        kwargs["masks"] = masks
    if has_keypoints:
        kwargs["keypoints"] = keypoints
        kwargs["keypoint_labels"] = keypoint_labels
    else:
        kwargs["keypoints"] = []
        kwargs["keypoint_labels"] = []

    compose_kwargs = {}
    if has_boxes:
        compose_kwargs["bbox_params"] = A.BboxParams(format="albumentations")
    if has_keypoints:
        compose_kwargs["keypoint_params"] = A.KeypointParams(
            format="xy", label_fields=["keypoint_labels"], remove_invisible=True
        )

    transform = A.ReplayCompose(
        transforms,
        **compose_kwargs,
    )

    transformed = transform(**kwargs)
    transform_record = transformed['replay']
    transform_record['transformed_sample_id'] = sample.id

    transformed_image = transformed["image"]

    if has_boxes:
        transformed_boxes = transformed["bboxes"]
        transformed_boxes = {
            bbox[-1]: _convert_bbox_from_albumentations(bbox[:-1])
            for bbox in transformed_boxes
        }

    if has_masks:
        transformed_masks = transformed["masks"]
        transformed_masks_dict = {
            id: mask for id, mask in zip(masks_dict.keys(), transformed_masks)
        }
    else:
        transformed_masks_dict = {}

    if has_keypoints:
        transformed_keypoints = transformed["keypoints"]
        transformed_keypoint_labels = transformed["keypoint_labels"]
        transformed_keypoints_dict = {
            kp_id: _convert_keypoint_from_albumentations(
                kp, [transformed_image.shape[1], transformed_image.shape[0]]
            )
            for kp_id, kp in zip(transformed_keypoint_labels, transformed_keypoints)
        }

    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(new_filepath, transformed_image)

    
    new_sample = fo.Sample(filepath=new_filepath, tags=['augmented'], transform=transform_record)

    if has_boxes:
        for detection_field in detection_fields:
            new_sample = _update_detection_field(
                sample,
                new_sample,
                detection_field,
                transformed_boxes,
                transformed_masks_dict,
            )

    if has_masks:
        for mask_field in mask_fields:
            new_sample = _update_mask_field(
                sample, new_sample, mask_field, transformed_masks_dict
            )

    if has_keypoints:
        for keypoint_field in keypoint_fields:
            new_sample = _update_keypoints_field(
                sample, new_sample, keypoint_field, transformed_keypoints_dict
            )

    sample._dataset.add_sample(new_sample)
    return new_sample.id


################################################
################################################
######## Transformation Input Helpers ##########


def _process_function_args(func):
    docstring = inspect.getdoc(func)

    args_section = docstring.split("Args:")[1].split("Targets:")[0]
    args_lines = args_section.splitlines()

    args = _join_lines_with_indentation(args_lines)
    args = [arg.strip() for arg in args if arg.strip() != ""]
    args = [_get_arg_details(arg) for arg in args]
    return [arg for arg in args if arg is not None]


def _get_arg_type_str(arg_name_and_type):
    arg_type = "(".join(arg_name_and_type.split("(")[1:])
    arg_type = ")".join(arg_type.split(")")[:-1])
    return arg_type


def _get_arg_details(arg):
    arg_name_and_type = arg.split(":")[0].strip()

    try:
        if "(" not in arg_name_and_type:
            arg_name = arg_name_and_type.split(":")[0].strip()
            if arg_name in NAME_TO_TYPE:
                arg_type = NAME_TO_TYPE[arg_name]
                arg_description = "".join(arg.split(":")[1:]).strip()
        else:
            arg_name = arg_name_and_type.split("(")[0].strip()
            arg_type = _get_arg_type_str(arg_name_and_type)
            arg_description = "".join(arg.split(":")[1:]).strip()
        return {"name": arg_name, "type": arg_type, "description": arg_description}
    except:
        return None


def tuple_creator(inputs, arg_type):
    def input_adder(name, **input_args):
        default = input_args.pop("default", None)

        first_input_args = {
            "view": types.FieldView(space=3),
        }
        second_input_args = {
            "view": types.FieldView(space=3),
        }
        if default is not None and type(default) == tuple and len(default) == 2:
            first_input_args["default"] = default[0]
            first_input_args["required"] = False
            second_input_args["default"] = default[1]
            second_input_args["required"] = False

        obj = types.Object()
        constructor = obj.float if arg_type == "float" else obj.int
        constructor("first", **first_input_args)
        constructor("second", **second_input_args)
        inputs.define_property(name, obj, **input_args)

    return input_adder


def _get_input_factory(inputs, arg_type):
    if arg_type == "bool":
        return inputs.bool
    elif arg_type == "int":
        return inputs.int
    elif arg_type == "float":
        return inputs.float
    elif arg_type == "str":
        return inputs.str
    elif "int, int" in arg_type:
        return tuple_creator(inputs, "int")
    elif "float, float" in arg_type:
        return tuple_creator(inputs, "float")
    elif "tuple of int" in arg_type:
        return tuple_creator(inputs, "int")
    elif "tuple of float" in arg_type:
        return tuple_creator(inputs, "float")
    elif arg_type == "int, list of int":
        ## doesn't support list of int yet
        return inputs.int
    elif arg_type == "int, float":
        return inputs.float
    elif arg_type == "(float, float) or float":
        ## doesn't support tuple yet
        return inputs.float
    elif arg_type == "(int, int) or int":
        ## doesn't support tuple yet
        return inputs.int
    elif "float," in arg_type or "float or" in arg_type:
        ## doesn't support tuple yet
        return inputs.float
    elif "number," in arg_type or "number or" in arg_type:
        ## doesn't support tuple yet
        return inputs.float
    return None


def _add_transform_inputs(inputs, transform_name):
    transform_args = _process_function_args(getattr(A, transform_name))

    t = getattr(A, transform_name)
    camel_case_name = _camel_to_snake(transform_name)

    parameters = inspect.signature(t).parameters

    for arg in transform_args:
        default = None
        arg_name = arg["name"]
        arg_type = arg["type"]
        arg_description = arg["description"]

        input_args = {
            "label": arg_name,
            "description": arg_description,
            "required": True,
        }

        if arg_name in parameters:
            default = parameters[arg_name].default
        if default is not None and default != None and default != inspect._empty:
            input_args["default"] = default
            input_args["required"] = False
        elif default is None:
            input_args["required"] = False

        input_factory = _get_input_factory(inputs, arg_type)
        if input_factory is not None:
            input_factory(f"{camel_case_name}__{arg_name}", **input_args)


################################################
################################################
######## Transformation Creation Helpers #######


def _extract_transform_inputs(ctx, transform_name):
    prefix = _camel_to_snake(transform_name) + "__"
    transform_params = {}
    for param in ctx.params:
        if param.startswith(prefix):
            param_name = param[len(prefix) :]
            transform_params[param_name] = ctx.params[param]
    return transform_params


def _unwrap_dict_tuple(d):
    if isinstance(d, dict):
        if "first" in d and "second" in d:
            return (d["first"], d["second"])
    return d


def _unwrap_dict_tuples(args, kwargs):
    args = [_unwrap_dict_tuple(arg) for arg in args]
    kwargs = {key: _unwrap_dict_tuple(val) for key, val in kwargs.items()}
    return args, kwargs


def _create_transform(ctx, transform_name):
    input_dict = _extract_transform_inputs(ctx, transform_name)

    t = getattr(A, transform_name)
    params = inspect.signature(t).parameters

    args = []
    kwargs = {}

    for param_name, param in params.items():
        if param_name in input_dict:
            if param.default != inspect._empty:
                if (
                    input_dict[param_name] != None
                    and input_dict[param_name] is not None
                ):
                    kwargs[param_name] = input_dict[param_name]
            else:
                args.append(input_dict[param_name])

    args, kwargs = _unwrap_dict_tuples(args, kwargs)

    return t(*args, **kwargs)


####### Unifying functions #######


def get_input_parser(transform_name):
    function_name = f"_{transform_name}_input"
    if function_name in globals():
        # Defined explicitly above
        input_parser_function = globals()[function_name]
    else:
        # Define the function dynamically using a lambda function
        input_parser_function = lambda ctx, inputs: _add_transform_inputs(
            inputs, transform_name
        )
    return input_parser_function


def get_transform_func(transform_name):
    function_name = f"_{transform_name}_transform"
    if function_name in globals():
        # Defined explicitly above
        transform_function = globals()[function_name]
    else:
        # Define the function dynamically using a lambda function
        transform_function = lambda ctx: _create_transform(ctx, transform_name)
    return transform_function


def _get_albumentations_run_names(ctx):
    run_keys = ctx.dataset.list_runs()
    run_keys = [
        run_key
        for run_key in run_keys
        if run_key.startswith(ALBUMENTATIONS_RUN_INDICATOR)
    ]
    run_names = [ctx.dataset.get_run_info(run_key).config.name for run_key in run_keys]
    return run_names


def _transforms_from_primitive_input(ctx, inputs, num=0):
    transform_choices = sorted(SUPPORTED_TRANSFORMS)
    transforms_group = types.RadioGroup()

    for tc in transform_choices:
        transforms_group.add_choice(tc, label=tc)

    inputs.enum(
        f"transforms__{num}",
        transforms_group.values(),
        label="Transform to apply",
        description="The Albumentations transform to apply to your images",
        view=types.AutocompleteView(),
        required=True,
    )


def _get_saved_transform_run_names(ctx):
    run_keys = ctx.dataset.list_runs()
    run_keys = [
        run_key
        for run_key in run_keys
        if run_key.startswith(ALBUMENTATIONS_RUN_INDICATOR)
    ]
    run_names = [ctx.dataset.get_run_info(run_key).config.name for run_key in run_keys]
    return run_names


def _get_run_key_from_name(ctx, run_name):
    run_keys = ctx.dataset.list_runs()
    run_keys = [
        run_key
        for run_key in run_keys
        if run_key.startswith(ALBUMENTATIONS_RUN_INDICATOR)
    ]
    for run_key in run_keys:
        if ctx.dataset.get_run_info(run_key).config.name == run_name:
            break
    return run_key


def _format_transform(config):
    transform = config.get("transform", {})
    if "transform" in transform:
        transform = transform["transform"].copy()
    
    for key in ("bbox_params", "keypoint_params", "additional_targets", "is_check_shapes"):
        if key in transform:
            transform.pop(key, None)
    return transform


def _execute_run_info(ctx, run_key):
    info = ctx.dataset.get_run_info(run_key)

    timestamp = info.timestamp.strftime("%Y-%M-%d %H:%M:%S")
    version = info.version
    config = info.config.serialize()
    config = {k: v for k, v in config.items() if v is not None}
    transform = _format_transform(config)

    label_fields = config.get("label_fields", None)

    return {
        "timestamp": timestamp,
        "version": version,
        "transform": transform,
        "label_fields": label_fields,
    }


def _initialize_run_output(ctx, run_key=False):
    outputs = types.Object()
    if run_key:
        outputs.str("run_key", label="Run key")
    outputs.str("timestamp", label="Creation time")
    outputs.str("version", label="FiftyOne version")
    outputs.str("label_fields", label="Label fields")
    outputs.obj("transform", label="Transform", view=types.JSONView())
    return outputs


def _transforms_from_saved_input(ctx, inputs, num=0):
    run_names = _get_saved_transform_run_names(ctx)

    transforms_group = types.RadioGroup()

    for rn in run_names:
        transforms_group.add_choice(rn, label=rn)

    inputs.enum(
        f"transforms__{num}",
        transforms_group.values(),
        label="Transform to apply",
        description="Choose a saved transform to apply to your images",
        view=types.DropdownView(),
        required=True,
    )


def _transforms_input(ctx, inputs, num=0):
    source = ctx.params.get(f"transform_source__{num}", None)
    if source == "Primitive":
        _transforms_from_primitive_input(ctx, inputs, num=num)
    elif source == "Saved":
        _transforms_from_saved_input(ctx, inputs, num=num)


def _cleanup_last_transform(dataset):
    run_key = LAST_ALBUMENTATIONS_RUN_KEY

    if run_key not in dataset.list_runs():
        return

    results = dataset.load_run_results(run_key)
    if results.save_augmentations:
        results.save_augmentations = False
        dataset.save_run_results(run_key, results, overwrite=True)
        return

    ids = results.new_sample_ids
    fps = dataset.select(ids).values("filepath")
    for fp in fps:
        os.remove(fp)

    dataset.delete_samples(ids)


def _store_last_transform(
    transforms, dataset, target_view, label_fields, new_sample_ids
):
    run_key = LAST_ALBUMENTATIONS_RUN_KEY
    # transform_dict = transform.to_dict()
    # transform_dict = {"transform": transform}
    transform_dict = A.Compose(transforms).to_dict()

    config = dataset.init_run()
    config.transform = transform_dict
    config.label_fields = label_fields

    if run_key not in dataset.list_runs():
        dataset.register_run(run_key, config)
        results = dataset.init_run_results(run_key)
    else:
        dataset.update_run_config(run_key, config)
        results = dataset.load_run_results(run_key)

    results.target_view = target_view._serialize()
    results.new_sample_ids = new_sample_ids
    results.save_augmentations = False
    dataset.save_run_results(run_key, results, overwrite=True)


def _save_transform(dataset, transform, name):
    hash = _create_hash()

    run_key = f"albumentations_transform_{hash}"

    config = dataset.init_run()
    config.name = name
    config.transform = transform
    dataset.register_run(run_key, config)


def _save_augmentations(ctx):
    """
    Save the samples generated by the last Albumentations run. If this is not
    done, the samples will be deleted when the next Albumentations run is
    executed.
    """
    dataset = ctx.dataset
    run_key = LAST_ALBUMENTATIONS_RUN_KEY

    if run_key not in dataset.list_runs():
        return

    results = dataset.load_run_results(run_key)
    results.save_augmentations = True
    dataset.save_run_results(run_key, results, overwrite=True)


def _transform_source_input(ctx, inputs, num=0):
    source_choices = ["Primitive", "Saved"]

    source_choices_group = types.RadioGroup()

    for choice in source_choices:
        source_choices_group.add_choice(choice, label=choice)

    inputs.enum(
        f"transform_source__{num}",
        source_choices_group.values(),
        label="Transform source",
        description="Select a transform from `Albumentations` primitives, or a saved transform",
        view=types.TabsView(),
        required=True,
    )


def _label_fields_input(ctx, inputs):
    if ctx.selected and len(ctx.selected) > 0:
        sample = ctx.dataset.select(ctx.selected[0]).first()
    else:
        sample = ctx.view.first()

    label_fields = _get_label_fields(sample)

    if not label_fields:
        return
    elif len(label_fields) == 1:
        inputs.bool(
            "label_fields",
            label="Transform label field",
            description=f"Check to transform the label field '{label_fields[0]}'",
            default=True,
            view=types.CheckboxView(),
        )
    else:
        inputs.view("labels_header", types.Header(label="Select Labels", divider=True))
        inputs.bool(
            "label_fields",
            label="Transform all label fields",
            description="Check to transform all label fields along with the image",
            default=True,
            view=types.CheckboxView(),
        )

        if ctx.params.get("label_fields", True):
            return

        for lf in label_fields:
            inputs.bool(
                f"label_field__{lf}",
                label=f"Transform {lf}?",
                default=False,
                view=types.CheckboxView(spaces=3),
            )


def _get_label_fields_to_transform(ctx):
    if ctx.params.get("label_fields", True):
        if ctx.selected:
            sample = ctx.dataset.select(ctx.selected[0]).first()
        else:
            sample = ctx.view.first()
        label_fields = _get_label_fields(sample)
        return label_fields
    else:
        label_fields = []
        for param in ctx.params:
            if param.startswith("label_field__"):
                if ctx.params[param]:
                    label_fields.append(param[len("label_field__") :])
        return label_fields


class AugmentWithAlbumentations(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="augment_with_albumentations",
            label="Augment with Albumentations",
            description="Apply an Albumentations transform to the image and all label fields listed for the sample.",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        if not ctx.dataset or not ctx.view:
            inputs.view(
                "no_dataset_warning", 
                types.Warning(label="No dataset", description="Need dataset to apply an Albumentations transform")
            )
            return types.Property(inputs)
        
        form_view = types.View(
            label="Augment with Albumentations",
            description="Apply an Albumentations transform to the image and all label fields listed for the sample.",
        )

        inputs.int(
            "num_augs",
            label="Number of augmentations per sample",
            description="The number of random augmentations to apply to each sample",
            default=1,
            view=types.FieldView(),
        )

        inputs.int(
            "num_transforms",
            label="Number of transforms",
            description="The number of transforms to compose and apply to samples",
            default=1,
            required=True,
        )

        _label_fields_input(ctx, inputs)

        num_transforms = ctx.params.get("num_transforms", 1)
        if num_transforms is not None and num_transforms < 1:
            inputs.view(
                "no_transforms_error",
                types.Error(
                    label="No transforms",
                    description="The number of transforms must be greater than 0",
                ),
            )
            return types.Property(inputs, view=form_view)

        if num_transforms is not None:
            for i in range(num_transforms):
                inputs.view(
                    f"transform_{i}_header",
                    types.Header(label=f"Transform {i+1}", divider=True),
                )
                _transform_source_input(ctx, inputs, num=i)
                _transforms_input(ctx, inputs, num=i)
                transform_name = ctx.params.get(f"transforms__{i}", None)
                try:
                    transform_input_parser = get_input_parser(transform_name)
                    transform_input_parser(ctx, inputs)
                except:
                    pass

        _list_target_views(ctx, inputs)
        _execution_mode(ctx, inputs)
        return types.Property(inputs, view=form_view)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        num_transforms = ctx.params.get("num_transforms", 1)

        transforms = []
        for i in range(num_transforms):
            transform_name = ctx.params.get(f"transforms__{i}", None)
            transform_source = ctx.params.get(f"transform_source__{i}", None)
            if transform_source == "Primitive":
                transform = get_transform_func(transform_name)(ctx)
                transforms.append(transform)
            elif transform_source == "Saved":
                run_name = transform_name
                for run_key in ctx.dataset.list_runs():
                    config = ctx.dataset.get_run_info(run_key).config
                    if "name" in config.serialize() and config.name == run_name:
                        break
                transform_dict = ctx.dataset.get_run_info(run_key).config.transform
                new_transforms = [
                    A.from_dict({"transform": td})
                    for td in transform_dict["transform"]["transforms"]
                ]
                transforms.extend(new_transforms)

        num_augs = ctx.params.get("num_augs", 1)

        label_fields = _get_label_fields_to_transform(ctx)

        target = ctx.params.get("target", None)
        _cleanup_last_transform(ctx.dataset)
        target_view = _get_target_view(ctx, target)

        new_sample_ids = []

        for sample in target_view:
            for _ in range(num_augs):
                new_sample_id = transform_sample(sample, transforms, label_fields)
                new_sample_ids.append(new_sample_id)

        _store_last_transform(
            transforms, ctx.dataset, target_view, label_fields, new_sample_ids
        )
        ctx.trigger("reload_dataset")


class GetLastAlbumentationsRunInfo(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="get_last_albumentations_run_info",
            label="Get info about the last Albumentations run",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        if LAST_ALBUMENTATIONS_RUN_KEY not in ctx.dataset.list_runs():
            inputs.view(
                "warning",
                types.Warning(
                    label="No Albumentations runs yet!",
                    description="To create a transform, use the `Augment with Albumentations` operator",
                ),
            )
            inputs.str("run_key", required=True, view=types.HiddenView())
            return types.Property(inputs, view=types.View())

        view = types.View(label="Get last Albumentations run info")
        return types.Property(inputs, view=view)

    def execute(self, ctx):
        run_key = LAST_ALBUMENTATIONS_RUN_KEY
        return _execute_run_info(ctx, run_key)

    def resolve_output(self, ctx):
        outputs = _initialize_run_output(ctx)
        view = types.View(label="Last Albumentations run info")
        return types.Property(outputs, view=view)


class ViewLastAlbumentationsRun(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="view_last_albumentations_run",
            label="View last Albumentations run",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        view = types.View(label="View last Albumentations run")
        return types.Property(inputs, view=view)

    def execute(self, ctx):
        run_key = LAST_ALBUMENTATIONS_RUN_KEY

        if run_key not in ctx.dataset.list_runs():
            return {
                "message": "To create a transform, use the `Augment with Albumentations` operator",
            }

        results = ctx.dataset.load_run_results(run_key)
        new_sample_ids = results.new_sample_ids
        view = ctx.dataset.select(new_sample_ids)

        ctx.trigger(
            "set_view",
            params=dict(view=serialize_view(view)),
        )


class SaveLastAlbumentationsTransform(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="save_albumentations_transform",
            label="Save transform",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        run_key = LAST_ALBUMENTATIONS_RUN_KEY

        info = ctx.dataset.get_run_info(run_key)

        config = info.config.serialize()
        config = {k: v for k, v in config.items() if v is not None}

        transform = _format_transform(config)

        inputs.obj(
            "transform",
            label="Transform",
            description="Serialized description of the transform",
            default=transform,
            required=False,
            view=types.JSONView(readonly=True),
        )

        inputs.str(
            "name",
            label="Name",
            description="The name of the transform. Cannot contain spaces or special characters.",
            required=True,
        )
        view = types.View(label="Save transform")
        return types.Property(inputs, view=view)

    def execute(self, ctx):
        last_run_key = LAST_ALBUMENTATIONS_RUN_KEY

        if last_run_key not in ctx.dataset.list_runs():
            return {
                "message": "To create a transform, use the `Augment with Albumentations` operator",
            }

        transform = ctx.dataset.get_run_info(last_run_key).config.transform
        name = ctx.params.get("name", None)
        _save_transform(ctx.dataset, transform, name)
        ctx.trigger("reload_dataset")


class SaveLastAlbumentationsAugmentations(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="save_albumentations_augmentations",
            label="Save augmentations",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        view = types.View(label="Save augmentations")
        return types.Property(inputs, view=view)

    def execute(self, ctx):
        _save_augmentations(ctx)
        ctx.trigger("reload_dataset")


class GetAlbumentationsRunInfo(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="get_albumentations_run_info",
            label="Get info about a saved Albumentations run",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        run_names = _get_albumentations_run_names(ctx)

        if len(run_names) == 0:
            inputs.view(
                "no_runs_error",
                types.Error(
                    label="No saved Albumentations runs",
                    description="To create a transform, use the `Augment with Albumentations` operator",
                ),
            )
            inputs.str("run_key", required=True, view=types.HiddenView())
            return types.Property(inputs, view=types.View())

        run_choices = types.RadioGroup()
        for run_name in run_names:
            run_choices.add_choice(run_name, label=run_name)

        inputs.enum(
            "run_name",
            run_choices.values(),
            label="Run name",
            description="The run key of the saved Albumentations run",
            view=types.DropdownView(),
            required=True,
        )

        view = types.View(label="Get saved Albumentations run info")
        return types.Property(inputs, view=view)

    def execute(self, ctx):
        run_name = ctx.params.get("run_name", None)
        run_key = _get_run_key_from_name(ctx, run_name)
        return _execute_run_info(ctx, run_key)

    def resolve_output(self, ctx):
        outputs = _initialize_run_output(ctx)
        view = types.View(label="Albumentations run info")
        return types.Property(outputs, view=view)


class DeleteAlbumentationsRun(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="delete_albumentations_run",
            label="Delete Albumentations run",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        run_names = _get_albumentations_run_names(ctx)

        if len(run_names) == 0:
            inputs.view(
                "no_runs_error",
                types.Error(
                    label="No saved Albumentations runs",
                    description="To create a transform, use the `Augment with Albumentations` operator",
                ),
            )
            inputs.str("run_key", required=True, view=types.HiddenView())
            return types.Property(inputs, view=types.View())

        run_choices = types.RadioGroup()
        for run_name in run_names:
            run_choices.add_choice(run_name, label=run_name)

        inputs.enum(
            "run_name",
            run_choices.values(),
            label="Run name",
            description="The run key of the saved Albumentations run",
            view=types.DropdownView(),
            required=True,
        )

        view = types.View(label="Delete saved Albumentations run")
        return types.Property(inputs, view=view)

    def execute(self, ctx):
        run_name = ctx.params.get("run_name", None)
        run_key = _get_run_key_from_name(ctx, run_name)
        ctx.dataset.delete_run(run_key)
        ctx.trigger("reload_dataset")



model_id = "stabilityai/stable-diffusion-2-inpainting"

def _get_target_view(ctx, target):
    if target == "SELECTED_SAMPLES":
        return ctx.view.select(ctx.selected)

    if target == "DATASET":
        return ctx.dataset

    return ctx.view

def create_pipeline(model_id):
  scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
  pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id,
                                                        scheduler=scheduler,
                                                        revision="fp16",
                                                        torch_dtype=torch.float16)

  pipe = pipe.to("cuda")
  pipe.enable_xformers_memory_efficient_attention()
  return pipe

def generate_inputs(im_path,mask_path, mask_id):
  print("the mask id is ===", mask_id)
  #select_class = id2label[mask_id]

  source_image = cv2.imread(im_path)
  print("sourceimg", source_image.shape)
  sd_mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

  out = (sd_mask+(-mask_id*np.ones_like(sd_mask)))
  mask=-(np.clip(1e10*np.multiply(out,out),a_min=0,a_max=255)-255)
  pil_image = Image.fromarray(source_image).resize((512,512))
  pil_mask = Image.fromarray(mask).resize((512,512))
  return pil_image, pil_mask

def augpaint(pipe, prompt, pil_image, pil_mask, num_images_per_prompt, guidance_scale, num_inference_steps):

  # Generate a random seed
  generator = torch.Generator(device="cuda").manual_seed(10)

  # Run stable diffusion pipeline in inpainting mode and store the generated images in a list
  encoded_images = []
  for i in range(num_images_per_prompt):
      image = pipe(prompt=prompt, guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps, generator=generator,
                    image=pil_image, mask_image=pil_mask).images[0]
      encoded_images.append(image.resize((550,825)))
  return encoded_images[0]

def create_hash():
  randint = random.randint(0, 100000000)
  hash = hashlib.sha256(str(randint).encode("utf-8")).hexdigest()[:10]
  return hash

def list_classes(mask_path):

  mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
  list_unique = np.unique(mask)
  return list_unique

def transform_sample(sample, select_class, prompt, num_images_per_prompt, guidance_scale, num_inference_steps):
    id2label = {0: 'nan',1: 'accessories',2: 'bag',3: 'belt',4: 'blazer',5: 'blouse',6: 'bodysuit',7: 'boots',8: 'bra',
            9: 'bracelet',10: 'cape',11: 'cardigan',12: 'clogs',13: 'coat',14: 'dress',15: 'earrings',16: 'flats',
            17: 'glasses',18: 'gloves',19: 'hair',20: 'hat',21: 'heels',22: 'hoodie',23: 'intimate',24: 'jacket',25: 'jeans',
            26: 'jumper',27: 'leggings',28: 'loafers',29: 'necklace',30: 'panties',31: 'pants',32: 'pumps',33: 'purse',
            34: 'ring',35: 'romper',36: 'sandals',37: 'scarf',38: 'shirt',39: 'shoes',40: 'shorts',41: 'skin',42: 'skirt',
            43: 'sneakers',44: 'socks',45: 'stockings',46: 'suit',47: 'sunglasses',48: 'sweater',49: 'sweatshirt',50: 'swimwear',
            51: 't-shirt',52: 'tie',53: 'tights',54: 'top',55: 'vest',56: 'wallet',57: 'watch',58: 'wedges'}
    label2id = {label: id for id, label in id2label.items()}

    hash = create_hash()
    filename = sample.filepath.split("/")[-1][:-4]+"_"+str(hash)+".png"
    pipe = create_pipeline(model_id)
    im,mask = generate_inputs(
        sample.filepath, sample.ground_truth.mask_path,
        label2id[select_class])

    out = augpaint(pipe, prompt, im, mask,num_images_per_prompt, guidance_scale, num_inference_steps)

    shutil.copy(sample.ground_truth.mask_path,
                sample.ground_truth.mask_path[:-4]+"_"+str(hash)+".png",
                )

    new_sample = fo.Sample(
        filepath=sample.filepath[:-4]+"_"+str(hash)+".png",
        ground_truth=fo.Segmentation(
            mask_path=sample.ground_truth.mask_path[:-4]+"_"+str(hash)+".png"),
    )

    return new_sample

def labels_from_id(labels_filepath):
    df = pd.read_csv(labels_filepath)
    id2label = {}
    for i,j in df.iterrows():
        if i==0:
            id2label[i] = 'nan'
        else:
            id2label[i] = j['label_list']
    return id2label


class SDAugment(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="augment_with_sd_inpainting",
            label="Augment with Stable Diffusion Inpainting",
            description="Apply Augmentation with Stable Diffusion Inpainting Model to an image based on a mask found in image.",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Augment with Stable Diffusion Inpainting",
            description="Apply an Stable Diffusion Inpainting to the (image,mask) pair the sample.",
        )

        inputs.int(
            "num_augs",
            label="Number of augmentations per sample",
            description="The number of random augmentations to apply to each sample",
            default=1,
            view=types.FieldView(),
        )

        target_view = ctx.view.select(ctx.selected)
        labels_filepath = "labels.csv"
        id2label = {0: 'nan',1: 'accessories',2: 'bag',3: 'belt',4: 'blazer',5: 'blouse',6: 'bodysuit',7: 'boots',8: 'bra',
            9: 'bracelet',10: 'cape',11: 'cardigan',12: 'clogs',13: 'coat',14: 'dress',15: 'earrings',16: 'flats',
            17: 'glasses',18: 'gloves',19: 'hair',20: 'hat',21: 'heels',22: 'hoodie',23: 'intimate',24: 'jacket',25: 'jeans',
            26: 'jumper',27: 'leggings',28: 'loafers',29: 'necklace',30: 'panties',31: 'pants',32: 'pumps',33: 'purse',
            34: 'ring',35: 'romper',36: 'sandals',37: 'scarf',38: 'shirt',39: 'shoes',40: 'shorts',41: 'skin',42: 'skirt',
            43: 'sneakers',44: 'socks',45: 'stockings',46: 'suit',47: 'sunglasses',48: 'sweater',49: 'sweatshirt',50: 'swimwear',
            51: 't-shirt',52: 'tie',53: 'tights',54: 'top',55: 'vest',56: 'wallet',57: 'watch',58: 'wedges'}
        #id2label = labels_from_id(labels_filepath)
        
        for sample in target_view:
            mask = cv2.imread(sample.ground_truth.mask_path,cv2.IMREAD_GRAYSCALE)
            unique_choices = tuple([id2label[i] for i in np.unique(mask)[1:]])
            
            break

        class_choices = types.Dropdown(label="Class")
        for clas in unique_choices:
            class_choices.add_choice(clas, label=clas)


        inputs.enum(
            "class_choices",
            class_choices.values(),
            default="skin",
            view=class_choices,
        )

        inputs.str(
            "prompt",
            label="Prompt",
            description="The prompt to generate new data from",
            required=True,
        )


        inference_steps_slider = types.SliderView(
                label="Num Inference Steps",
                componentsProps={"slider": {"min": 50, "max": 200, "step": 10}},
            )
        inputs.int(
            "inference_steps",
             default=50,
              view=inference_steps_slider
        )


        guidance_scale_slider = types.SliderView(
                label="Guidance_scale",
                componentsProps={"slider": {"min": 1, "max": 30, "step": 2}},
            )
        inputs.int(
            "guidance_scale",
             default=7,
              view=guidance_scale_slider
        )


        return types.Property(inputs, view=form_view)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        num_images_per_prompt = ctx.params.get("num_augs", 1)
        select_class = ctx.params.get("class_choices", "skin")
        prompt = ctx.params.get("prompt", "None provided")
        guidance_scale = int(ctx.params.get("guidance_scale", 7))
        num_inference_steps = int(ctx.params.get("num_inference_steps", 50))

        label_fields = _get_label_fields_to_transform(ctx)

        target_view = ctx.view.select(ctx.selected)

        for sample in target_view:
            new_sample = transform_sample(sample, select_class, prompt,
                 num_images_per_prompt, guidance_scale, num_inference_steps)
            dataset.add_sample(new_sample)
            break
            

        ctx.trigger("reload_dataset")

# def register(plugin):
#     plugin.register(SDAugment)


def register(plugin):

    plugin.register(SDAugment)
    plugin.register(AugmentWithAlbumentations)
    plugin.register(GetLastAlbumentationsRunInfo)
    plugin.register(ViewLastAlbumentationsRun)
    plugin.register(SaveLastAlbumentationsTransform)
    plugin.register(SaveLastAlbumentationsAugmentations)
    plugin.register(GetAlbumentationsRunInfo)
    plugin.register(DeleteAlbumentationsRun)
