"""
Model initialization utilities.

Note: the large-model branch `large_model_resnet50` adds support for
ImageNet-style backbones (ResNet-50/101/152, EfficientNet-B0) for the
deferred scaling study (R2/R-Cong reviewer requests).

These ImageNet backbones expect 224x224 inputs; for SplitCIFAR-10/100 and
CORe50 (32x32 native) we rely on the dataset transform pipeline to upsample
inputs to 224x224 before they enter the model. The transform-side change is
not made on this branch by design -- it is the trigger Timmy will pull when
the scaling experiment is run.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.models import pytorchcv_wrapper

# Optional pytorchcv direct import for ImageNet-style backbones; only used
# when an ImageNet model is requested via args.model.
try:
    from pytorchcv.model_provider import get_model as _ptcv_get_model
    _PTCV_AVAILABLE = True
except ImportError:
    _PTCV_AVAILABLE = False


# Dataset -> num_classes lookup for ImageNet-style classification heads.
_NUM_CLASSES_BY_BENCHMARK = {
    "split_cifar10": 10,
    "split_cifar100": 100,
    "core50": 10,            # CORe50 uses 10 categories in the OCL setting
    "perm_mnist": 10,
    "endless": 5,
    "soft_robot_ic": 12,
    "soft_robot_il": 5,
}

# ImageNet-style backbones we expose on this branch. The string here is the
# pytorchcv model name; `args.model` is matched against this list.
_IMAGENET_BACKBONES = {
    "resnet50": ("resnet50", "output"),       # pytorchcv key, head attribute name
    "resnet101": ("resnet101", "output"),
    "resnet152": ("resnet152", "output"),
    "resnet200": ("resnet200", "output"),     # if you want even larger
    "efficientnet_b0": ("efficientnet_b0b", "output"),
    # Paper Fig. 5 backbones added on this branch (not from pytorchcv):
    "mobilenetv1": ("mobilenetv1", "output"),      # avalanche.models.MobilenetV1
    "vit_tiny": ("vit_tiny_patch16_224", "head"),  # timm; .output aliased below
}


def _resolve_num_classes(benchmark_name, args):
    """Map benchmark + scenario to a num_classes head size."""
    if benchmark_name == "soft_robot":
        scenario = getattr(args, "scenario_soft_robot", "ic")
        return _NUM_CLASSES_BY_BENCHMARK[f"soft_robot_{scenario}"]
    return _NUM_CLASSES_BY_BENCHMARK.get(benchmark_name, 10)


def _initialize_imagenet_backbone(args, benchmark_name):
    """
    Initialize an ImageNet-style backbone (ResNet-50/101/152, EfficientNet-B0)
    for the deferred large-model scaling study. Re-binds the classification
    head to match the benchmark's num_classes.

    NOTE: these backbones expect 224x224 inputs. The dataset transform
    pipeline is responsible for upsampling 32x32 inputs (SplitCIFAR/CORe50)
    before the model sees them. This branch does not modify the transform
    pipeline -- that is the trigger to pull when running the scaling
    experiment.
    """
    num_classes = _resolve_num_classes(benchmark_name, args)

    # MobileNetV1 (avalanche) and ViT-Tiny (timm) -- the paper's Fig. 5 backbones.
    if args.model == "mobilenetv1":
        from avalanche.models import MobilenetV1
        model = MobilenetV1(pretrained=False)
        in_features = model.output.in_features
        model.output = nn.Linear(in_features, num_classes)
        return model
    if args.model == "vit_tiny":
        from timm import create_model
        model = create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        # The AOCL framework reads/re-binds model.output; timm ViT's classifier is
        # model.head. Alias .output to the real head so downstream code works unchanged.
        model.output = model.head
        return model

    if not _PTCV_AVAILABLE:
        raise ImportError(
            "pytorchcv is required for ImageNet-style backbones on the "
            "large_model_resnet50 branch. Install via: pip install pytorchcv"
        )

    ptcv_name, head_attr = _IMAGENET_BACKBONES[args.model]
    model = _ptcv_get_model(ptcv_name, pretrained=False)

    num_classes = _resolve_num_classes(benchmark_name, args)
    head = getattr(model, head_attr)
    if isinstance(head, nn.Linear):
        in_features = head.in_features
        setattr(model, head_attr, nn.Linear(in_features, num_classes))
    else:
        # EfficientNet-B0 in pytorchcv exposes `output` as a Sequential ending
        # in a Conv2d 1x1; rebuild the final 1x1 with the new num_classes.
        last = head[-1]
        if isinstance(last, nn.Conv2d):
            new_last = nn.Conv2d(
                in_channels=last.in_channels,
                out_channels=num_classes,
                kernel_size=last.kernel_size,
                stride=last.stride,
                padding=last.padding,
                bias=(last.bias is not None),
            )
            head[-1] = new_last
        else:
            raise NotImplementedError(
                f"Unsupported head structure for {args.model}: tail={type(last).__name__}"
            )

    return model


def initialize_model(args, benchmark_name):
    """
    Initialize a model based on the provided arguments.

    Args:
        args: Command-line arguments
        benchmark_name: Name of the benchmark

    Returns:
        Initialized model
    """
    # ImageNet-style backbones (large-model branch addition).
    if args.model in _IMAGENET_BACKBONES:
        model = _initialize_imagenet_backbone(args, benchmark_name)

    # Existing CIFAR-style ResNet path (resnet20/56/110/1001).
    elif args.model.startswith("resnet"):
        if benchmark_name == "perm_mnist":
            dataset_name = "mnist"
        elif benchmark_name == "split_cifar10":
            dataset_name = "cifar10"
        elif benchmark_name == "split_cifar100":
            dataset_name = "cifar100"
        else:
            dataset_name = "cifar10"  # Default resnet

        model = pytorchcv_wrapper.resnet(dataset_name, depth=int(args.model[6:]), pretrained=False)
        if benchmark_name == "split_cifar100":
            in_features = model.output.in_features
            model.output = nn.Linear(in_features, 100)
    else:
        raise ValueError(f"Model {args.model} not supported")

    # Modify model for specific benchmarks.
    if benchmark_name == "endless":
        # endless head re-bind only applies to CIFAR-style ResNets that have
        # the 5184-feature pre-pool tensor; ImageNet backbones already have
        # the right head attached above.
        if args.model not in _IMAGENET_BACKBONES:
            model.output = torch.nn.Linear(5184, 5)
            if args.semseg:
                _setup_semseg_model(model)

    if benchmark_name == "soft_robot" and args.model not in _IMAGENET_BACKBONES:
        model.output = torch.nn.Linear(
            64, 12 if args.scenario_soft_robot == "ic" else 5
        )

    return model


def _setup_semseg_model(model):
    """
    Set up a model for semantic segmentation.

    Args:
        model: Model to be modified
    """
    # Remove final_pool to retain spatial resolution
    model.final_pool = nn.Identity()

    # Modify output layer for segmentation: remove fixed upsampling
    model.output = nn.Sequential(
        nn.Conv2d(64, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 8, kernel_size=1)
    )

    # Override the forward function to upsample dynamically
    def _seg_forward(x):
        input_size = x.shape[-2:]
        x = model.features(x)
        x = model.final_pool(x)
        x = model.output(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x

    model.forward = _seg_forward
