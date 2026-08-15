import tinyimagenet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from avalanche.benchmarks.classic import EndlessCLSim, PermutedMNIST, SplitCIFAR10, SplitCIFAR100, SplitMNIST, SplitFMNIST, SplitCUB200, CORe50
from avalanche.training import Naive, Replay, EWC, GEM, AGEM, GSS_greedy, MIR, SCR, AR1
from avalanche.training.plugins import ReplayPlugin, EvaluationPlugin, GEMPlugin, EWCPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    cpu_usage_metrics,
    gpu_usage_metrics,
    disk_usage_metrics,
    ram_usage_metrics,
    timing_metrics,
    MAC_metrics,
    StreamAccuracy,
    StreamForgetting,
)
from avalanche.logging import InteractiveLogger, CSVLogger
from avalanche.models import pytorchcv_wrapper
import argparse
import random
import numpy as np

import time

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # print beginning timestamp
    print("Starting experiment at", time.strftime("%Y-%m-%d %H:%M:%S"))


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="Classes",
        choices=["Classes", "Illumination", "Weather"],
        help="Select scenario: Classes, Illumination, Weather; only eligible for EndlessCLSim.",
    )
    parser.add_argument(
        "--scenario_core50",
        type=str,
        default="ni",
        choices=["ni", "nc", "nic"],
        help="Select scenario: ni, nc, nic; only eligible for core50.",
    )
    parser.add_argument("--semseg", action="store_true", default=False)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default="endless",
                        choices=["endless", "split_cifar10", "split_cifar100", "core50"])

    # added parameters for testing
    parser.add_argument("--training_bs", type=int, default=16)
    parser.add_argument("--eval_bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--mem_size", type=int, default=50000)
    parser.add_argument("--algorithm", type=str, default="naive", choices=["naive", "replay", "gem", "ewc",
                                                                           "gss_greedy", "agem", "mir", "scr",
                                                                           "ar1"])
    parser.add_argument("--model", type=str, default="resnet20",
                        choices=["simple_mlp", "resnet20", "resnet56", "resnet110", "resnet1001"])
    parser.add_argument("--optimization", type=str, default="none",
                        choices=["none", "gem", "ewc", "both"])
    parser.add_argument("--download_only", action="store_true", default=False)

    args = parser.parse_args()

    if args.semseg:
        import avalanche.evaluation.metrics.accuracy as _acc_mod
        _acc_mod.is_semseg_acc = True  # enable per-pixel accuracy in the custom Avalanche fork

    device = torch.device(f"cuda:{args.cuda}" if args.cuda != -1 else "cpu")

    # Model
    if args.model[:6] == "resnet":
        if args.benchmark == "perm_mnist":
            dataset_name = "mnist"
        elif args.benchmark == "split_cifar10":
            dataset_name = "cifar10"
        elif args.benchmark == "split_cifar100":
            dataset_name = "cifar100"
        else:
            dataset_name = "cifar10" # default resnet
        model = pytorchcv_wrapper.resnet(dataset_name, depth=int(args.model[6:]), pretrained=False)
    else:
        raise ValueError("Model not supported")

    # CL Benchmark Creation
    if args.benchmark == "endless":
        target_transform = None

        # modify output layer to match the number of classes in the benchmark
        model.output = torch.nn.Linear(5184, 5)
        if args.semseg:
            # Remove final_pool to retain spatial resolution
            model.final_pool = nn.Identity()
            # Modify output layer for segmentation: remove fixed upsampling
            model.output = nn.Sequential(
                nn.Conv2d(64, 512, kernel_size=3, padding=1),  # Increase feature depth
                nn.ReLU(),
                nn.Conv2d(512, 8, kernel_size=1)  # Final segmentation output (8 classes)
            )
            # Override the forward function to upsample dynamically
            def _seg_forward(x):
                input_size = x.shape[-2:]  # Get original input spatial dimensions (e.g., 135x240)
                x = model.features(x)  # Extract features (downsampled)
                x = model.final_pool(x)  # Identity here, so retains current feature map size
                x = model.output(x)  # Apply segmentation head (produces [B, num_classes, H_feat, W_feat])
                # Upsample to the original input image size
                x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

                return x
            model.forward = _seg_forward

            # torch.from_numpy(target).long()
            target_transform = lambda x: torch.from_numpy(x).long()

        benchmark = EndlessCLSim(
            scenario=args.scenario,  # choice from ["Classes", "Illumination", "Weather"]
            sequence_order=None,
            task_order=None,
            semseg=args.semseg,
            dataset_root=args.dataset_root,
            target_transform=target_transform,
        )

    elif args.benchmark == "split_cifar10":
        benchmark = SplitCIFAR10(n_experiences=10, fixed_class_order=list(range(10)), shuffle=False, return_task_id=False)
    elif args.benchmark == "split_cifar100":
        benchmark = SplitCIFAR100(n_experiences=10, fixed_class_order=list(range(10)), shuffle=False, return_task_id=False)
    elif args.benchmark == "core50":
        benchmark = CORe50(scenario=args.scenario_core50, mini=True, object_lvl=False)
        # choice from ["ni", "nc", "nic"]
        # ni - new instances, nc - new classes, nic - new instances and classes
        # mini - True for 32x32, False for 128x128
        # object_lvl – True for a 50-way classification at the object level. False if you want to use the categories as classes.
    elif args.benchmark == "perm_mnist":
        # not tested. We don't need to use this benchmark
        benchmark = PermutedMNIST(n_experiences=3)
    else:
        raise ValueError("Invalid benchmark name")

    if args.benchmark == "endless":
        scenario = args.scenario
        input_size = [3, 64, 64]
    elif args.benchmark == "core50":
        # mini version of core50, resolution 32x32
        scenario = args.scenario_core50
        input_size = [3, 32, 32]
    else:
        scenario = 'na'
        input_size = [3, 32, 32]


    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # Prepare for training & testing
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    interactive_logger = InteractiveLogger()

    csv_logger = CSVLogger("log-{}-{}-{}-{}-bs{}-lr{}-mem{}-plugin-{}".format(
        args.benchmark,
        scenario,
        args.model,
        args.algorithm,
        args.training_bs,
        args.lr,
        args.mem_size,
        args.optimization
    ))

    logger = [interactive_logger, csv_logger]

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        ram_usage_metrics(every=True, minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, experience=True, stream=True),
        MAC_metrics(experience=True),
        # cpu_usage_metrics(experience=True, stream=True), # no need
        # gpu_usage_metrics(experience=True, stream=True, gpu_id=0), # hardcode gpu_id to 0
        # disk_usage_metrics(experience=True, stream=True), # huge overhead
        loggers=logger,
    )

    training_plugins = []
    # add replay plugin if algorithm is not replay
    if args.algorithm != "replay":
        training_plugins.append(ReplayPlugin(mem_size=args.mem_size))

    if args.optimization == "gem":
        training_plugins.append(GEMPlugin(patterns_per_experience=1, memory_strength=0.5))
    elif args.optimization == "ewc":
        training_plugins.append(EWCPlugin(ewc_lambda=0.5))
    elif args.optimization == "both":
        training_plugins.append(GEMPlugin(patterns_per_experience=1, memory_strength=0.5))
        training_plugins.append(EWCPlugin(ewc_lambda=0.5))
    elif args.optimization == "none":
        pass
    else:
        raise ValueError("Invalid optimization name")


    # Continual learning strategy
    if args.algorithm == "naive":
        cl_strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
        )
    elif args.algorithm == "replay":
        cl_strategy = Replay(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            mem_size=args.mem_size,
            plugins=training_plugins,
        )
    elif args.algorithm == "gem":
        cl_strategy = GEM(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            patterns_per_exp=1,
            plugins=training_plugins,
        )
    elif args.algorithm == "ewc":
        cl_strategy = EWC(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            ewc_lambda=0.5,
            plugins=training_plugins,
        )
    elif args.algorithm == "gss_greedy":
        cl_strategy = GSS_greedy(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            mem_size=args.mem_size,
            mem_strength=1,
            input_size=input_size,
            plugins=training_plugins,
        )
    elif args.algorithm == "agem":
        cl_strategy = AGEM(
            model,
            optimizer,
            criterion,
            train_mb_size=args.training_bs,
            train_epochs=args.epoch,
            eval_mb_size=args.eval_bs,
            device=device,
            evaluator=eval_plugin,
            patterns_per_exp=1,
            plugins=training_plugins,
        )
    elif args.algorithm in ["mir", "scr", "ar1"]:
        raise NotImplementedError("Algorithm not implemented")
    else:
        raise ValueError("Invalid algorithm name")

    if args.download_only:
        print("Download only mode")
        exit(0)

    import time
    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    latencies = []
    memories = []

    # Extract the stream loss from the evaluation results, rounded to 3 decimal places
    def _stream_loss(res):
        import re

        # Compile a regex to match keys like "Loss_Stream/eval_phase/test_stream/TaskXXX"
        # where XXX is a three-digit number.
        pattern = re.compile(r"Loss_Stream/eval_phase/test_stream/Task\d{3}")

        loss_stream = 0.0
        try:
            value = next(
                    value for key, value in res.items() if pattern.fullmatch(key)
                )
        except StopIteration:
            raise KeyError(f"No key matching pattern found in: {res}")
        finally:
            loss_stream = value * 1.0
            # print("Loss stream: ", loss_stream)

        return loss_stream

    # Extract the top1 stream accuracy from the evaluation results, rounded to 3 decimal places
    def _stream_accuracy(res):
        import re

        # Compile a regex to match keys like "Top1_Acc_Stream/eval_phase/test_stream/TaskXXX"
        # where XXX is a three-digit number.
        pattern = re.compile(r"Top1_Acc_Stream/eval_phase/test_stream/Task\d{3}")

        top1_stream_accuracy = 0.0
        try:
            value = next(
                    value for key, value in res.items() if pattern.fullmatch(key)
                )
        except StopIteration:
            raise KeyError(f"No key matching pattern found in: {res}")
        finally:
            top1_stream_accuracy = value * 100.0
            # print("Top1 accuracy: ", top1_stream_accuracy)

        return top1_stream_accuracy

    begin_time = time.time()

    for experience in train_stream:
        cl_strategy.train(experience)
        # torch cuda memory
        memories.append(torch.cuda.memory_allocated(device))
        res = cl_strategy.eval(test_stream)
        results.append(res)
        latencies.append(time.time() - begin_time)

    print("Training completed")
    # print latencies round to 3 decimal places
    latencies = [round(x, 3) for x in latencies]
    print("Latencies: ", latencies)

    # print memories in MB and round to 3 decimal places
    memories = [round(x / 1024**2, 3) for x in memories]
    print("Memories: ", memories)

    top1 = []
    # print top1 accuracy
    for res in results:
        top1.append(_stream_accuracy(res))

    print("Top1 accuracy:", top1)

    loss_stream = []
    # print stream loss
    for res in results:
        loss_stream.append(_stream_loss(res))

    print("Stream loss:", loss_stream)

    # print End timestamp
    print("Ending experiment at", time.strftime("%Y-%m-%d %H:%M:%S"))