#!/bin/sh
python main.py --benchmark "endless" --scenario "Classes" --algorithm "replay" --global_scheduler_mode "default" --training_bs 16 --eval_bs 16 --semseg
python main.py --benchmark "endless" --scenario "Illumination" --algorithm "replay" --global_scheduler_mode "default" --training_bs 16 --eval_bs 16  --semseg
python main.py --benchmark "endless" --scenario "Weather" --algorithm "replay" --global_scheduler_mode "default" --training_bs 16 --eval_bs 16  --semseg