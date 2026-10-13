# Historical development notes

These commands refer to pre-refactor scripts, local paths, and legacy checkpoint
formats. They are preserved for provenance and are not runnable release guidance.
Use the [paper release README](README.md), [release audit](docs/validation/paper-release-audit.md),
and [native experiment guide](projects/corl2026_cg/README.md) for this branch.

## Preserved commands

python scripts/policies/convert_hdf5_to_small_files.py \
  --input_dir /mnt/ssd/yixiao/cg \
  --output_root /mnt/ssd/yixiao/cg/datasetv2/processed_all \
  --no_images \
  --demos_per_shard 10 \
  --overwrite \
  --mix_tasks \
  --mixed_task_slug state


python scripts/policies/test_robot_flow_trainer_v2_state.py   --config src/policies/config/default_run_config_state_only_ac20_k8.json


python scripts/policies/test_robot_flow_trainer_v2_state.py   --config src/policies/config/default_run_config_state_convert_in_robot_dataset.json


python scripts/policies/test_robot_flow_trainer_v2_state.py   --config src/policies/config/default_run_config_state_direct_select_ac20_k8.json


python scripts/policies/test_robot_flow_trainer_v2_state.py   --config src/policies/config/default_run_config_state_direct_all_10kstep.json


python src/policies/tools/convert_gen_to_small_files.py   --input_root /mnt/ssd/yixiao/cg/4tasks/gen_20260415-162543/task   --output_root /mnt/ssd/yixiao/cg/4tasks/gen_20260415-162543/smallfiles5   --demos_per_shard 5   --overwrite

python scripts/policies/test_robot_flow_trainer_v2.py   --config src/policies/config/task4_test.json


python scripts/policies/test_robot_flow_trainer_v2.py --config /home/yixiao/vla/cg/robopolicy/src/policies/config/test_models/cfg_l4_1s_S_ft_self.json ; python scripts/policies/test_robot_flow_trainer_v2.py --config /home/yixiao/vla/cg/robopolicy/src/policies/config/test_models/cfg_l4_1s_S_ft_self_150k.json


python scripts/policies/test_robot_flow_trainer_v2.py --config /home/yixiao/vla/cg/robopolicy/src/policies/config/test_models/cfg_l4_1s_S_ft_cross.json ; python scripts/policies/test_robot_flow_trainer_v2.py --config /home/yixiao/vla/cg/robopolicy/src/policies/config/test_models/cfg_l4_1s_S_ft_cross_150k.json


python scripts/sweeps/rewrite_run_config_folder_paths.py   --in_dir  /home/yixiao/vla/cg/robopolicy/src/policies/config/cg_task_set/1s   --out_dir /home/yixiao/vla/cg/robopolicy/src/policies/config/cg_task_set/1s_rewritten   --dataset-dir /mnt/localssd/vla/cg/cg_l4_1s   --runs-dir /mnt/localssd/vla/ckpt/cg   --pattern "*.json"

CUDA_VISIBLE_DEVICES=1 python scripts/policies/test_robot_flow_trainer_v2.py --config /home/yixiao/vla/cg/robopolicy/src/policies/config/cg_task_set/1s_rewritten/l4_1s_diag_cor_transformer_cross.json

CUDA_VISIBLE_DEVICES=2 python scripts/policies/test_robot_flow_trainer_v2.py --config /home/yixiao/vla/cg/robopolicy/src/policies/config/cg_task_set/1s_rewritten/l4_1s_L_cor_transformer_cross.json

CUDA_VISIBLE_DEVICES=3 python scripts/policies/test_robot_flow_trainer_v2.py --config /home/yixiao/vla/cg/robopolicy/src/policies/config/cg_task_set/1s_rewritten/l4_1s_S_transformer_cross.json

CUDA_VISIBLE_DEVICES=4 python scripts/policies/test_robot_flow_trainer_v2.py --config /home/yixiao/vla/cg/robopolicy/src/policies/config/cg_task_set/1s_rewritten/l4_1s_Sfull_transformer_cross.json






python scripts/sweeps/rewrite_run_config_folder_paths.py \
  --in_dir  /home/yixiao/vla/cg/robopolicy/src/policies/config/cg_task_set/1s \
  --out_dir /home/yixiao/vla/cg/robopolicy/src/policies/config/cg_task_set/1s_rewritten \
  --dataset-dir /mnt/localssd/vla/cg/cg_l4_1s \
  --runs-dir /mnt/localssd/vla/ckpt/cg \
  --pattern "*.json"
