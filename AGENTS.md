# R2V-Replay Codex Workspace Guide

本文件是 `/Users/azure/Documents/adg改` 的项目级执行规范。任何 Codex 进入本 workspace 后，必须先阅读本文件，再执行代码、Git、远端同步或实验任务。

## 1. Project Background

本项目研究 **R2V-Replay: Rare-to-Valuable Experience Mining for Sparse-Reward Replay**。

核心场景是 sparse-reward reinforcement learning with experience replay。Replay buffer 中有大量真实但低信息量的 zero-reward transitions，也有少量 rare valuable experiences，例如 success proximity、task progress、high return、credit-assignment-relevant precursor transitions。

核心命题：

- Sparse-reward replay 的瓶颈不是 transition 数量不足，而是 useful learning signal 被大量 common zero-reward experience 淹没。
- Rarity 只说明“不常见”，不说明“有价值”。
- R2V-Replay 的目标是从 rare candidates 中筛出 task-relevant、risk-controlled、consistency-checked 的 valuable replay subset。
- Diffusion 在本项目中首先是 high-recall rare candidate discovery 工具，不是最终 value judge。
- Generation 可以作为后续 baseline 或扩展用途，但不是第一轮实现，也不是主贡献。

当前项目阶段：

- Related-work / flow-matching gate 已经 `complete enough`。
- 本轮不要做 broad literature search。
- 当前重点是实现 isolated Level 0 No-RL diagnostic，验证 R2V 机制链条是否成立。

## 2. Non-Negotiable Research Boundaries

以下边界不可违反：

- `zero-reward transitions != corrupted data`
  真实 replay 中的 zero-reward transitions 是真实环境交互，只是大多信息量较低。不要称它们为 dirty、corrupted、bad data、polluted data。
- `rare != valuable`
  本项目不能假设 rare sample 一定 valuable。核心 claim 正是 rare candidates 中同时存在 rare valuable 和 rare useless。
- `diffusion != final value judge`
  Diffusion rarity score 只负责 high-recall candidate discovery。最终 replay admission 必须结合 task utility、OOD/support risk、dynamics consistency、reward consistency。
- 不要照搬 ADG clean/dirty data 语义。
  ADG 可以启发 diffusion-as-detector，但 R2V 的 ER setting 中 zero-reward replay 不是 dirty data，不是 corrupted data recovery。
- 不要把方法写成 `SynthER + trivial filter`。
  SynthER 的中心是学习 replay distribution `p_D(x)` 并 upsample / generate replay。R2V 的中心是 sparse-reward replay 中的 rare-to-valuable mining and safe replay admission。
- 不要把第一轮写成 generation 方法。
  第一轮不生成 replay transition。Diffusion scorer 只输出 rarity score。Generation 推迟到后续 baseline 或 extension。
- 不要只设计大实验。
  先做 Level 0 No-RL diagnostic，再考虑 Level 1 toy RL。CleanRL、SynthER、CORL、D4RL、高维视觉任务都推迟。

## 3. Role Split

### Pro

Pro 是研究 PI、审稿 gate、实验总设计师，负责：

- 固定研究 idea 和论文边界。
- 判断 novelty 是否足够。
- 设计最小可行实验和 paper-scale roadmap。
- 审查 Codex 的代码结果、日志、图、失败样本。
- 做 `go / revise / stop` 决策。
- 决定何时从 Level 0 进入 Level 1 / Level 2。

### Codex

Codex 是代码实验总管，负责：

- 在本地 workspace 中实现代码。
- 检查 repo 状态、路径、依赖、命令。
- 编写可复现实验脚本、测试、日志和 artifacts。
- 运行 Level 0 diagnostic。
- 生成 metrics、plots、failure cases、报告包。
- 必要时同步 GitHub `dder` 和远端 Ubuntu smoke test。
- 每轮向 Pro 汇报 evidence，而不是自行扩大研究方向。

Codex 不应擅自：

- 改论文主 claim。
- 做 broad literature search。
- 引入大型 RL repo。
- 开始 RL training。
- 引入 synthetic generation。
- 把已有文献产物混入实验代码。
- 在没有 Pro `go` 决策前进入 Level 1 / Level 2。

## 4. Current Implementation Decision

当前决策：`go`。

第一轮实现路线：

- 在当前 workspace 下新建隔离实验代码目录：`experiments/r2v_replay/`。
- 实现 Level 0 No-RL diagnostic。
- 不训练 RL agent。
- 不 clone / survey CleanRL、SynthER、CORL。
- 不实现 PER。
- 不实现 HER。
- 不实现 D4RL / high-dimensional observation。
- 不实现 generated replay。
- 不接 online / offline RL training loop。

第一轮目标是验证：

```text
rare candidate discovery
-> rare != valuable
-> task utility improves selected valuable ratio
-> risk / consistency admission reduces harmful candidates
```

进入下一阶段前，必须先提交 Level 0 artifacts 给 Pro 审查。

## 5. Local Layout And Files To Avoid

Workspace 根目录：

```text
/Users/azure/Documents/adg改
```

已有文献产物：

```text
literature_story_mining/
flow_literature_mining/
```

这两个目录是已有 related-work / literature mining 产物。第一轮实验代码不得写入、移动、重命名、删除、格式化或混入这些目录。

推荐首轮实验目录：

```text
experiments/r2v_replay/
```

首轮最小目录结构：

```text
experiments/r2v_replay/
  README.md
  requirements.txt
  configs/
    level0_sparse_grid.yaml

  r2v_replay/
    __init__.py
    sparse_grid.py
    replay_dataset.py
    encoders.py
    rarity_scorers.py
    utility_scorers.py
    risk_scorers.py
    consistency_scorers.py
    selector.py
    diagnostics.py
    plotting.py
    utils.py

  scripts/
    build_sparse_grid_replay.py
    run_level0_diagnostic.py
    plot_level0.py
    package_level0_report.py

  tests/
    test_sparse_grid.py
    test_replay_dataset.py
    test_scorers.py
    test_selector.py

  outputs/
  artifacts/
  reports/
```

`outputs/`、`artifacts/`、大型 zip 报告、缓存文件不得提交到 Git，除非 Pro 明确要求提交小型 summary markdown。

建议把已有文献目录加入本地 exclude，而不是默认写进 repo `.gitignore`：

```bash
printf "\nliterature_story_mining/\nflow_literature_mining/\n" >> .git/info/exclude
```

## 6. Code Quality And Command Etiquette

Codex 执行规则：

- 先查状态，再改代码。

```bash
cd "/Users/azure/Documents/adg改"
pwd
git status --short
```

- 小步提交，小 diff。每个实现阶段保持 reviewable diff，不要一次性引入大框架。
- 先搜索已有路径 / API，再假设。不要编造未知路径、远端路径、服务器用户名、密钥位置、repo URL。
- 行为变化必须有测试。新增 dataset、scorer、selector、plotter 时，至少补对应 smoke test 或 unit test。
- 先跑最快相关检查，优先 `pytest -q`，再跑单 seed diagnostic，最后才跑多 seed。
- 不要提交 secrets。不要把 token、SSH key、服务器 IP、私有路径、`.env`、credential 文件写入代码或日志。
- 不要做未经请求的 telemetry / network calls。除非任务明确要求，不要联网下载大型依赖、clone 新 repo、调用外部 API、上传日志。
- 实验必须可复现。每个 run 必须记录 command、config path、seed、git branch、commit hash、Python version、numpy/sklearn version、machine info、start/end time、metrics csv。
- 不要使用 `git add .`。应精确添加需要提交的文件，例如：

```bash
git add .gitignore experiments/r2v_replay
```

添加前必须检查：

```bash
git status --short
```

- 不要把 eval labels 用进 scorer。`label_for_eval_only` 只能用于 metrics 和 plotting。任何 scorer、selector、model training 不能读取 label 作为输入。

## 7. Level 0 Experiment Contract

### 7.1 Dataset Contract

第一轮使用自定义 sparse grid replay。

环境建议：

- `grid size`: `15x15` 或 `17x17`
- `start region`: 左下角 `2x2`
- `goal`: 右上角固定位置
- `obstacles`: 形成 narrow corridor / bottleneck
- `decoy region`: dead-end 或 side corridor
- `max episode length`: `60` 或 `80`
- `reward`: 只有到达 goal 时 `r = 1`，其余 `r = 0`
- `done`: 到达 goal 或达到最大步数

Replay 必须由真实环境交互生成，不能先按 label 采样 transition。

行为策略 mixture 建议：

- `70% random_wander_policy`
- `15% noisy_goal_seeking_policy`
- `10% decoy_seeking_policy`
- `5% near_success_policy`

生成后再 post-hoc 标注 eval labels。

每条 transition 至少保存：

```text
state
action
reward
next_state
done
episode_id
timestep
behavior_policy_id
distance_to_goal
next_distance_to_goal
return_to_go
label_for_eval_only
```

### 7.2 Evaluation Labels

`common_zero`

- 真实 transition。
- `reward == 0`
- `episode_return == 0`
- not in decoy region
- not within `K_pre` steps before success
- distance_to_goal not clearly decreasing
- 这是大量真实但低信息量 transition，不是 corrupted data。

`rare_valuable_positive`

- 真实 transition。
- `reward == 1`
- `done == true`
- `next_state == goal`
- 目标比例：`0.5% - 2%`

`rare_valuable_zero_precursor`

- 真实 transition，reward 仍然为 0，但属于成功前驱。
- `reward == 0`
- `episode_return == 1`
- `steps_to_success <= K_pre`
- `K_pre = 8` 或 `10`
- 并且至少满足一个辅助条件：
  - `distance_to_goal - next_distance_to_goal >= 1`
  - state 位于通向 goal 的 critical corridor / doorway / bottleneck
- 目标比例：`3% - 6%`

`rare_useless`

- 真实 transition。
- `reward == 0`
- `episode_return == 0`
- state or next_state in decoy dead-end / rare side corridor
- 通常还应满足 distance_to_goal 不下降，或 next_state 更远离 goal。
- 目标比例：`3% - 8%`

`optional_invalid`

- 只用于 H4 stress test，不属于真实 replay 主数据。
- 生成方式：
  - 复制真实 `(s, a)`
  - 随机替换 `s'`
  - 随机翻转 reward
  - 制造 teleport next_state
  - 制造 impossible wall-crossing next_state
- 比例：`2% - 5% stress-only`
- 硬规则：
  - optional_invalid 不参与 diffusion training。
  - optional_invalid 不参与 dynamics model training。
  - optional_invalid 不计入真实 replay。
  - optional_invalid 只用于 risk / consistency stress evaluation。

目标总体比例：

```text
common_zero:                    85% - 92%
rare_valuable_positive:          0.5% - 2%
rare_valuable_zero_precursor:    3% - 6%
rare_useless:                    3% - 8%
optional_invalid:                2% - 5% stress-only, separate split
```

### 7.3 Required Scorers

第一轮必须实现：

- `TransitionEncoder`
- `DiffusionRarityScorer`
- `KNNRarityScorer`
- `TaskUtilityScorer`
- `OODRiskScorer`
- `DynamicsConsistencyScorer`
- `R2VSelector`

`TransitionEncoder` 最小支持两种输入：

- `obs_action_next`: `[s, a, s', done]`
- `full_transition`: `[s, a, r, s', done]`

默认使用 `obs_action_next`，避免 reward shortcut。`full_transition` 只作为 ablation。

`TaskUtilityScorer` 第一轮只用：

```text
U_task =
    w_r   * I[r > 0]
  + w_RTG * normalized_return_to_go
  + w_P   * normalized_progress

progress = max(0, distance_to_goal - next_distance_to_goal)
```

必须保存每个 utility component。

### 7.4 Required Baselines

第一轮必须包含：

- `uniform_random`
- `reward_only`
- `rtg_only`
- `progress_only`
- `knn_rarity_only`
- `diffusion_rarity_only`
- `random_from_diffusion_rare_topk`
- `r2v_knn`
- `r2v_diffusion`

第一轮推迟：

- PER
- HER
- SynthER-style upsampling
- generated replay
- no-filter synthetic replay as RL baseline
- segment encoder
- critic value / advantage scorer
- ensemble OOD / dynamics
- pixel observation
- MiniGrid
- CleanRL
- CORL
- D4RL

### 7.5 H1-H4 Metrics And Thresholds

`H1: Diffusion rarity high-recall candidate discovery`

Definitions:

```text
rare_any = rare_valuable_positive
         + rare_valuable_zero_precursor
         + rare_useless

rare_valuable = rare_valuable_positive
              + rare_valuable_zero_precursor
```

Metrics:

- `AUROC(common_zero vs rare_any)`
- `Recall@Top10%(rare_any)`
- `Recall@Top10%(rare_valuable)`
- `Recall@Top20%(rare_valuable)`
- `Enrichment@Top10%(rare_valuable)`

Pass thresholds:

- `AUROC(common_zero vs rare_any) >= 0.75`
- `Recall@Top10%(rare_any) >= 0.60`
- `Recall@Top10%(rare_valuable) >= 0.50`
- `Recall@Top20%(rare_valuable) >= 0.75`
- `Enrichment@Top10%(rare_valuable) >= 2.0x random`

If diffusion is close to random but kNN succeeds, revise diffusion scorer instead of rejecting R2V.

`H2: rare-only must mix rare useless`

Metrics:

- `rare_useless_fraction_in_diffusion_top10`
- `valuable_precision_in_diffusion_top10`
- `rare_useless_enrichment_vs_uniform`

Pass threshold:

- `rare_useless_fraction_in_diffusion_top10 >= 15%`
- or `rare_useless_enrichment_vs_uniform >= 2.0x`

If diffusion Top10% is almost all rare valuable, the dataset is too easy and cannot prove `rare != valuable`; revise dataset.

`H3: task utility filtering improves valuable ratio`

Compare:

- `diffusion_rarity_only`
- `reward_only`
- `rtg_only`
- `progress_only`
- `r2v_diffusion_without_risk`
- `r2v_diffusion_full`

Metrics:

- `valuable_precision@budget`
- `rare_valuable_zero_precursor_recall@budget`
- `positive_reward_fraction_selected`
- `selected_zero_reward_valuable_fraction`

Pass thresholds:

- `R2V valuable_precision >= 1.5x diffusion_rarity_only`
- or `R2V valuable_precision - diffusion_rarity_only >= 20 percentage points`
- `R2V retains >= 50% of rare_valuable_zero_precursor found by rarity Top10%`
- `R2V selected set positive_reward_fraction <= 50%`
- `R2V rare_valuable_zero_precursor_recall >= 2x reward_only`

These conditions prevent R2V from degenerating into reward-only replay.

`H4: risk / consistency admission reduces harmful candidates`

H4 uses optional_invalid stress split only. Invalid transitions are not real replay.

Metrics:

- `AUROC(invalid vs real) by R_ood`
- `AUROC(invalid vs real) by E_dyn`
- `AUROC(invalid vs real) by E_reward`
- `invalid_selection_rate_without_admission`
- `invalid_selection_rate_with_admission`
- `false_rejection_rate_on_rare_valuable_zero_precursor`

Pass thresholds:

- `AUROC(invalid vs real) by max(E_dyn, E_reward, R_ood) >= 0.85`
- `invalid_selection_rate_with_admission <= 20% of invalid_selection_rate_without_admission`
- `false_rejection_rate_on_rare_valuable_zero_precursor <= 30%`

Admission should reduce at least 80% invalid selection while keeping at least 70% zero-reward valuable precursors.

### 7.6 Artifacts To Report

After Level 0, Codex must report:

```text
outputs/seed*/diagnostic_*/metrics.csv
outputs/seed*/diagnostic_*/score_table.csv
outputs/seed*/diagnostic_*/composition_summary.json
outputs/seed*/diagnostic_*/selector_summary.json
outputs/seed*/figures/
outputs/seed*/reports/level0_report.md
outputs/seed*/reports/level0_report.zip
```

Required figures:

- `score_hist_by_label.png`
- `rare_candidate_composition.png`
- `utility_vs_rarity_scatter.png`
- `risk_vs_utility_scatter.png`
- `precision_recall.png`
- `selected_composition_bar.png`

Required failure files:

- `top_selected_examples.csv`
- `top_failure_cases.csv`
- `rejected_valuable_precursors.csv`
- `selected_rare_useless.csv`

## 8. GitHub And Ubuntu Handoff Rules

GitHub remote repository name: `dder`.

Do not assume local `origin` is already configured. Verify first:

```bash
git remote -v
git branch --show-current
git status --short
```

If `origin` is missing, ask the user or inspect available repo URL before setting it. Do not invent owner/name or SSH URL.

Before pushing:

- Confirm only intended files are staged.
- Do not stage `literature_story_mining/` or `flow_literature_mining/`.
- Do not stage `outputs/`, `artifacts/`, large reports, model checkpoints, or local virtualenvs.
- Do not include secrets or private server paths.

Recommended local flow:

```bash
git status --short
git add AGENTS.md .gitignore experiments/r2v_replay
git status --short
git commit -m "Add R2V level0 diagnostic scaffold"
git remote -v
git push -u origin main
```

Ubuntu handoff rules:

- Do not assume server hostname, username, repo path, conda env, SSH key, or credential.
- If secrets are needed, ask the user to provide them via environment variables or existing SSH config.
- Prefer reproducible command logs.
- On remote, record `hostname`, `pwd`, `git rev-parse HEAD`, Python version, numpy/sklearn version, and exact command.
- Run smoke tests first, not full experiments.

Remote smoke-test template, only after user confirms server access details:

```bash
git clone <dder_repo_url>
cd dder
python3 -m venv .venv
source .venv/bin/activate
pip install -r experiments/r2v_replay/requirements.txt
pytest -q experiments/r2v_replay/tests
python experiments/r2v_replay/scripts/build_sparse_grid_replay.py --help
python experiments/r2v_replay/scripts/run_level0_diagnostic.py --help
```

## 9. Pro Checkpoint Loop

After each implementation/result stage, Codex must package evidence and ask Pro for `go / revise / stop`.

Use this checkpoint format:

```text
R2V-Replay Round 1 Level 0 Diagnostic Report

workspace:
branch:
commit:
local run machine:
ubuntu smoke test: pass | fail | not run

1. Code status
- implemented modules:
- tests passed:
- known issues:

2. Dataset summary
- config path:
- number of transitions:
- number of episodes:
- label distribution:
- success episode ratio:
- optional_invalid handling:

3. H1 result: diffusion rarity high-recall candidate discovery
- AUROC common vs rare:
- Recall@Top10 rare_any:
- Recall@Top10 rare_valuable:
- Recall@Top20 rare_valuable:
- enrichment:
- comparison with kNN:
- pass | fail:

4. H2 result: rare != valuable
- rare_useless fraction in diffusion Top10:
- rare_useless enrichment:
- valuable precision in rarity-only:
- pass | fail:

5. H3 result: task utility filtering
- R2V valuable precision:
- rarity-only valuable precision:
- reward-only zero-reward precursor recall:
- R2V zero-reward precursor recall:
- positive reward fraction selected:
- pass | fail:

6. H4 result: risk / consistency admission
- invalid AUROC:
- invalid selection reduction:
- false rejection rate on zero-reward precursor:
- pass | weak pass | fail:

7. Figures and artifacts
- report zip:
- metrics csv:
- key plots:
- top selected examples:
- top failure cases:

8. Codex recommendation
go | revise | stop

9. Codex questions for Pro
- question 1:
- question 2:
```

Pro next-round decision rules:

```text
go:
H1-H3 pass, H4 pass or weak pass, and plots clearly show rare != valuable.

revise:
Only one of H1-H3 is weak, or H4 fails but does not affect real replay diagnostic.

stop:
H1 fails and kNN/diffusion are both close to random;
H2 fails because rare-only is almost all valuable;
H3 fails because R2V degenerates into reward-only;
data or code has label leakage.
```

## 10. Stop And Revise Rules

Stop or revise before RL training if:

- Rarity cannot high-recall find rare valuable transitions.
- Rare-only does not mix rare useless transitions.
- R2V degenerates to reward-only replay.
- Risk gate kills valuable precursors.
- Dataset labels leak into scorer/model/selector.
- Optional invalid transitions are mixed into real replay training.
- Results pass only because `full_transition` uses reward shortcut.
- Figures cannot visually support `rare != valuable`.

Concrete revise actions:

- Diffusion fails but kNN succeeds:
  revise diffusion scorer, score type, normalization, training epochs, or embedding. Do not reject R2V yet.
- H2 fails because dataset is too easy:
  increase decoy policy, dead-end region, rare side corridor, or lower near-success policy ratio.
- Utility selects only immediate reward:
  lower `w_r`, increase RTG/progress weight, audit rare_valuable_zero_precursor.
- Risk / consistency over-penalizes rare valuable:
  replace hard gate with soft penalty, use quantile threshold, report false rejection.
- Metrics pass only on one seed:
  run at least seeds `0, 1, 2` and report mean/std.

Allowed next steps only after Pro explicitly returns `go`:

- Level 1 toy RL
- CleanRL or self-written DQN
- PER baseline
- online replay wrapper
- Level 2 repo survey
- SynthER / CORL inspection
- D4RL / Minari feasibility
- high-dimensional observation diagnostic
- generation baseline

Until then, all work stays inside Level 0 No-RL diagnostic.
