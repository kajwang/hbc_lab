# Go2+ARX5 四任务迁移指南

## 目标和边界

这份指南面向现有 Robot Lab 任务：

`/home/kaijun/wbc/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/skill/pnp_go2arx5_hier_drc`

第一版只做 **一个任务一个 policy**，依次完成：

1. PnP。
2. 双手推车。
3. 右臂开门。
4. 右臂拉开并关上抽屉。

第一版不做多任务联合训练，不引入 BPS、GraspGen、场景 occupancy、G1 双臂姿态课程或多形状泛化。它们是后续研究变量，不应混入四任务基线。

## 最值得复用的公共代码

### 1. Contact label 接口

参考：

- `source/hbc_lab/hbc_lab/tasks/manager_based/skill/contact_labels.py`
- `source/hbc_lab/hbc_lab/tasks/manager_based/skill/pose_motion.py`

保留最小接口：

```text
effector_mask   哪个末端需要参与任务
target_region   当前阶段末端应接近/接触的位置
target_orientation 当前阶段建议的末端方向（任务需要时启用）
contact_mode    GRASP 或 SUPPORT，由环境选择奖励语义
```

Go2+ARX5 只有一个操作臂时，`effector_mask` 可以退化为一维，但建议保留同一数据结构，后续更换本体时不必重写任务接口。Actor 只接收可部署的目标位姿和自身状态；接触力、DRC 权重和仿真内部质量不进入 actor。

### 2. DRC 阶段权重

参考：

- `g1_dex1_hier_drc/mdp/drc_math.py`
- `g1_dex1_hier_drc/mdp/contact_progress.py`
- 原 Go2 版本的 `mdp/drc_math.py`

沿用 approach/couple/manip 三段连续权重：

```text
W_app    = (1 - c_couple) * (1 - exp(-alpha * d_ee))
W_couple = (1 - c_couple) * exp(-alpha * d_ee)
W_manip  = c_couple
```

其中：

- `d_ee` 必须用 ARX5 的 `gripper_dummy_link` 或真实夹持中心到实时目标的距离。
- `c_couple` 应由夹爪两侧 pad 接触、闭合状态和距离门控组成，再做 EMA；不要只看 wrist 距离。
- 所有接触传感器必须过滤到目标物体/目标 handle，不能把桌面、门框或非目标抽屉的接触算成 grasp。
- 操作阶段应使用物体或关节的累计任务进度，不能只奖励单步位移。

### 3. 物理速率高层动作

参考：`g1_dex1_hier_drc/mdp/high_level_actions.py`。

现有 Go2 动作接口已经支持 base、EE 增量、夹爪和可选 residual。建议把 EE 增量统一改成物理速率：

```text
ee_pose[t+1] = integrate(ee_pose[t], ee_twist[t] * high_level_dt)
```

建议第一版动作：

```text
base velocity target: vx, vy, wz
EE linear velocity:   dx/dt, dy/dt, dz/dt
EE angular velocity:  wx, wy, wz（确有 orientation 需求时）
binary gripper:       open/close
```

必须同时设置：速度上限、加速度上限、EE 工作空间软惩罚、命令速度/加速度惩罚。不要只惩罚 `action[t]-action[t-1]`，因为恒定非零增量能让命令无限漂移但 action-rate 为零。

### 4. Squashed Gaussian 与数值保护

参考：

- `source/hbc_lab/hbc_lab/learning/squashed_gaussian.py`
- `g1_dex1_hier_drc/config/agents/rsl_rl_ppo_cfg.py`
- `g1_dex1_hier_drc/config/g1_dex1_env_cfg.py` 中的 termination 和 finite clip。

这是应当同步到 Go2 的 actor saturation 修复：

1. PPO 在 latent Gaussian 中采样和计算 log-prob。
2. 只在送入环境前执行 `tanh(latent_action)`。
3. 确定性推理也使用 `tanh(actor_mean)`。
4. `log_std` 限制到对应 `std=[0.08, 0.40]` 的范围。
5. Actor 最后一层小尺度初始化，当前实现为 gain `0.01`。
6. 优化器更新前检查参数和梯度是否 finite。
7. 环境侧保留 joint velocity explosion、nonfinite state termination 和有限值裁剪。

不要采用“训练时用未截断动作算概率、环境里再硬 clamp”的半套方案，它会让 PPO 优化的动作与真正执行的动作不一致。

至少记录：

```text
latent_mean_abs
executed_action_abs
action_boundary_095_ratio
action_retention_ratio = mean(1 - tanh(mean)^2)
每个动作块的 boundary ratio
```

如果 Go2 动作维度不同，`posture_boundary_095_ratio` 等诊断切片必须按 Go2 的动作布局重写，不能照搬 G1 的 `[3:5]`。

## 四个任务如何迁移

### PnP：作为公共基线

起点仍是 Robot Lab 的 `pnp_go2arx5_hier_drc`。建议保留已经验证过的：

- `gripper_dummy_link` 作为 EE 中心。
- Link7/Link8 两侧接触和对向受力作为 grasp 依据。
- 二元夹爪控制。
- approach/couple/manip DRC。
- 实时 object pose 和目标 pose。

基础随机化：

- 机器人 reset：小范围 XY 和 yaw。
- 物体初始位置：相对机器人前方采样 X/Y，并随机 yaw。
- 放置目标：从物体出发采样半径和方位，目标放在机器人相对初始物体的另一侧，避免目标平台堵住 approach 路径。
- 物体动力学：先使用质量 curriculum，建立稳定 grasp 后再逐渐降到真实质量并扩大 log-space 方差。
- 摩擦：目标物体和夹爪 pad 使用明确、可复现的 material 配置。

HBC 后续随机化代码可作为可选第二阶段参考：

- `randomized_env_cfg.py`
- `domain_randomization_curriculum.py`
- `events.reset_randomized_object_and_support_platforms`

其最终范围包括物体 X `1.0~2.7 m`、Y `-1.0~1.0 m`、支撑高度 `0.0~0.7 m`、目标距离 `1.5~4.0 m`，以及离散尺寸池。对 Go2 第一版不要一开始全开；先复现旧 PnP，再由 grasp occupancy/成功率逐级放宽。

### 推车：双手设计需降维为单臂语义

参考：

- `g1_dex1_cart_push_hier_drc/config/cart_env.py`
- `g1_dex1_cart_push_hier_drc/mdp/{events,progress,rewards,scenes}.py`
- `assets/articulated_objects.py` 中 `CART_CFG`。

G1 当前设计使用两个 handle target，并要求双手都建立 grasp。Go2+ARX5 只有一只夹爪，因此迁移时改为：

- 一个实时 handle center target。
- 单夹爪两侧 pad 同时接触 handle 形成 `c_couple`。
- `d_goal` 使用实时 handle/cart 基准点到目标的平面距离。
- `R_manip = 0.3 + transport_progress`，并由 `W_manip` 门控。

环境随机化：

- Cart 在机器人前方 `1.5~2.5 m`，带小幅横向扰动。
- 目标相对 cart 前进 `2~4 m`，横向扰动 `-0.5~0.5 m`。
- 轮子和转向关节保持被动 actuator 参数。
- 第一版固定整车质量；若加质量随机化，必须按比例修改所有 link 的质量和惯量，不能只改 root link。

### 开门：两个累计 keyframe

参考：

- `g1_dex1_door_open_hier_drc/config/door_env.py`
- `g1_dex1_door_open_hier_drc/mdp/{events,observations,rewards,scenes}.py`
- `assets/articulated_objects.py` 中 `DOOR_CFG`。

当前任务使用两个 keyframe：

1. 实时 handle pose 绕自身轴旋转到 90 度。
2. 保持 handle 已旋转姿态，门板打开到目标角度（当前成功目标 60 度）。

锁舌在 handle 超过 82 度后释放，避免门板在 keyframe 1 稳定判定前提前泄力。门铰链未解锁时施加 spring-damper 锁定力矩。

Go2 迁移注意：

- target 必须由实时 handle frame 更新；门板转动后，handle 的位置和方向都会变。
- 夹爪接触只过滤 door handle link，门板和门框接触不能满足 grasp。
- Actor 看当前 keyframe target pose，不必看离散 phase id。
- Handle/hinge 角可给 critic 和奖励；若 actor 需要它们，实机必须能从视觉/编码器估计。
- 门的 reset 为前方约 `2.0 m`，Y 小扰动；第一版不随机 scale 和 dynamics。

### 抽屉：随机上下层，先开再关

参考：

- `g1_dex1_drawer_hier_drc/config/drawer_env.py`
- `g1_dex1_drawer_hier_drc/mdp/{events,observations,rewards,scenes}.py`
- `assets/articulated_objects.py` 中 `SEKTION_CABINET_CFG`。

当前设计：

- 每个环境以 50% 概率选上层或下层抽屉。
- Actor 只看被选 handle 的实时 pose 和当前 keyframe target。
- Keyframe 1：沿柜体局部 +X 拉出 `0.30 m`。
- Keyframe 2：回到 reset 时的 closed pose。
- 抽屉只平移，因此 position mask 开启、rotation mask 关闭。
- Contact sensor 静态过滤两个 handle，再按 `selected_drawer` 从 force matrix 中选对应 filter；否则会把另一层把手接触算进来。

环境随机化：柜体位于机器人前方约 `2.0 m`，Y 为 `-0.10~0.10 m`；上下层随机，所有抽屉 joint reset 到零。第一版不要随机抽屉阻尼或行程。

USD 中的关节驱动参数可能覆盖 Python 配置。仿真启动后必须读取实际 joint stiffness/damping/friction 验证，必要时在 startup/reset 后显式写入。当前参考值为 drawer stiffness `1.0`、damping `0.1`、friction `0.1`。

## Keyframe 的通用实现

参考：`source/hbc_lab/hbc_lab/tasks/manager_based/skill/pose_motion.py`。

每个 keyframe 保存：

```text
target position/quaternion
position mask/rotation mask
position/orientation tolerance
stable step count
```

当前阶段误差为 `E_k`，进入阶段时记录 `E_k_start`：

```text
phase_progress = clamp(1 - E_k / max(E_k_start, eps), 0, 1)
cumulative_progress = keyframe_index + phase_progress
normalized_progress = cumulative_progress / number_of_keyframes
```

因此完成第一阶段后，第二阶段即使短暂停止，累计进度也不会掉回零。只有目标位姿连续满足 tolerance 若干步才切换，避免接触抖动造成 phase 来回跳。

实机部署的切换依据必须是可估计的对象 pose/joint state，而不是仿真专有 phase。建议 actor 只观察当前 target pose；环境或上层状态估计器负责根据对象状态切换 target。

## 容易遗漏但必须检查的项目

1. **Frame 一致性**：训练、play、导出和冻结 low-level policy 必须使用同一 EE anchor、同一 quaternion 顺序和同一 root/yaw frame。
2. **实时 pose**：接触目标必须每步更新，不能 reset 后永远使用 initial pose。
3. **Reset 顺序**：资产写入 pose 后要等传感器/FrameTransformer 更新，再缓存 initial pose 和 keyframe。
4. **Contact filter**：逐个断点验证两侧 pad 的 force、目标 filter index 和抓取门控；不要只看可视化接触。
5. **Drive 实值**：读取仿真中的实际 stiffness/damping/friction/joint limit，不能只相信 config。
6. **夹爪语义**：明确 open/close joint target，训练和 play 都必须走同一二元动作路径。
7. **可部署观测**：接触力、真实质量和 DRC 权重只做 reward/critic；actor 使用状态估计、关节编码器、FK 和视觉对象 pose 可获得的信息。
8. **历史观测**：保留 10 帧 actor history 时必须从头训练，旧 checkpoint 的第一层维度不能直接加载；如扩维，只能做明确的零填充 ablation。
9. **动作安全**：保留 low-level action finite 检查和合理 clip，但 clip 值改变后即使网络维度不变，也可能改变旧 checkpoint 行为。
10. **指标**：每个任务至少记录 `d_ee`、两侧 pad contact、`c_couple`、三个 DRC 权重、object/joint progress、success、object fall、action saturation。
11. **可视化**：只显示 actor 真正收到的 target pose 和 keyframe；marker 与物理 frame 分开实现，避免可视化缓存误导调试。
12. **任务成功优先**：motion-quality 正则最多先占当前阶段奖励量级的约 10%，若成功率下降超过 5%，先降低正则而不是继续加约束。

## 推荐实施顺序

1. 将 squashed Gaussian、finite checks 和动作诊断移植到 Robot Lab，先确认旧 PnP 能稳定训练。
2. 把 Go2 EE action 改成按 `high_level_dt` 积分的物理速率，并保持二元夹爪。
3. PnP 从零训练，验证接近、闭合、搬运和质量 curriculum。
4. 推车复用 PnP 单夹爪 contact/DRC，只替换资产、实时 handle target 和 transport progress。
5. 开门加入两段累计 keyframe 和锁舌，先验证 handle 能转，再验证门板进度。
6. 抽屉复用 keyframe 框架，增加上下层随机选择和 contact filter index 选择。
7. 四个任务分别固定成功 checkpoint 后，再做位置、质量、摩擦等随机化；不要边搭任务边同时扩大 domain。

## 最小验证门槛

每个任务正式开 4096 环境前，先完成：

```text
16 env 可视化 smoke
资产 root/joint/link/frame 名称检查
两侧 pad force 断点检查
实际 actuator 参数检查
随机 reset 100 次无穿模/飞散
actor/critic/action 维度打印并固化
100~500 iteration 数值稳定性 smoke
```

满足这些条件后再跑长训练，能避开本项目已经遇到过的大多数低级故障。
