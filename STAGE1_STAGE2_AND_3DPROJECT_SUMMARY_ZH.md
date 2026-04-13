# Stage1 / Stage2 与 3DProject 适配总结

这份文档总结了三部分内容：

- `Latent-Sketchpad` 里的 `stage1` 和 `stage2` 各自到底训练什么
- `stage2` 的监督信号（GT）来自哪里
- 这次把 `3dproject` 数据接入 `Latent-Sketchpad` 的具体工作内容

## 当前 Qwen3.5 补充说明

这份文档最初主要记录 `Qwen2.5-VL` 路线。当前已经额外完成了一条独立的 `Qwen3.5-4B` 路线，关键状态如下：

- 已建立独立环境：`sketchpad-qwen35-cu128`
- 当前长期训练统一改为：`DeepSpeed ZeRO-2`
- 当前新的 `stage1` 冻结策略已经对齐到用户在 `LLaMA-Factory` 那轮 full-SFT 的实际日志行为：
  - 冻结 `model.visual`
  - 冻结 `model.visual.merger`
  - 训练语言模型，包括 `embed_tokens`
- 当前这版 `stage1` 的可训练参数量已经与 `LLaMA-Factory` 对齐：
  - `4,205,751,296`

另外还确认了一个很关键的问题：

- 当前 `stage2` 最终模型之所以没有进入生图模式，不是因为 `max_new_tokens` 太短，也不是因为 `<image>` 被 decode 时隐藏了
- 真正原因是：模型根本没有生成任何图像 special token
  - `vision_start_token_id`
  - `image_token_id`
  - `vision_end_token_id`

这意味着当前生成链没有真正切换到“图像生成模式”。

而当前的最小修正方向是：

- 在 `stage1` 中保留 `vision_start` 的监督
- 先让模型学会何时输出“开始生成图像”的切换 token
- 再继续推进 `stage2` 的完整生成能力

已经确认到的标签监督情况是：

- `vision_start`：现在已保留监督
- `image_pad`：仍不监督
- `vision_end`：仍不监督

因此当前最优先的工作是：

1. 跑完新的 `stage1` 对齐训练
2. 验证新 `stage1` 模型是否开始生成 `vision_start`
3. 再决定 `stage2` 是继续补稳定版切换逻辑，还是直接进入完整版 perceiver 路线

## 1. 整体区别

`Latent-Sketchpad` 不是普通的 VLM SFT。

它是在基础 VLM 的文本生成能力之外，再增加一条中间视觉特征预测链路：

- 通过 `regression_head` 预测中间视觉特征
- 再通过预训练好的 Sketch Decoder 把这些视觉特征解码成中间图像

所以它的两阶段训练是明确分工的：

- `stage1`：任务/文本对齐阶段
- `stage2`：中间视觉 latent / sketch 预测阶段

## 2. Stage1 训练什么

### 目标

`stage1` 主要训练模型去学会：

- 任务的输出格式
- 长推理文本的组织方式
- 最终答案的输出格式

也就是说，`stage1` 更像是先让模型变成一个“会按这个任务要求说话和回答”的模型。

### 代码层面的实际行为

在 `train.py` 里：

- 启用 `--stage1`
- 如果输出目录里不包含 `stage2`，则会设置 `ignore_image = True`

在 `data/dataset.py` 里：

- 当 `self.stage1` 为真时，部分样本会被变成更偏文本的版本
- 多余的 `<image>` 占位会被移除
- 某些样本的 `label_img` 会被清空

所以 `stage1` 并不是主要训练“中间图生成”，而是更偏向：

- 保留任务输入图文上下文
- 强化文本 reasoning 和最终答案输出

### 冻结什么、训练什么（以当前 Qwen 路线为准）

下面这部分是根据 `train.py` 里实际的 `requires_grad` 逻辑整理的。

#### 第一步：默认先全部解冻

代码一开始会先执行：

- `for param in model.parameters(): param.requires_grad = True`

也就是说，初始状态是：

- 整个模型默认全部可训练

#### 第二步：冻结 embedding

如果没有显式传 `--text_embed`，则会冻结文本 embedding：

- Qwen 路线冻结：`model.model.language_model.embed_tokens`

所以默认情况下：

- 文本 embedding 不训练

#### 第三步：冻结视觉 backbone

对 Qwen 路线，代码会固定冻结：

- `model.visual`
- `model.vision_tower`

这意味着：

- 原始视觉编码器不训练
- 视觉主干不训练

#### 第四步：connector 的处理

如果传了 `--unfreeze-connector`，则会把下面这部分重新设为可训练：

- `model.visual.merger.mlp`

所以对当前本地 `stage1` 训练来说：

- 视觉主干冻结
- connector 中的 `visual.merger.mlp` 允许训练

#### 第五步：regression_head 总是解冻

无论 stage1 还是 stage2，代码都会执行：

- `for param in model.regression_head.parameters(): param.requires_grad = True`

也就是说：

- `regression_head` 默认总是训练的

#### 第六步：如果启用 LoRA

如果传了 `--use-lora`，那么 LoRA 会挂到这些文本注意力投影层上：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

并且有明确排除：

- 名字里包含 `visual`
- 名字里包含 `vision_tower`
- 名字里包含 `regression_head`

也就是说 LoRA 不会挂到：

- 视觉主干
- 视觉塔
- regression head

另外，Qwen 路线里 `modules_to_save` 指定为：

- `lm_head`
- `visual.merger.mlp`

所以 LoRA 打开后，实际训练部分可以概括为：

- 语言模型注意力里的 LoRA adapter
- `lm_head`
- `visual.merger.mlp`
- `regression_head`

#### 当前本地 stage1 的实际可训练部分

结合我们当前本地 stage1 命令：

- `--stage1`
- `--use-lora`
- `--unfreeze-connector`
- `--image_loss_weight 0.0`

当前真正训练的部分可以近似理解为：

- 文本主干中的 LoRA adapter（注意力投影层）
- `lm_head`
- `visual.merger.mlp`
- `regression_head`

而被冻结的主要部分是：

- 文本 embedding
- 原始视觉编码主干 `visual`
- `vision_tower`

所以当前 stage1 并不是“全参训练整个 Qwen2.5-VL”，而是：

- **文本侧 LoRA + connector + regression head 的混合训练**

### 为什么说 stage1 还是偏文本任务

虽然 `regression_head` 也是可训练的，但 stage1 下：

- `ignore_image = True`
- 当前本地训练又设置了 `image_loss_weight = 0.0`

因此在训练目标上，stage1 主要仍然是：

- 文本输出
- reasoning 对齐
- 最终答案格式对齐

而不是把中间图预测训到位。

### 使用的监督信号

`stage1` 主要依赖：

- `label_text`
- `label_text` 里的最终答案 token

我们当前本地 `stage1` 训练时使用的是：

- `--image_loss_weight 0.0`
- `--stage1`
- `--use-lora`

因此在这条链路里，`stage1` 的作用可以理解为：

- 先把模型适配到 `3dproject` 的推理/回答风格
- 让模型学会输出任务需要的最终答案格式
- 为后续 `stage2` 做准备

## 3. Stage2 训练什么

### 目标

`stage2` 主要训练的是：

- 中间视觉 latent 的预测能力
- 中间 sketch / 中间图的生成能力

具体来说，就是训练模型：

- 根据前面的图文上下文
- 在应该输出中间图的位置
- 预测出对应的视觉表示

### 代码层面的实际行为

在 `train.py` 里：

- 如果输出目录包含 `stage2`，则设置 `ignore_image = False`
- `stage2` 的常见配置还包括：
  - `--freeze-backbone`
  - `--text_loss_weight 0.0`

这意味着 `stage2` 的训练重点不是文本，而是：

- 冻结大部分 backbone
- 专门优化视觉特征预测链路

### 冻结什么、训练什么（以当前代码逻辑为准）

`stage2` 的典型脚本 `scripts/train_stage2.sh` 里给出的关键参数是：

- `--freeze-backbone`
- `--text_loss_weight 0.0`

结合 `train.py` 的实际逻辑，这意味着：

#### 第一步：先按普通规则做一遍冻结/解冻

也就是仍然会先经历上面 stage1 里那套步骤：

- 全部参数先设为可训练
- 冻结文本 embedding
- 冻结 `visual`
- 冻结 `vision_tower`
- 如果有 `--unfreeze-connector` 则解冻 `visual.merger.mlp`
- 解冻 `regression_head`

#### 第二步：`--freeze-backbone` 再把整个 backbone 全冻结

一旦传入：

- `--freeze-backbone`

代码会执行：

- `for param in model.parameters(): param.requires_grad = False`

也就是说，这一步会把前面所有参数统一清零成不可训练。

#### 第三步：重新解冻 regression_head

然后代码又会执行：

- `for param in model.regression_head.parameters(): param.requires_grad = True`

所以在不考虑 LoRA 的情况下，`stage2` 最核心、最稳定会训练的部分就是：

- `regression_head`

#### 如果 stage2 同时启用了 LoRA

如果 `stage2` 也配合 `--use-lora`，那么还会额外挂上：

- 文本注意力层的 LoRA adapter

以及 `modules_to_save`：

- `lm_head`
- `visual.merger.mlp`

但从“设计目标”上看，`stage2` 的重点并不在文本，而是在：

- `regression_head`
- 中间视觉特征预测

### 用一句最直接的话概括

如果只看“谁是 stage2 的主训练对象”，答案是：

- **主训练对象是 `regression_head`**

而不是整套语言 backbone。

### GT 从哪里来

`stage2` 的 GT 不是外部单独保存的 latent 文件。

GT 来自数据中的真实中间过程图，也就是：

- `label_img`

对于每个样本，`data/dataset.py` 会读取：

- `input_img`
- `label_img`

然后在 `model/uni_qwen.py` 里：

- `label_pixel_values` 是由 `label_img` 加载得到的真实中间图张量
- `target_vit = self.get_vit_features(...)`
- `target_features = self.visual(...)`

也就是说：

- GT 图像来源：`label_img` 里的真实中间图
- GT 特征来源：这些真实中间图经过模型自己的视觉编码器后得到的特征

### 优化的 loss

在 `uni_qwen.py` 中，模型会把预测出的视觉特征与 GT 视觉特征进行对齐，支持的 loss 包括：

- `mse`
- `l1`
- `cosine`

目前本地使用的是：

- `loss_type: l1`

所以 `stage2` 的训练本质是：

- 让模型学会“根据前文图文上下文，预测下一张中间过程图的视觉表示”

### 更细一点：GT 对齐发生在什么位置

在 `uni_qwen.py` 的前向里，逻辑是这样的：

1. 先找出标签序列中哪些 token 属于图像 token
2. 用这些位置前一时刻的 hidden states 作为条件输入
3. 把这些 hidden states 送进 `perceiver_forward -> regression_head`
4. 得到模型预测的中间视觉特征
5. 再把 `label_img` 对应真实中间图提特征，作为 target
6. 计算预测特征和 target 特征之间的 image loss

所以 stage2 学的是：

- 在“该出图”的位置，预测出与 GT 中间图一致的视觉表示

## 4. 这件事如何对应到 3DProject

在我们转换后的 `3dproject` 数据里：

- `input_img`：题目的原始 composite 图
- `label_text`：交错的 reasoning 文本
- `label_img`：中间过程图，例如 `target_progress_*.png`

因此这个数据集天然适合 `Latent-Sketchpad`：

- 有输入图
- 有推理文本
- 有中间过程图
- 有最终答案

对应关系就是：

- `stage1`：学 `3dproject` 的文本推理/答案输出风格
- `stage2`：用 `3dproject` 的中间过程图做视觉监督

## 5. 这次本地适配完成了什么工作

### 模型和权重

已经准备好：

- 基座模型：`/workspace/home/oujingfeng/project/models/Qwen2.5-VL-7B-Instruct`
- Sketch Decoder：`/workspace/home/oujingfeng/project/models/Latent-Sketchpad.Sketch_Decoder/sketch_decoder_qwen25_vl.ckpt`

### 数据转换

已经把原始 `3dproject` 交错图文数据转换成 `Latent-Sketchpad` 所需格式：

- `input_text`
- `input_img`
- `label_text`
- `label_img`

转换脚本：

- `unimrg/scripts/convert_3dproject_to_latent_sketchpad.py`

转换结果目录：

- `/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/latent_sketchpad_3dproject_currentstyle_3329/`

其中包括：

- `train.json`：2663 条
- `val.json`：333 条
- `test.json`：333 条

### 代码修复和本地适配

已经修复/适配的点包括：

- `train.py`、`inference.py`、`evaluate.py` 改为支持本地环境变量路径，不再依赖 `/path/to/...`
- `train.py` 现在支持 `WANDB_DISABLED=true` 时不强行注册 wandb callback
- `train.py` 现在支持独立 `--eval_data_path`
- 修复了 LoRA 刚挂上就 `merge_and_unload()` 的问题
- 增加了本地训练和评测的封装脚本

### 环境问题

已创建并验证可用的 conda 环境：

- `sketchpad-clean`

已解决的实际问题包括：

- `flash-attn`
- `torchrun`
- `CUDA_HOME`
- DeepSpeed 启动
- 多卡训练不能走错误的 `DataParallel`
- 本地模型/数据路径注入

### 训练与推理验证

已经验证：

- 官方 MazePlanning 公共推理链路可在本地跑通
- `3dproject` 单样本前向可跑通
- `3dproject` 的 `stage1` 两卡训练可跑通

### Stage1 已有评测结果

之前一个 `stage1` 的 `checkpoint-200` 测试结果约为：

- `125 / 333 = 37.54%`

这个结果应理解为：

- `stage1` 已经学会了格式和答案输出
- 但这还不是完整 `Latent-Sketchpad` 路线的最终效果

## 6. 当前正在进行的训练方向

当前新的训练方向是：

- 重新跑 `stage1`
- 使用独立 `val.json`，不再从训练集随机切验证集
- 有效 batch size 为 `16`
- 开启在线 `wandb`
- 训练完成后先评测 final checkpoint
- 如果 final 不好，再回测 earlier checkpoints

## 7. 最核心的结论

这次工作的最重要总结是：

- `stage1` 主要负责任务/文本对齐
- `stage2` 才是真正训练中间视觉状态预测
- `stage2` 的 GT 来自数据里的真实中间过程图 `label_img`

因此，如果最终要判断 `Latent-Sketchpad` 是否真的能通过“图文交替推理”提升能力，不能只看 `stage1`，真正有意义的比较应该是：

- 原始 VLM baseline
- `stage1`
- `stage2`

三者之间的效果差异。

## 8. Decoder / Aligner 近期排查结论（2026-04-12）

### 8.1 官方 decoder 实际训练的是什么

官方 `decoder` 训练的核心并不是重新训练整套图像 decoder，而是训练：

- `vision feature -> VAE latent` 的 `aligner`

真正把 latent 解码成图的是冻结的：

- `stabilityai/sdxl-vae`

从 `decoder/train.py` 可以确认：

- `vae_ref` 冻结
- `vae_encoder` 冻结
- 真正优化的是 `ClipToLatentAligner`

### 8.2 官方不同 backbone 的 decoder 训练是否同一套架构

是同一套 `ClipToLatentAligner` 架构，不同点主要在：

- 输入给 aligner 的视觉特征来源不同
- `input_dim` 不同
- 少量训练超参（如 batch size / epochs / eval step）不同

官方已给出的三条路线如下：

- OpenCLIP：`input_dim = 1024`
- Gemma3：`input_dim = 1152`
- Qwen2.5-VL：`input_dim = 1280`

共同点：

- `layer = 12`
- `image_size = 224`
- `learning_rate = 1e-4`
- `dense_align = true`

### 8.3 Qwen2.5 的关键接口问题已经确认

`Qwen2.5-VL` 视觉配置同时存在：

- `vision_config.hidden_size = 1280`
- `vision_config.out_hidden_size = 3584`

官方 `decoder` config 指定：

- `input_dim = 1280`

这说明官方 `Qwen2.5` decoder 预期吃的是：

- **pre-merger 的 1280 维视觉 token**

之前本地效果差的根因，不是单纯环境问题，而是 `Qwen2.5` 给 decoder 的特征接口不对。修正后，无论在旧环境还是新建 `vit-vae` 环境，`Qwen2.5` 都可以正常做重建。

当前修正后的 `Qwen2.5` 视觉特征 shape 为：

- `(1, 1024, 1280)`

### 8.4 Qwen3.5 当前 decoder 输入特征

当前 `Qwen3.5` decoder wrapper 实际送入 aligner 的特征为：

- `model.model.visual(...).last_hidden_state`

对应当前实测 shape：

- `(1, 784, 1024)`

同时 `Qwen3.5` 配置中也存在：

- `vision_config.hidden_size = 1024`
- `vision_config.out_hidden_size = 2560`

因此当前 `Qwen3.5` 并不是像早期 `Qwen2.5` 那样明显喂错 merger 后特征，但仍然存在一个核心未决问题：

- 当前取到的 `1024` 维 `last_hidden_state` 是否就是 decoder 最适合吃的 feature tap

### 8.5 Stage1 是否训练视觉塔

当前本地 `Qwen3.5 stage1` 代码里：

- `model.model.visual` 冻结
- `model.model.visual.merger` 在 `stage1` 下也冻结

因此：

- `Qwen3.5 stage1` 默认不训练视觉塔主体

官方 `Qwen2.5` 路线的 `train.py` 也同样默认冻结视觉塔主体，因此两者在这一点上是一致的。

### 8.6 当前 `non_background_weight` 是否实际生效

当前本地和官方共享的 `decoder/train.py` 中：

- 旧的按 `foreground mask` 加权 MSE 已被注释掉
- 实际执行的是 `focal_loss(decoded, imgs_vae, foreground_masks)`

因此：

- `non_background_weight` 目前基本没有真正参与 image reconstruction loss

这不仅适用于本地 `Qwen3.5`，也同样适用于官方 `Qwen2.5 / Gemma / OpenCLIP` 当前共享训练代码。

### 8.7 为什么 `Qwen2.5` 重建好，但训练时 `latent_loss` 仍然很大

已经做过一个验证：

- 从官方 `Qwen2.5` decoder checkpoint 继续在 `QuickDraw` 上短训几步

结果显示：

- `image_loss` 和 `embed_loss` 会明显变小
- 但 `latent_loss` 仍可能很大

这说明：

- `latent_loss` 的绝对数值本身并不能直接等价于“重建质量差”
- 更值得关注的是：
  - `image_loss`
  - `embed_loss`
  - 实际重建图质量

当前对比结果是：

- 官方 `Qwen2.5` checkpoint continued-training：`image_loss` 很小，`embed_loss` 很小
- 当前已训练好的 `Qwen3.5 aligner`：`image_loss` 和 `embed_loss` 明显偏大

这支持一个更强的判断：

- `Qwen3.5` 当前问题不只是 `latent_loss` 数值怪，而是 aligner 对齐质量本身仍明显不如官方 `Qwen2.5` checkpoint

### 8.8 新建官方 decoder 独立环境

已按官方 `decoder` 环境要求创建独立 conda 环境：

- `vit-vae`

并补装：

- `torchscale`
- `xformers==0.0.28`
- `lightning`
- `torchdata`
- `transformers==4.54.0`
- `timm==1.0.13`

注意：

- 该环境本身存在 `torchscale` 与 `timm==1.0.13` 的版本冲突提示
- 但至少已经足以验证官方 `Qwen2.5` decoder 的重建链路
- `Qwen3.5` 无法直接在 `vit-vae` 环境中运行，因为 `transformers==4.54.0` 不支持 `Qwen3.5`

### 8.9 新一轮对齐官方 recipe 的 Qwen3.5 aligner 训练

当前已将 `decoder/configs/dense-config-qwen35.json` 改为：

- 尽量对齐官方 `Qwen2.5` decoder recipe
- 但仍保留 `3dproject` 图片混入

当前关键配置：

- `learning_rate = 1e-4`
- `epochs = 10`
- `batch_size = 16`
- `gray_image = false`
- `cate_num = -1`
- `number_per_class = -1`
- `train_split = 0.999`
- `val_split = 0.0003`
- `eval_every_n_steps = 500`
- `save_every_n_train_steps = 500`
- `disable_eval = false`
- `accumulate_grad_batches = 2`
- `input_dim = 1024`
- 同时保留 `3dproject train/val` 混入

本轮新训练输出目录：

- `/workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/qwen35_aligner_full_run2`

tmux session：

- `qwen35_aligner_full_run2`

实际启动命令如下：

```bash
tmux new-session -d -s qwen35_aligner_full_run2 'source "/workspace/home/miniconda3/etc/profile.d/conda.sh" && conda activate sketchpad-qwen35-cu128 && export CONFIG_FILE_PATH="/workspace/home/oujingfeng/project/Latent-Sketchpad/decoder/configs/dense-config-qwen35.json" && export MOUNT_DIR="/workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/qwen35_aligner_full_run2" && export DATA_DIR="/workspace/home/oujingfeng/project/Latent-Sketchpad/decoder" && python "/workspace/home/oujingfeng/project/Latent-Sketchpad/decoder/train.py" --gpus 2 > "/workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/qwen35_aligner_full_run2/train.log" 2>&1'
```

### 8.10 当前最重要的判断

截至目前，最可信的判断是：

- `Qwen2.5` 之前效果差，主因是 decoder 输入接口不对
- `Qwen3.5` 当前问题不像单纯环境问题，更像：
  - 当前 feature tap 还不一定是最佳的
  - 自训 aligner 的对齐质量仍明显不如官方 `Qwen2.5` checkpoint

因此，如果新一轮更贴官方 recipe 的 `Qwen3.5` 训练之后效果仍然差，下一步最该排查的就不再是表层训练超参，而是：

- `Qwen3.5` 的候选视觉特征 tap
- 哪一种 feature 更适合送入 decoder / aligner

## 9. spatialviz-ours 融合数据集（2026-04-12）

### 9.1 当前 merged 数据集位置

已将：

- 现有 `3dproject`
- 新上传的 `spatialviz-ours`

统一转换并合并为新的 `Latent-Sketchpad` 数据集：

- `/workspace/home/oujingfeng/project/unimrg/datasets/spatialviz/latent_sketchpad_spatialviz_merged_qwen35`

其中包含：

- `train.json`
- `val.json`
- `test.json`
- `summary.json`

### 9.2 已覆盖任务类别

当前 merged 数据集已覆盖：

- `3dproject`
- `3drotation`
- `cubecounting`
- `cubereconstruction`
- `cubeunfolding`
- `2drotation`
- `arrowmoving`
- `blockmoving`
- `paperfolding`

即：

- `spatialviz-ours` 的 8 个任务
- 加上已有的 `3dproject`

### 9.3 各任务导入数量

当前 merged 数据集统计：

- `train = 11916`
- `val = 1490`
- `test = 1490`

其中 `spatialviz-ours_total = 11567`

各任务原始导入数：

- `3drotation = 976`
- `cubecounting = 2000`
- `cubereconstruction = 1000`
- `cubeunfolding = 1000`
- `2drotation = 2549`
- `arrowmoving = 3000`
- `blockmoving = 574`
- `paperfolding = 468`

### 9.4 prompt / token 处理已统一

当前 `spatialviz-ours` 转换脚本对所有任务统一生成 `input_text` 前缀：

```text
You should first provide a reasoning process, then provide a single option (A, B, C or D) as the final answer. The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags respectively.
Question: ...
<image>
```

训练 / 评测 / 推理三条线都统一使用：

- `data/dataset_qwen35.py`

其中会把 `<image>` 统一替换为：

- `<|vision_start|><|image_pad|><|vision_end|>`

因此当前：

- 训练 prompt
- 评测 prompt
- 推理 prompt

在 `<image>` token 化规则上是一致的。

### 9.5 所有任务都包含中间图像段 supervision

已统计 merged `train.json` 中各任务 `label_text` 的 `<image>` 数量分布，确认：

- 所有任务 `zero_image = 0`

这意味着：

- 所有任务都在给 `stage1` 提供图文交替 supervision
- `stage1` 会学习中间推理图像段的 `vision_start / image_pad / vision_end` 结构 token

部分任务的中间图像段数量较多，例如：

- `blockmoving`: `19~28` 张中间图
- `arrowmoving`: 最多到 `13` 张
- `3drotation`: `9~12` 张

### 9.6 图片路径已统一到单一根目录

当前 merged 数据集中的所有 `input_img / label_img` 均已统一为相对于：

- `/workspace/home/oujingfeng/project`

的路径。

因此训练 / 评测 / 推理时统一使用：

- `--image_dir /workspace/home/oujingfeng/project`

即可同时读取：

- `datasets/spatialviz-ours/...`
- `unimrg/datasets/spatialviz/...`

### 9.7 smoke test 已覆盖所有任务类别

当前 smoke test 不是随机抽样，而是：

- 对 `train / val / test`
- 按 `task_name` 分组
- 每个任务类别至少抽 1 条
- 检查：
  - `input_text`
  - `label_text`
  - `input_img`
  - `label_img`
  - 图片路径是否真实存在

已确认所有任务类别均可在单一 `image_root=/workspace/home/oujingfeng/project` 下正常读入。

## 10. merged 数据集长度分布（2026-04-12）

已对 merged `train.json` 做 token 长度统计，统计文件：

- `/workspace/home/oujingfeng/project/Latent-Sketchpad/outputs/spatialviz_merged_train_token_lengths.json`

整体分布：

- `num_samples = 11916`
- `min = 255`
- `max = 2513`
- `p50 = 625`
- `p90 = 1269`
- `p95 = 1857`
- `p99 = 2307`

长度较长的任务主要包括：

- `blockmoving` (`max = 2513`)
- `arrowmoving` (`max = 2349`)
- `3drotation` (`max = 1820`)

当前训练代码：

- 没有先扫描全数据得到一个全局 `max token`
- 也没有显式 `max_length/truncation`
- 只是在 collator 阶段按 batch 内最长样本做 padding

## 11. merged stage1 / stage2 脚本（2026-04-12）

### 11.1 新增脚本

已新增以下 merged 数据集专用脚本：

- `scripts/train_spatialviz_merged_stage1_qwen35_full.sh`
- `scripts/train_spatialviz_merged_stage2_qwen35_full.sh`
- `scripts/eval_spatialviz_merged_stage1_qwen35.sh`
- `scripts/eval_spatialviz_merged_stage2_qwen35.sh`
- `scripts/infer_spatialviz_merged_stage2_qwen35.sh`
- `scripts/summarize_eval_by_task.py`

其中：

- 训练脚本默认使用 merged `train.json`
- 评测脚本默认使用 merged `test.json`
- 评测后会自动输出按 `task_name` 汇总的 `task_summary.json`

### 11.2 prompt / 数据入口是否统一

当前 merged 线路中：

- `stage1` 训练
- `stage2` 训练
- `stage1` 评测
- `stage2` 评测
- `stage2` 推理

都已统一到：

- merged 数据集 JSON
- 单一 `image_dir=/workspace/home/oujingfeng/project`
- 同一套 `<image> -> Qwen 视觉 special token` 替换规则

### 11.3 当前 merged stage1 训练参数

当前 merged `stage1` 因长序列样本较多，显存压力主要来自：

- 长样本的 CE loss
- `transformers` 内部 `logits.float()`

已多次调整后，当前运行中的配置为：

- `learning_rate = 1e-4`
- `per_device_train_batch_size = 1`
- `gradient_accumulation_steps = 8`
- `num_train_epochs = 3`
- `image_loss_weight = 0.0`
- `stage1`
- `disable_eval`

并额外设置：

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

当前训练会学习：

- 中间推理图像段的 `vision_start / image_pad / vision_end` 结构 token

但不会在 `stage1` 中训练真正的图像 latent 回归内容。
