# book_cpt

书籍原始知识表达数据生成流水线。`book_cpt` 内部独立实现 PDF 渲染、MinerU 解析、版面标准化、裁剪、LLM/VLM provider pool、并发、校验、去重和导出逻辑，支持 SFT 多模态/文本任务和 LLaMA-Factory `stage: pt` 纯文本领域知识语料任务。

## 支持任务

- `paragraph_summary`
- `title_body_alignment`
- `page_to_structured_description`
- `chart_table_to_text`
- `page_content_restatement`
- `cross_page_synthesis`
- `chapter_key_conclusions`
- `evidence_to_conclusion_chain`
- `domain_knowledge_corpus`

导出格式按 `task_template.md` 固定：

- Alpaca：`paragraph_summary`、`title_body_alignment`、`chapter_key_conclusions`
- ShareGPT messages：`page_to_structured_description`、`chart_table_to_text`、`page_content_restatement`、`cross_page_synthesis`、`evidence_to_conclusion_chain`
- PT text JSONL：`domain_knowledge_corpus`

生成阶段要求大模型根据当前输入证据生成 `instruction`，用于增加问题表达多样性；代码不再使用固定问题模板兜底。生成的问题不得偏离任务主题，未生成 `instruction` 的样本会被丢弃。最终导出只保留模板字段，中间样本仍保留证据和 metadata 便于回溯。

`chart_table_to_text` 只使用从书籍中提取的图表图片，即每本书输出目录下的 `images/extracted/`，例如 `book_cpt\outputs\飞行员航空知识手册\images\extracted`。该任务不会使用整页页面图，也不会使用临时版面块裁剪图；如果某个图表块没有 `extracted_image_path`，则跳过该任务。

`chart_table_to_text` 的回答字段中不得出现图号、章节号、本文、文中、书中、本书、该书、本章、本节、章节等来源位置词，也不要写 `图1`、`表2`、`figure 3` 等编号表达；违反该规则的样本会在本地校验阶段被过滤。

质量校验会先执行本地硬规则，再按导出格式增加维度评审：Alpaca 问答样本检查“文本格式质量”和“问答相关性”；带图片的 ShareGPT 样本检查“文本格式质量”“问答相关性”“视觉依赖度”和“图像-问题对应性”。其中图像-问题对应性会把当前样本图片传给 VLM 复核，并将结果写入中间样本 `metadata.quality_checks`；默认 `quality_review_fail_closed=true`，复核失败的样本不会进入 validated。

`domain_knowledge_corpus` 面向 LLaMA-Factory `stage: pt` 继续预训练，基于 MinerU/OCR 解析后的连续页面文本进行清洗、去噪和语义完整切块。最终导出文件为 `exports/pt/domain_knowledge_corpus.jsonl`，每行只包含 `text` 字段，不包含 `instruction/input/output` 或对话字段。

默认生成全部任务。可通过 `--task` 或 `--tasks` 只生成指定任务，并同步收窄路由、provider 白名单、中间样本和导出文件。

```powershell
python book_cpt\app\cli.py --input-dir book_cpt\data --output-root book_cpt\outputs
python book_cpt\app\cli.py --input-dir book_cpt\data --output-root book_cpt\outputs --tasks paragraph_summary,page_content_restatement
python book_cpt\app\cli.py --input-dir book_cpt\data --output-root book_cpt\outputs --task evidence_to_conclusion_chain
python book_cpt\app\cli.py --input-dir book_cpt\data --output-root book_cpt\outputs --task domain_knowledge_corpus
```

## 断点续跑和补缺任务

已经完成 MinerU 解析的批量任务，建议续跑时带上 `--reuse-mineru`：

```powershell
python app\cli.py --input-dir data --output-root outputs --reuse-mineru
```

续跑时会按任务类型判断完成状态：9 个任务导出都已完成的书会直接跳过；只缺部分任务的书会复用已有 MinerU、normalized、deduped 和 export 缓存，只把缺失任务送入大模型生成。任务完成状态以 `samples/cache_state.json`、`samples/deduped/{task_type}.jsonl` 和对应 `exports/.../{task_type}.jsonl` 为准；如果某个待生成任务有 jobs 但没有产出有效样本，它不会被标记为完成，下次还会继续补跑。

需要强制重跑时再使用 `--force-rebuild`，它会关闭 MinerU 之后所有缓存复用。

### 原 PDF 已删除时从输出缓存运行

如果原 PDF 已不存在，但每本书的输出缓存或 MinerU 中间文件仍然完整，可以只扫描缓存目录：

```powershell
python app\cli.py --output-root outputs_test --task domain_knowledge_corpus --cache-only
```

如果缓存目录就是 `book_test/<书名>/mineru` 这种结构，把 `--output-root` 指到 `book_test`：

```powershell
python app\cli.py --output-root book_test --task domain_knowledge_corpus --cache-only
```

`--cache-only` 不需要 `--input-dir` 或 `--book`，并会自动启用 MinerU/normalized 复用、关闭 PDF 页面渲染和块裁剪。每本书目录支持两种形式。优先使用完整输出缓存：

```text
manifest.jsonl
mineru/raw/{book_id}.json
mineru/parsed/{book_id}.json
mineru/image_map.json
```

如果没有 `manifest.jsonl`，也可以只保留 MinerU 缓存：

```text
mineru/raw/{book_id}.json
mineru/parsed/{book_id}.json
mineru/image_map.json
```

此时程序会从 raw/parsed 同名 JSON 推断 `book_id`，用书目录名作为 `book_name`，并在运行时重新写入 `manifest.jsonl`。

缓存文件缺失或内容无效时会返回明确错误，且不会回退请求 MinerU。该参数不能与 `--input-dir`、`--book` 或 `--force-rebuild` 同时使用。

只处理单本书时使用 `--book` 指定 PDF 路径：

```powershell
python book_cpt\app\cli.py --book "book_cpt\data\飞行员航空知识手册.pdf" --output-root book_cpt\outputs
```

也可以同时指定输出目录和任务类型：

```powershell
python book_cpt\app\cli.py --book "book_cpt\data\飞机推进 [美法罗基著] 2011年版.pdf" --output-root book_cpt\outputs --tasks paragraph_summary,chapter_key_conclusions
```

如果希望手动指定书名，可使用 `书名=PDF路径` 格式：

```powershell
python app\cli.py --book "E:\航天\book_cpt\data\飞行员航空知识手册.pdf" --output-root book_cpt\outputs
```

调试结构和导出路径时可跳过大模型生成：

```powershell
python app\cli.py --book "E:\航天\book_cpt\data\飞行员航空知识手册.pdf" --output-root outputs --skip-vlm
```

## 并发参数

流水线支持按阶段设置并发数：

- `--book-workers`：书籍级并发数，同时处理多少本 PDF。多本书批处理时有效。
- `--page-workers`：单本书内部的页面渲染并发数，用于 PDF 页面转图片。
- `--crop-workers`：单本书内部的版面块裁剪并发数，用于生成局部块图片。
- `--max-workers`：单本书内部的大模型样本生成并发数，用于并行调用 LLM/VLM。
- `--mineru-workers`：MinerU 书籍解析并发数。远端 MinerU 服务不稳定时建议设为 `1`，这样仍可保持书籍级流水线并发，但 OCR/解析阶段会排队进入 MinerU。
- `--mineru-retry-count`：MinerU 失败或结果不完整时的重试次数。
- `--mineru-min-page-coverage`：MinerU 解析结果的最小页覆盖率，低于该比例会判定为不完整并重试。
- `--force-rebuild`：不复用 MinerU 之后的 normalized、samples、exports 缓存，适合清理旧的不完整结果。
- `--no-reuse-normalized` / `--no-reuse-samples` / `--no-reuse-exports`：分别关闭对应阶段缓存复用。
- `--recursive` / `--no-recursive`：是否递归扫描 `--input-dir` 下的子目录，默认只扫一级（可在 `config.py` 的 `runtime.recursive` 里改默认值）。


示例：

```powershell
python app\cli.py --input-dir book_cpt\data --output-root book_cpt\outputs --book-workers 1 --page-workers 4 --crop-workers 4 --max-workers 4 --reuse-mineru
```

单本书处理时通常不用设置 `--book-workers`，重点调 `--page-workers`、`--crop-workers` 和 `--max-workers`：

```powershell
python app\cli.py --book "E:\航天\book_cpt\data\飞行员航空知识手册.pdf" --output-root outputs --book-workers 1 --page-workers 4 --crop-workers 4 --max-workers 6 --reuse-mineru
```

参数含义可以简单理解为：

```text
总并发压力 ≈ book-workers × 每本书内部 max-workers
```

例如 `--book-workers 2 --max-workers 4` 时，理论上最多会有约 8 个大模型生成请求并行。`page-workers` 和 `crop-workers` 主要消耗 CPU、磁盘和图片处理资源；`max-workers` 主要消耗大模型服务并发额度。机器资源或模型服务不稳定时，建议先从 `--book-workers 1 --page-workers 2 --crop-workers 2 --max-workers 2` 开始。

### 随仓库带的默认值

`config.py` 里的默认并发与模型池是配套算过的，直接跑不用传参：

```text
book_workers 16 × max_workers 16 = 256 路生成并发
16 个 VLM 实例 × 声明 max_concurrency 16 = 256 个槽位
摊薄后每个进程在每个实例上占 1 槽 —— 不多不少，刚好填满
```

- `page_workers` 压到 2：页面渲染是 CPU/磁盘密集，会跟 16 个书籍进程叠乘（16×2 = 32 个渲染进程）。
- `crop_workers` 仍是 4，叠乘后是 64 个裁剪进程。book_cpt 的 `crop_blocks` 是开着的
  （journal_cpt 那边整个关掉了，所以它不在意这个值），机器扛不住就调小。
- MinerU 槽位是文件锁，全局 2 实例 × 16 = 32，**不随 `book_workers` 叠乘**。

跑不满或者压垮服务时，优先动 `--book-workers` 和 `--max-workers`，两者的乘积才是打到模型服务上的真实并发。

如果问题集中在 MinerU 服务，推荐保留多本书调度，但单独限制 MinerU：

```powershell
python app\cli.py --input-dir book_cpt\data --output-root book_cpt\outputs --book-workers 3 --mineru-workers 1 --mineru-retry-count 3 --mineru-min-page-coverage 0.8 --page-workers 2 --crop-workers 2 --max-workers 4 --force-rebuild
```

### 多实例调度

`config.py` 的 `mineru.providers` 和 `vlm_pool.providers` 都可以配多个实例，调度是抢占式的：谁先空出来谁接下一个任务。
当前随仓库带的是 16 个 VLM 实例和 2 个 MinerU 实例（`vlm-http-client` 后端，`server_url` 指向推理服务）。

- MinerU 槽位是 `output-root/.runtime/mineru_slots/<provider>/slot_N.lock` 文件锁，跨进程（乃至共享盘上跨主机）都成立；
  实例连不上时按 `mineru.cooldown_seconds` 进冷却，期间不再往它派活，解析结果不完整则不算实例的锅、不冷却。
  只配 `mineru.url` 不配 `providers` 时行为与单实例时完全一致。
  `vlm-http-client` 后端必须带 `server_url`，否则服务端会去读 `MINERU_VL_SERVER` 环境变量，读不到就以 409 返回。
- VLM 侧的冷却状态写在 `output-root/.runtime/vlm_cooldown/` 下，多个书籍进程共享；
  一个实例挂了只需被踩一次，不用每个进程各踩一遍。provider 的 `name` 是冷却文件名，重名会在启动时直接报错。
- `--book-workers > 1` 时每个子进程各建一份 VlmPool，所以 provider 的 `max_concurrency` 和
  `min_interval_seconds` 会按进程数自动摊薄，实际压到服务上的并发就是配置里写的那个数。

### 去水印（默认开启）

会把整册重复出现（覆盖率超过 `watermark.min_page_coverage`，默认 0.6）的水印 Form XObject 调用剔掉，
另存一份干净 PDF 再送 MinerU，报告写在 `preprocessed/watermark_cleaning.json`。命中的判据是三选一：
可选内容组（`/OC`）、`watermark.form_names` 里点名的对象名、`watermark.image_sizes` 里点名的图片尺寸。

- **只有「这一轮真的重新清理过」才让缓存失效**（输入 PDF 变了，基于旧 PDF 的缓存不能再用）。
  第二轮起如果源 PDF 和判定参数都没变，直接沿用上一轮的产物，报告里 `reused: true`，
  缓存照常复用 —— 否则带水印的书每轮都要从头重跑 MinerU 和全部样本。
  快速路径靠报告里的 `source_hash` + `config_signature` 判断；对不上时会重扫一遍，
  再拿 `content_hash` 兜底比对，产物逐字节相同仍算复用。
- 没找到候选就原样透传，不会多写一份 PDF。
- 代价是每本书都要用 pypdf 完整解析一遍各页的内容流；大部头书这一步不便宜。
  想关掉：`config.py` 里把 `watermark.enabled` 设成 `false`。
- 关掉后损坏 PDF 的检测仍然生效（照样走跳过而不是失败）。

## 运行观测

- 进度条：书籍级进度写 stderr，形如 `书籍 [████░░░░] 12/40  30.0% | ✓11 ✗1 | 03:12<07:24 | 飞行员航空知识手册`。
  重定向到文件或 `nohup` 时（非 TTY）自动退化成每完成一本打一行的 `[进度] ...`，不会往日志里塞回车符。
  用 `--no-progress` 关掉。
- 生成阶段进度：单本书内部每完成若干个生成任务打一条带 ETA 的日志
  （`generate progress book_id=... jobs=37/120 samples=41 failed=0 用时=6.4分 预计剩余=14.3分`），
  节奏由 `runtime.generation_progress_every` / `generation_progress_seconds` 控制。
- 日志级别：`--log-level DEBUG|INFO|WARNING|ERROR`（也可用环境变量 `BOOK_CPT_LOG_LEVEL`）。
  `metric`/`checkpoint` 明细降到了 DEBUG，`--quiet` 等价于 `--log-level WARNING`，只留进度条和错误。
- 收尾摘要：跑完在 stderr 打一段 `完成 38/40 本，样本 5123 条`，并分别列出跳过和失败的书。
- 损坏 PDF：pypdf 打不开的文件（截断、加密、结构损坏）算「跳过」不算「失败」，
  记进 `<output-root>/skipped_books.jsonl`，不会中断整批。
- 结构有瑕疵但能读的 PDF（`too many kids in page tree` 之类）：MuPDF 的报错是 C 库
  直写 stderr 的，Python logging 压不住，所以默认关掉逐条输出，改由 `render_pages`
  汇总成一条 `pdf structure complaints book_id=... count=N`。想看原始报错就把
  `config.py` 的 `render.silence_mupdf_errors` 设成 `false`。
- Ctrl-C 会取消排队中的任务并立刻退出。已完成的书都已落盘，重跑走缓存接上。
- 整批生成全失败时抛 `GenerationBatchError`，消息里带 book_id、任务分布和去重后的错误原文，
  不再静默返回 0 条样本。
- HTTP 报错不再只说「service unavailable」：401/404/400 会带上状态码、返回体和常见原因提示；
  连不上（超时、DNS）和「服务端拒绝」是两类不同的错误信息。

## 思维链与 JSON 容错

推理模型默认会输出思维链，服务端没开 reasoning parser 时 `<think>...</think>` 会留在 `content` 里，
把 JSON 解析整条带偏。现在：

- 每个 provider 默认补 `chat_template_kwargs.enable_thinking=false`（即使它显式写了空的 `chat_template_kwargs`）。
  确实需要保留思维链的 provider，在 `config.py` 里给它加 `"disable_thinking": false`。
- 解析前统一剥掉 `<think>`/`<reasoning>` 等成对、只剩闭合标签、被 max_tokens 截断只剩开标签这三种形态；
  流式和非流式都会跳过 `reasoning_content`。答案整个落在思维链里时直接报错，提示调 `enable_thinking` 或 `max_tokens`。
- JSON 修复：容忍字符串里的真实换行、补转义非法反斜杠（LaTeX 的 `\alpha`、`\%` 之类）、
  去掉尾随逗号、`expected_count=1` 时模型返回单个对象而不是数组也能收下。

依赖自检：

```powershell
python book_cpt\check_deps.py
```

## 输出

每本书按独立目录保存：

- `manifest.jsonl`
- `pages/page_index.jsonl`
- `images/pages/`
- `images/blocks/`
- `images/extracted/`
- `mineru/raw/` 和 `mineru/parsed/`
- `normalized/`
- `samples/raw/{task_type}.jsonl`
- `samples/validated/{task_type}.jsonl`
- `samples/deduped/{task_type}.jsonl`
- `exports/sharegpt/{task_type}.jsonl`
- `exports/alpaca/{task_type}.jsonl`
- `exports/pt/{task_type}.jsonl`
- `preprocessed/watermark_cleaning.json`（开启去水印时）

批次级文件写在 `--output-root` 根下：

- `skipped_books.jsonl`：损坏而跳过的 PDF 及原因
- `.runtime/mineru_slots/`、`.runtime/mineru_cooldown/`、`.runtime/vlm_cooldown/`：多实例调度用的槽位和冷却状态

中间样本保留 `input_payload`、`output_payload`、`evidence` 和 `metadata`，用于从训练样本回溯到原 PDF、页码、块 ID、图片和 MinerU/normalized 结果。
