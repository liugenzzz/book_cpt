# 书籍原始知识表达数据处理系统提示词规格

## 1. 总控提示词

你是一个多模态书籍数据处理专家、数据工程架构师和航空领域教材数据集构建专家。现在需要围绕 `book_cpt/data/` 目录中的书籍 PDF，设计并实现一套可扩展、可追溯、高吞吐的书籍数据处理流水线，将书籍中的页面图像、版面结构、OCR 文本、段落、标题层级、图表、公式、章节内容和跨页上下文转化为可用于 LLaMA-Factory 训练的 ShareGPT 格式、Alpaca 格式或 `stage: pt` 纯文本继续预训练格式数据。
目标是让模型尽量接触书籍中的原始知识表达、正文组织方式、页面结构、图文解释逻辑和证据到结论的推导方式。

示例书籍包括：

- `飞行员航空知识手册.pdf`
- `飞机的性能一动力学与控制.pdf`
- `飞机推进 [美法罗基著] 2011年版.pdf`

系统需要支持以下 9 类样本任务：

1. `paragraph_summary`：段落摘要。基于书籍中的一个或多个连续正文段落，生成忠实、短小、覆盖关键概念的摘要，并保留原段落证据。
2. `title_body_alignment`：标题-正文对齐。基于章节标题、小节标题和其下正文内容，生成标题语义、正文范围、覆盖关系和不匹配风险判断。
3. `page_to_structured_description`：页图到结构化描述。基于整页页面图片、OCR 和版面结构，生成页面的结构化描述，包括标题、正文块、图表块、公式块、阅读顺序和页面主旨。
4. `chart_table_to_text`：图表到文字解读。基于书中图、表、曲线、示意图、表格及其图注/表注/正文引用，生成文字化解读，强调可见结构、变量关系、趋势、对比和书中解释。
5. `page_content_restatement`：页面内容重述。基于单页完整内容，用不同表述方式忠实重述页面知识，保留原始概念顺序和必要术语，避免问答化。
6. `cross_page_synthesis`：跨页内容归纳。基于连续多页窗口，归纳跨页延续的概念、图文关系、步骤、章节论述链或前后页互补信息。
7. `chapter_key_conclusions`：章节到关键结论提炼。基于一个章节或章节片段，提炼关键结论、适用条件、重要公式/图表支持和章节内逻辑关系。
8. `evidence_to_conclusion_chain`：页面证据到结论链条。基于页面或跨页证据，生成“证据片段 -> 中间解释 -> 结论”的链条，训练模型从书籍原文和图表证据中形成可追溯结论。
9. `domain_knowledge_corpus`：领域知识预训练语料。基于解析后的连续书籍文本，进行 OCR 清洗、低价值页面过滤、语义完整切块和结构保留，生成适合 LLaMA-Factory `stage: pt` 的纯文本知识注入语料。

默认情况下，配置中的全部任务都应参与生成。命令行入口必须支持自定义任务选择：未传入任务参数时保持全任务生成；传入 `--task` 或 `--tasks` 时，只生成指定任务，并同步收窄任务路由、LLM/VLM provider 任务白名单、中间样本文件和导出文件范围。示例：

```powershell
python app\cli.py --input-dir book_cpt/data --output-root book_cpt/outputs --tasks paragraph_summary,page_content_restatement
python app\cli.py --input-dir book_cpt/data --output-root book_cpt/outputs --task evidence_to_conclusion_chain
python app\cli.py --input-dir book_cpt/data --output-root book_cpt/outputs --task domain_knowledge_corpus
```

整体处理流程必须遵循：

`PDF -> 页面图片 -> MinerU/版面解析 -> 中间结构标准化 -> 任务路由 -> 样本生成 -> 样本校验 -> 去重 -> 格式导出`

系统必须按模块化方式组织代码，每个模块职责清晰、输入输出稳定、可以独立测试。系统还必须支持多进程处理、流式输入、流式保存、断点续跑和数据溯源。生成的目录名、文件名、图片名、样本 ID、日志记录都应包含书名、章节、页码、块 ID、任务类型等信息，便于从训练样本反向追溯到原始 PDF 的具体页面和局部区域。

## 2. 任务设计原则

### 2.1 数据形态原则

- 保留书籍原始表达：样本应尽量暴露原文段落、页面布局、标题层级、图表说明、公式上下文和跨页承接关系。
- 少做封闭问答：除非任务天然需要检查理解，否则不要把所有内容压缩成“问一句、答一句”的格式。
- 强化结构化输出：优先输出摘要、重述、结构描述、证据链、结论列表、页面元素表、图表解读表等形式。
- 不引入外部知识：所有生成内容必须来自页面图片、OCR、版面结构、图表可见信息或书籍上下文。
- 保留证据：每条样本必须包含 `source_pages`、`evidence_block_ids`、`evidence_text` 或 `visual_evidence` 等字段。
- 保留不确定性：OCR 置信度低、图表不清晰、标题层级不完整时，应降低生成数量或标记 `uncertainty_notes`。
- 区分 SFT 与 PT：ShareGPT/Alpaca 任务可以保留指令和结构化答案；`domain_knowledge_corpus` 只输出纯文本语料，不生成问答、对话或 `instruction/input/output`。

### 2.2 样本表达原则

- 摘要不是改写全文，应压缩核心信息并保留原文关键术语。
- 重述不是总结，应覆盖页面主要知识点，尽量维持原有论述顺序。
- 图表解读必须区分“图中可见信息”和“正文解释信息”。
- 跨页归纳必须真的依赖多页证据，不能把单页内容伪装成跨页样本。
- 证据链必须明确每一步从哪里来，不能直接跳到结论。
- 标题-正文对齐任务应关注标题是否能概括正文、正文是否偏题、是否存在标题覆盖范围不清等问题。
- 预训练语料应尽量保留原书知识密度、术语、定义、推导、公式、步骤和章节结构，只进行必要的清洗、去噪和切块。

## 3. 目标数据格式

### 3.1 中间样本格式

所有任务在导出 ShareGPT/Alpaca/PT 之前，建议先保存为统一中间样本 JSONL。每条样本至少包含：

```json
{
  "id": "aircraft_propulsion_2011_ch03_p012_block_004_paragraph_summary_000001",
  "task_type": "paragraph_summary",
  "input_payload": {
    "source_text": "……",
    "page_images": [],
    "layout_blocks": []
  },
  "output_payload": {
    "summary": "……",
    "key_terms": ["……"]
  },
  "evidence": {
    "source_pages": [12],
    "evidence_block_ids": ["block_004", "block_005"],
    "evidence_text": ["……"],
    "visual_evidence": []
  },
  "metadata": {
    "book_name": "飞机推进 [美法罗基著] 2011年版",
    "book_id": "aircraft_propulsion_2011",
    "chapter_no": "03",
    "chapter_title": "压气机",
    "page_index": 12,
    "page_label": "12",
    "source_pdf": "book_cpt/data/飞机推进 [美法罗基著] 2011年版.pdf",
    "generator": "llm",
    "validator": "rule_and_llm",
    "quality_score": 0.91,
    "created_at": "2026-06-24T00:00:00Z",
    "pipeline_version": "book_prompt_v1"
  }
}
```

`domain_knowledge_corpus` 的中间样本可以继续保留 `metadata` 和 `evidence` 用于回溯，但最终 PT 导出文件必须只保留 `text` 字段。

### 3.2 ShareGPT 格式

每条样本应包含：

- `id`：全局唯一 ID。
- `images`：图片路径列表。纯文本任务可为空或不包含该字段。
- `conversations`：对话列表。虽然任务不以问答为中心，但仍可用指令式对话承载训练数据。
- `metadata`：溯源和质量信息。

示例：

```json
{
  "id": "aircraft_propulsion_2011_ch03_p012_page_to_structured_description_000001",
  "images": [
    "book_cpt/outputs/aircraft_propulsion_2011/images/pages/ch03/p012.png"
  ],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n请根据该书籍页面图片、OCR 和版面结构，生成页面内容的结构化描述。"
    },
    {
      "from": "gpt",
      "value": "{\n  \"page_main_topic\": \"……\",\n  \"reading_order\": [\"标题\", \"正文段落\", \"图表说明\"],\n  \"content_blocks\": []\n}"
    }
  ],
  "metadata": {
    "book_id": "aircraft_propulsion_2011",
    "page_index": 12,
    "task_type": "page_to_structured_description",
    "quality_score": 0.91
  }
}
```

### 3.3 Alpaca 格式

每条样本应包含：

- `instruction`：用户指令。
- `input`：原文、OCR、页面结构、图片占位符或图表上下文。
- `output`：结构化生成结果。
- `images`：可选，多模态任务包含图片路径。
- `metadata`：建议保留，便于回溯。

示例：

```json
{
  "instruction": "请对以下书籍段落生成忠实摘要，并保留关键术语。",
  "input": "段落原文：……",
  "output": "{\n  \"summary\": \"……\",\n  \"key_terms\": [\"……\"],\n  \"coverage_notes\": \"……\"\n}",
  "metadata": {
    "book_id": "aircraft_performance_dynamics_control",
    "page_index": 18,
    "task_type": "paragraph_summary"
  }
}
```

### 3.4 PT 纯文本格式

用于 LLaMA-Factory `stage: pt` 的领域知识注入任务，每行必须是一个完整 JSON 对象，且只包含 `text` 字段。不要包含 `instruction`、`input`、`output`、`messages`、`conversations`、`images` 或 `metadata`。

示例：

```json
{"text": "书名：飞机推进 [美法罗基著] 2011年版\n章节：第3章 压气机\n小节：3.1 基本概念\n\n压气机用于提高空气压力，并为后续燃烧过程提供所需的气流条件。……"}
{"text": "书名：飞机推进 [美法罗基著] 2011年版\n章节：第3章 压气机\n小节：3.2 工作过程\n\n在压气机工作过程中，气流通过叶片通道时压力和速度发生变化。……"}
```

## 4. 推荐目录与命名规范

输出目录建议如下。每本书单独一个目录，目录名建议包含稳定 `book_id` 和可读书名；同一本书下各任务类型必须分开保存，不要混写到同一个样本文件中。

```text
book_cpt/
  data/
    飞行员航空知识手册.pdf
    飞机的性能一动力学与控制.pdf
    飞机推进 [美法罗基著] 2011年版.pdf
  outputs/
    {book_id}_{book_name_safe}/
      manifest.jsonl
      pages/
        page_index.jsonl
      images/
        pages/
          ch{chapter_no}/p{page_no}.png
        blocks/
          ch{chapter_no}/p{page_no}/block_{block_id}_{block_type}.png
        extracted/
          ch{chapter_no}/p{page_no}/figure_{figure_id}.png
      mineru/
        raw/
        parsed/
      normalized/
        ch{chapter_no}/p{page_no}.json
      samples/
        raw/
          paragraph_summary.jsonl
          title_body_alignment.jsonl
          page_to_structured_description.jsonl
          chart_table_to_text.jsonl
          page_content_restatement.jsonl
          cross_page_synthesis.jsonl
          chapter_key_conclusions.jsonl
          evidence_to_conclusion_chain.jsonl
          domain_knowledge_corpus.jsonl
        validated/
          {task_type}.jsonl
        deduped/
          {task_type}.jsonl
      exports/
        sharegpt/
          {task_type}.jsonl
        alpaca/
          {task_type}.jsonl
        pt/
          domain_knowledge_corpus.jsonl
      logs/
        pipeline.log
        errors.jsonl
        metrics.json
        checkpoints.json
```

`book_id` 应从书名稳定生成，建议使用人工配置优先、自动 slug 兜底：

```json
{
  "飞行员航空知识手册.pdf": "pilot_aeronautical_knowledge_handbook",
  "飞机的性能一动力学与控制.pdf": "aircraft_performance_dynamics_control",
  "飞机推进 [美法罗基著] 2011年版.pdf": "aircraft_propulsion_2011"
}
```

样本 ID 建议格式：

```text
{book_id}_ch{chapter_no}_p{page_no}_{block_or_window_id}_{task_type}_{sample_no}
```

如果无法识别章节，则使用：

```text
{book_id}_ch_unknown_p{page_no}_{block_or_window_id}_{task_type}_{sample_no}
```

所有中间结构与最终样本必须保留以下溯源字段：

- `source_pdf`
- `book_name`
- `book_id`
- `chapter_title`
- `chapter_no`
- `page_index`
- `page_label`
- `block_id`
- `block_type`
- `bbox`
- `image_path`
- `mineru_parse_path`
- `normalized_page_path`
- `task_type`
- `created_at`
- `pipeline_version`

## 5. 中间结构标准

MinerU 或其他版面解析结果需要被标准化为统一页面结构，供后续模块使用。推荐页面结构：

```json
{
  "book_id": "aircraft_propulsion_2011",
  "book_name": "飞机推进 [美法罗基著] 2011年版",
  "source_pdf": "book_cpt/data/飞机推进 [美法罗基著] 2011年版.pdf",
  "chapter_no": "03",
  "chapter_title": "压气机",
  "page_index": 12,
  "page_label": "12",
  "page_image": "book_cpt/outputs/aircraft_propulsion_2011/images/pages/ch03/p012.png",
  "width": 1240,
  "height": 1754,
  "blocks": [
    {
      "block_id": "block_004",
      "block_type": "text",
      "bbox": [102, 231, 848, 512],
      "text": "……",
      "markdown": "……",
      "reading_order": 4,
      "confidence": 0.96,
      "paragraph_id": "para_003",
      "parent_title_id": "title_002",
      "image_path": "book_cpt/outputs/aircraft_propulsion_2011/images/blocks/ch03/p012/block_004_text.png"
    }
  ],
  "full_text": "……",
  "titles": [],
  "paragraphs": [],
  "tables": [],
  "figures": [],
  "formulas": [],
  "prev_page": 11,
  "next_page": 13
}
```

对于图表说明、标题管辖范围、跨页延续内容，标准化结构建议额外保留关系字段：

```json
{
  "semantic_links": [
    {
      "link_id": "link_001",
      "relation_type": "title_controls_body",
      "source": {"page_index": 12, "block_id": "title_002"},
      "targets": [
        {"page_index": 12, "block_id": "block_004"},
        {"page_index": 12, "block_id": "block_005"}
      ],
      "confidence": 0.88
    },
    {
      "link_id": "link_002",
      "relation_type": "figure_caption_and_text_explanation",
      "source": {"page_index": 12, "block_id": "figure_001"},
      "targets": [
        {"page_index": 12, "block_id": "caption_001"},
        {"page_index": 13, "block_id": "block_002"}
      ],
      "context_window_pages": [12, 13],
      "confidence": 0.82
    }
  ]
}
```

版面块类型至少包括：

- `title`
- `section_title`
- `text`
- `paragraph`
- `figure`
- `figure_caption`
- `table`
- `table_caption`
- `formula`
- `list`
- `header`
- `footer`
- `page_number`
- `reference`
- `unknown`

## 6. 模块划分与单模块提示词

### 6.1 PDF 接入模块

职责：

- 扫描输入目录中的 PDF。
- 建立书籍 manifest。
- 提取 PDF 基础信息，如页数、文件大小、文件哈希、可选元数据。
- 为每本书分配稳定 `book_id`。
- 判断是否已处理、是否需要断点续跑。

单测提示词：

```text
你是 PDF 接入模块开发助手。请实现一个模块，用于扫描 book_cpt/data/ 目录下的 PDF 文件，为每本书生成稳定的 book_id、source_pdf、page_count、file_hash、output_dir 等元数据，并写入 outputs/manifest.jsonl。模块需要支持重复运行不产生重复记录，支持断点续跑，发现文件变化时能够重新计算哈希并标记状态为 changed。请给出 Python 实现、输入输出结构、异常处理和单元测试。
```

### 6.2 页面渲染模块

职责：

- 将 PDF 每页渲染为图片。
- 支持 DPI、图片格式、最大边长、灰度/彩色配置。
- 保留页码与章节信息。
- 支持跳过已存在且校验通过的页面图片。

单测提示词：

```text
你是页面渲染模块开发助手。请实现一个模块，将 PDF 按页渲染为 PNG 图片，输出路径必须包含 book_id、章节号和页码，例如 book_cpt/outputs/{book_id}/images/pages/ch03/p012.png。模块需要支持多进程渲染、断点续跑、图片完整性校验和渲染失败重试。请输出核心代码、配置项、日志字段和测试用例。
```

### 6.3 版面解析模块

职责：

- 调用 MinerU 或可替换版面解析器对 PDF 或页面图片进行解析。
- 保存解析器原始输出。
- 提取文本、标题、表格、公式、图片区域、阅读顺序和坐标。
- 记录解析版本、耗时、失败原因。

单测提示词：

```text
你是版面解析模块开发助手。请实现一个模块，调用 MinerU 对书籍 PDF 或页面图片进行版面解析，保存 raw 和 parsed 两类结果，并输出每页对应的解析路径。模块需要支持批处理、多进程或任务队列、失败重试、解析缓存和解析结果完整性检查。请设计适配器接口，使后续可以替换 MinerU 版本或接入其他 OCR/版面分析工具。
```

### 6.4 版面标准化模块

职责：

- 将解析器输出转换为统一 `PageRecord` 结构。
- 统一坐标系、块类型、阅读顺序、章节标题。
- 合并页眉页脚、过滤噪声块。
- 标记正文块、图表块、公式块、标题块。
- 建立标题与正文、图表与图注、正文引用与图表之间的关系。

单测提示词：

```text
你是版面标准化模块开发助手。请实现一个模块，将 MinerU 或其他解析器输出转换为统一 PageRecord JSON 结构。要求统一 block_id、block_type、bbox、text、markdown、reading_order、confidence、paragraph_id、parent_title_id、image_path 等字段。模块需要识别并过滤页眉、页脚、页码等噪声内容，恢复章节层级，建立标题-正文、图表-图注、图表-正文解释关系，并为后续样本生成提供 full_text、titles、paragraphs、figures、tables、formulas、semantic_links 等聚合字段。请给出数据类定义、转换逻辑、异常兼容策略和测试样例。
```

### 6.5 图像与局部块裁剪模块

职责：

- 根据标准化结构中的 `bbox` 裁剪局部图片。
- 支持按文本块、图块、表格、公式裁剪。
- 支持扩大边界、最小尺寸过滤、重复裁剪跳过。
- 保存局部块图片并回写 `image_path`。

单测提示词：

```text
你是图像与局部块裁剪模块开发助手。请实现一个模块，根据 PageRecord 中每个 block 的 bbox 从整页图片中裁剪局部块图片。输出路径必须包含 book_id、章节、页码、block_id 和 block_type。模块需要处理 bbox 越界、空白区域、过小块、旋转页面和图片缺失等异常，并将裁剪后的 image_path 回写到 PageRecord。请给出实现代码和图像校验方法。
```

### 6.6 任务路由模块

职责：

- 根据页面内容、块类型、标题层级、图表关系和跨页上下文决定生成哪些任务。
- 不为任务类型设置固定比例、配额或强制均衡策略。
- 只要页面、段落、图表、章节片段或跨页窗口适合生成某类任务，就生成对应任务；同一页面可以同时生成多种任务。
- 对明显不适合的任务跳过，例如无清晰正文不生成段落摘要，无图表不生成图表解读，无多页关联不生成跨页归纳。
- 支持配置化任务开关和采样策略。

单测提示词：

```text
你是任务路由模块开发助手。请实现一个模块，根据 PageRecord、semantic_links、章节结构和跨页上下文决定要生成哪些样本任务，包括 paragraph_summary、title_body_alignment、page_to_structured_description、chart_table_to_text、page_content_restatement、cross_page_synthesis、chapter_key_conclusions、evidence_to_conclusion_chain、domain_knowledge_corpus。

路由要求：
1. 连续正文段落完整、OCR 置信度较高时，路由到 paragraph_summary。
2. 存在明确标题块及其下属正文块时，路由到 title_body_alignment。
3. 页面包含多种版面块，且整页图片与 OCR 可用时，路由到 page_to_structured_description。
4. 存在图、表、曲线、公式图、示意图或表格，且有图注、表注或正文解释时，路由到 chart_table_to_text。
5. 单页内容较完整、适合按原顺序换一种表达方式重述时，路由到 page_content_restatement。
6. 连续多页之间存在概念延续、图文解释跨页、步骤跨页或章节论述承接时，路由到 cross_page_synthesis。
7. 章节或较长小节具有明确结论、公式、条件、图表支持时，路由到 chapter_key_conclusions。
8. 页面或跨页窗口中存在可形成“证据 -> 解释 -> 结论”链条的内容时，路由到 evidence_to_conclusion_chain。
9. 连续页面或章节片段包含足够正文知识，适合进行 OCR 清洗、去噪和语义完整切块时，路由到 domain_knowledge_corpus。
10. 不要为了满足任务比例而生成低质量样本；是否生成只由适配性、证据充分性和质量规则决定。

模块需要支持按书籍、章节、页面、块类型进行任务判断，并输出 SampleJob 列表。请设计可配置任务开关、适配性判断规则、默认路由策略和单元测试。
```

## 7. 九类任务生成规范

### 7.1 段落摘要：`paragraph_summary`

输入：

- 一个或多个连续正文段落。
- 段落所属标题、页码、章节信息。
- 段落 OCR 置信度和阅读顺序。

输出字段：

- `summary`：忠实摘要。
- `key_terms`：保留的关键术语。
- `covered_points`：摘要覆盖的核心点。
- `omitted_details`：被压缩省略的细节类型。
- `source_pages`
- `evidence_block_ids`
- `evidence_text`
- `confidence`

生成提示词：

```text
你是航空书籍段落摘要生成专家。请根据给定书籍段落生成忠实摘要。

要求：
1. 摘要必须来自输入段落，不得引入外部知识。
2. 保留原文中的关键术语、变量名、对象名称和条件限制。
3. 不要改写成问答形式。
4. 不要机械摘抄整段原文，除非短语是专有术语或公式名称。
5. 如果段落 OCR 破碎或含义不完整，应输出空数组或降低 confidence。

输出 JSON 数组，每条包含：
- summary
- key_terms
- covered_points
- omitted_details
- source_pages
- evidence_block_ids
- evidence_text
- confidence
```

### 7.2 标题-正文对齐：`title_body_alignment`

输入：

- 标题块或小节标题块。
- 标题控制范围内的正文段落、列表、图表说明。
- 标题层级和前后标题。

输出字段：

- `title_text`
- `body_scope_summary`
- `alignment_label`：`aligned | partially_aligned | weakly_aligned | uncertain`
- `alignment_explanation`
- `body_key_points`
- `possible_scope_issue`
- `source_pages`
- `evidence_block_ids`
- `confidence`

生成提示词：

```text
你是书籍标题-正文结构分析专家。请判断给定标题与其下正文内容是否语义对齐。

要求：
1. 说明标题表达的主题，以及正文实际覆盖的内容范围。
2. 判断标题是否能概括正文，正文是否偏离标题，是否存在范围过宽、过窄或承接不清。
3. 只根据给定标题和正文判断，不引入书外知识。
4. 输出应帮助模型学习书籍标题如何组织知识，而不是生成考试问答。

输出 JSON 数组，每条包含：
- title_text
- body_scope_summary
- alignment_label
- alignment_explanation
- body_key_points
- possible_scope_issue
- source_pages
- evidence_block_ids
- confidence
```

### 7.3 页图到结构化描述：`page_to_structured_description`

输入：

- 单页整页图片。
- 页面 OCR。
- 标准化版面结构。
- 页面块阅读顺序。

输出字段：

- `page_main_topic`
- `layout_summary`
- `reading_order`
- `content_blocks`
- `figures_tables_formulas`
- `page_role_in_chapter`
- `uncertainty_notes`
- `source_pages`
- `evidence_block_ids`
- `visual_evidence`
- `confidence`

生成提示词：

```text
你是书籍页面结构化描述专家。请根据整页页面图片、OCR 和版面结构，生成该页面的结构化描述。

要求：
1. 描述页面主旨、标题、正文块、图表块、公式块和阅读顺序。
2. 区分页面可见布局信息、OCR 文本信息和推断出的页面作用。
3. 不要只摘要文字内容，也不要生成问答。
4. 如果页面含图表或公式，应说明其在页面论述中的位置和作用。
5. 对看不清、OCR 缺失或版面关系不确定的部分写入 uncertainty_notes。

输出 JSON 数组，每条包含：
- page_main_topic
- layout_summary
- reading_order
- content_blocks
- figures_tables_formulas
- page_role_in_chapter
- uncertainty_notes
- source_pages
- evidence_block_ids
- visual_evidence
- confidence
```

### 7.4 图表到文字解读：`chart_table_to_text`

输入：

- 图、表、曲线、示意图、结构图、公式图或表格块图片。
- 图注/表注。
- 正文引用和解释。
- 必要时包含前后页上下文。

输出字段：

- `object_type`：`figure | table | chart | diagram | formula_figure | other`
- `visible_structure`
- `caption_meaning`
- `textual_interpretation`
- `data_or_relation_points`
- `limitations`
- `source_pages`
- `evidence_block_ids`
- `visual_evidence`
- `evidence_text`
- `confidence`

生成提示词：

```text
你是航空书籍图表文字解读专家。请根据图表图片、图注/表注和正文解释，将图表内容转化为文字解读。

要求：
1. 先描述图表中可见结构，再结合图注和正文解释其含义。
2. 对曲线图、表格和示意图，应说明变量、关系、趋势、对比或结构组成。
3. 明确区分“图中可见”与“正文说明”。
4. 不得补充图中和上下文没有出现的外部知识。
5. 如果图像模糊、坐标轴不可读或表格字段缺失，应在 limitations 中说明。

输出 JSON 数组，每条包含：
- object_type
- visible_structure
- caption_meaning
- textual_interpretation
- data_or_relation_points
- limitations
- source_pages
- evidence_block_ids
- visual_evidence
- evidence_text
- confidence
```

### 7.5 页面内容重述：`page_content_restatement`

输入：

- 单页 OCR 文本。
- 页面标题、正文块、图表说明和公式说明。
- 页面阅读顺序。

输出字段：

- `restatement`
- `preserved_terms`
- `content_order`
- `coverage_notes`
- `source_pages`
- `evidence_block_ids`
- `evidence_text`
- `confidence`

生成提示词：

```text
你是书籍页面内容重述专家。请根据单页内容，用不同表达方式忠实重述该页知识。

要求：
1. 重述应覆盖页面主要内容，尽量保持原页面的论述顺序。
2. 保留关键术语、公式名称、图表编号、变量名和条件限制。
3. 不要生成问答，不要只列摘要提纲。
4. 不要大量逐字照抄原文；但专有名词、公式、定义性短语可以保留原文。
5. 页面内容过碎、OCR 质量低或上下文不足时，应降低 confidence。

输出 JSON 数组，每条包含：
- restatement
- preserved_terms
- content_order
- coverage_notes
- source_pages
- evidence_block_ids
- evidence_text
- confidence
```

### 7.6 跨页内容归纳：`cross_page_synthesis`

输入：

- 连续 2 到 4 页页面图片。
- 每页 OCR 和版面结构。
- 跨页标题、图表、正文解释和语义链接。

输出字段：

- `synthesis_topic`
- `page_roles`
- `cross_page_summary`
- `continuity_relations`
- `key_points_by_page`
- `source_pages`
- `evidence_text_by_page`
- `visual_evidence_by_page`
- `confidence`

生成提示词：

```text
你是跨页书籍内容归纳专家。请根据连续多页页面内容，归纳跨页延续的知识表达。

要求：
1. 样本必须依赖至少两页证据，不能只复述单页内容。
2. 说明每一页在连续论述中的作用，例如提出概念、给出图示、解释公式、展开条件、总结结论。
3. 优先处理图文解释跨页、表格跨页、步骤跨页、章节论述承接和概念递进。
4. 不要生成问答。
5. 如果多页之间没有清晰关联，应输出空数组。

输出 JSON 数组，每条包含：
- synthesis_topic
- page_roles
- cross_page_summary
- continuity_relations
- key_points_by_page
- source_pages
- evidence_text_by_page
- visual_evidence_by_page
- confidence
```

### 7.7 章节到关键结论提炼：`chapter_key_conclusions`

输入：

- 一个章节、一个小节或一个较长章节片段的结构化文本。
- 章节标题层级。
- 章节内图表、公式、表格和结论性段落。

输出字段：

- `chapter_scope`
- `key_conclusions`
- `supporting_evidence`
- `conditions_or_assumptions`
- `related_figures_tables_formulas`
- `concept_dependencies`
- `source_pages`
- `confidence`

生成提示词：

```text
你是航空书籍章节结论提炼专家。请根据章节或小节内容提炼关键结论。

要求：
1. 结论必须来自输入章节内容，不得引入外部知识。
2. 每条结论都应给出支撑它的原文、图表、公式或页码。
3. 如果结论有适用条件、前提假设或参数限制，必须明确写出。
4. 保留章节内概念之间的依赖关系。
5. 不要把章节改写成问答。

输出 JSON 数组，每条包含：
- chapter_scope
- key_conclusions
- supporting_evidence
- conditions_or_assumptions
- related_figures_tables_formulas
- concept_dependencies
- source_pages
- confidence
```

### 7.8 页面证据到结论链条：`evidence_to_conclusion_chain`

输入：

- 单页或跨页窗口中的正文、图表、公式、表格。
- 可形成结论的证据片段。
- 标题和章节上下文。

输出字段：

- `conclusion`
- `evidence_chain`
- `chain_steps`
- `source_pages`
- `evidence_block_ids`
- `evidence_text`
- `visual_evidence`
- `reasoning_scope`：`single_page | cross_page | chapter_fragment`
- `confidence`

生成提示词：

```text
你是书籍证据链构建专家。请根据页面或跨页内容，生成从证据到结论的可追溯链条。

要求：
1. 链条必须从书中明确证据开始，经过必要解释，再到结论。
2. 每一步都要标注使用了哪些原文、图表、公式或页面信息。
3. 不要引入外部知识，不要凭常识补全缺失环节。
4. 如果证据不足以支持结论，应输出空数组或在 confidence 中体现不确定。
5. 结论应是书中内容可支持的归纳，不是模型自由发挥。

输出 JSON 数组，每条包含：
- conclusion
- evidence_chain
- chain_steps
- source_pages
- evidence_block_ids
- evidence_text
- visual_evidence
- reasoning_scope
- confidence
```

### 7.9 领域知识预训练语料：`domain_knowledge_corpus`

输入：

- 解析后的连续页面、章节片段或小节片段纯文本。
- 书名、章节路径、小节标题或页码范围。
- OCR 文本、Markdown 表格、公式、图注、表注和版面块类型。

输出字段：

- `text`

导出格式：

- `pt`
- 最终文件：`exports/pt/domain_knowledge_corpus.jsonl`
- 每行一个 JSON 对象，只包含 `text` 字段。

生成提示词：

```text
你是一个领域书籍数据清洗与预训练语料构建助手。你的任务是把我提供的解析后书籍文本处理成适合大语言模型继续预训练（stage: pt）的高质量文本语料。

重要规则：
1. 不要生成问答格式，不要生成 instruction/input/output。
2. 不要添加原文没有的知识、案例、结论或解释。
3. 不要过度摘要，尽量保留原书的知识密度、术语、定义、推导、公式、步骤和结构。
4. 可以修正明显的 OCR 错字、断行、空格、页眉页脚、页码、重复标题、乱码。
5. 删除目录、版权页、索引、参考文献列表、空白页、水印、重复页眉页脚等无训练价值内容。
6. 保留书名、章节标题、小节标题，并放在每个文本块开头。
7. 表格如果结构清晰，请转换成 Markdown 表格；如果表格解析混乱，请转换成准确的自然语言描述。
8. 公式请尽量保留原始形式，必要时用 LaTeX 表达；不要自行推导新公式。
9. 图注、表注可以保留；如果没有图片内容，不要凭空描述图片。
10. 按语义完整性切块，每个块应是连续、可独立阅读的文本片段。不要在句子、公式、表格中间截断。
11. 每个文本块长度控制在约 1200-2500 个中文字符；如果原文很短，可以合并相邻同小节内容；如果过长，可以拆成多个连续块。
12. 输出必须是 JSONL，每行一个 JSON 对象，只包含 text 字段。
13. text 字段内部格式为：
书名：{书名}
章节：{章节标题}
小节：{小节标题，可选}

{清洗后的正文}

输出格式示例：
{"text": "书名：领域教材示例\n章节：第1章 基础概念\n小节：1.1 研究对象与基本定义\n\n本节主要介绍该领域的研究对象、基本术语和核心问题。……"}
{"text": "书名：领域教材示例\n章节：第1章 基础概念\n小节：1.2 基本原理\n\n在上一节概念定义的基础上，本节进一步说明基本原理、适用条件和典型应用。……"}

注意：
- 上面是 JSONL 示例，不是 JSON 数组。
- 每一行必须是一个完整 JSON 对象。
- 不要在输出外层添加 Markdown 代码块，例如三个反引号加 json 的代码块标记。
- 不要输出解释说明，只输出 JSONL 数据。
- 如果输入片段主要是目录、版权页、索引、参考文献列表、空白页、水印或无法恢复的噪声，请不输出任何行。

输入信息：
书名：{填入书名}
章节路径：{填入章节/小节路径}
原始解析文本：
{粘贴解析后的文本}

请输出 JSONL，不要输出解释、不要输出 Markdown 代码块。
```

质检规则：

- 最终导出行只能包含 `text` 字段，不得包含 `instruction`、`input`、`output`、`messages`、`images` 或 `metadata`。
- `text` 内容必须能回溯到输入的 OCR/Markdown 解析文本，不得添加外部知识。
- 删除目录、版权页、索引、参考文献列表、页眉页脚、页码、水印和重复标题等低价值内容。
- 保留书名、章节标题、小节标题、关键术语、定义、公式、表格和图表注释。
- 表格和公式不得被截断；语义切块不得在句子、公式或表格中间断开。
- 单块长度建议约 1200-2500 个中文字符；短片段可合并，长片段可拆分。
- 如果输入噪声过多或无法构成可读知识片段，应输出空内容。

## 8. 样本校验模块

职责：

- 校验生成内容是否有证据支撑。
- 校验图片路径是否存在。
- 校验结构化字段是否完整。
- 校验输出是否符合对应任务类型。
- 校验是否存在外部知识注入、过度推断、无证据结论、页码错误。
- 校验是否过度复述原文；摘要和重述任务需要区别处理。
- 校验 ShareGPT/Alpaca/PT 格式是否合法。

校验提示词：

```text
你是书籍原始知识表达训练样本质量校验专家。请检查给定样本是否适合用于书籍 SFT/多模态训练或 stage: pt 继续预训练。

检查标准：
1. 输出是否符合 task_type 对应任务目标。
2. 生成内容是否能被 OCR 文本、版面结构、页面图片或图表证据支持。
3. 是否存在明显幻觉、外部知识注入、页码错误、图片路径错误。
4. 是否保留了必要的书籍原始表达、关键术语和证据字段。
5. 摘要是否过度摘抄，重述是否过度压缩，图表解读是否混淆可见信息与正文解释。
6. 跨页样本是否确实依赖至少两页证据。
7. 证据链样本是否每一步都有来源支撑。
8. PT 语料是否只包含 text 字段，且没有 instruction/input/output 或对话字段。
9. 是否存在格式错误、字段缺失、输出为空、重复或高度模板化。

输出：
- is_valid: true | false
- quality_score: 0 到 1
- reject_reason
- fix_suggestion
```

## 9. 去重模块

职责：

- 根据任务类型、输入证据、输出文本、页面和块 ID 进行去重。
- 支持精确去重和语义近似去重。
- 避免同一页面大量模板化样本。
- 保留不同任务对同一证据的合理多视角表达。

单测提示词：

```text
你是训练样本去重模块开发助手。请实现一个模块，对生成的书籍原始知识表达样本进行去重。要求支持基于 id 的精确去重、基于 input_evidence_hash 的证据去重、基于 output_payload 文本哈希去重、基于 embedding 或相似度的近似去重，并能按 book_id、page_index、chapter_no、task_type 控制每页或每章节最大样本数。请给出代码、阈值建议、日志统计和测试用例。
```

## 10. 导出模块

### 10.1 ShareGPT 导出模块

职责：

- 将已校验样本导出为 LLaMA-Factory 可用 ShareGPT 格式。
- 支持多图、单图、纯文本样本。
- 保留 metadata。
- 同一本书下必须按任务类型分别导出文件，不合并为单个 ShareGPT 文件。
- 支持 jsonl 与 json 数组两种形式。

单测提示词：

```text
你是 ShareGPT 导出模块开发助手。请实现一个模块，将 validated/deduped 样本转换为 LLaMA-Factory 可用的 ShareGPT 格式。要求正确处理 images 字段、<image> 占位符、conversations 字段、metadata 字段。同一本书目录下必须按任务类型分别导出独立 jsonl 文件，不允许把多种任务合并到同一个 train.jsonl。当前版本不要求实现数据集划分。请给出格式转换代码和合法性检查测试。
```

### 10.2 Alpaca 导出模块

职责：

- 将样本导出为 Alpaca 格式。
- 支持纯文本任务和带图 Alpaca 扩展格式。
- 同一本书下必须按任务类型分别导出文件，不合并为单个 Alpaca 文件。

单测提示词：

```text
你是 Alpaca 导出模块开发助手。请实现一个模块，将 validated/deduped 样本转换为 Alpaca 格式。要求包含 instruction、input、output、images 和 metadata 字段。同一本书目录下必须按任务类型分别导出独立 jsonl 文件，不允许把多种任务合并到同一个 train.jsonl。当前版本不要求 train/valid/test 数据集划分。请给出转换代码、字段校验和样例输出。
```

### 10.3 PT 导出模块

职责：

- 将 `domain_knowledge_corpus` 样本导出为 LLaMA-Factory `stage: pt` 可用 JSONL。
- 每行只包含 `text` 字段，不包含 `instruction`、`input`、`output`、`messages`、`conversations`、`images` 或 `metadata`。
- 同一本书下必须按任务类型分别导出文件，路径为 `exports/pt/domain_knowledge_corpus.jsonl`。
- 中间样本仍可保留 metadata 和 evidence 便于回溯，但最终导出文件必须保持纯训练语料格式。

单测提示词：

```text
你是 PT 纯文本导出模块开发助手。请实现一个模块，将 validated/deduped 中 task_type 为 domain_knowledge_corpus 的样本转换为 LLaMA-Factory stage: pt 可用 JSONL。要求每行 JSON 对象只包含 text 字段，过滤空 text，禁止导出 instruction/input/output、messages、images、metadata 等字段。同一本书目录下必须导出到 exports/pt/domain_knowledge_corpus.jsonl。请给出转换代码、字段校验和样例输出。
```

## 11. 日志、断点续跑与效率要求

系统需要优先使用流式处理，避免一次性将整本书全部加载到内存。推荐策略：

1. PDF 级并行：不同书籍可并行处理。
2. 页面级并行：渲染、版面解析、标准化、裁剪可按页并行。
3. 样本生成并行：LLM/VLM 调用可使用异步队列、速率限制和失败重试。
4. 流式写入：中间结构和样本使用 JSONL 逐行写入。
5. 缓存复用：页面图片、解析结果、标准化结构、局部裁剪图片均应可缓存。
6. 断点续跑：每个阶段写入 checkpoint，重跑时跳过已完成且校验通过的文件。
7. 批量校验：格式校验可本地批量执行，LLM 校验只用于高风险样本。
8. 背压控制：LLM 生成速度慢时，任务队列不能无限堆积。
9. 失败隔离：单页失败不影响整本书继续处理。
10. 指标统计：按书籍、章节、页码、任务类型统计耗时和产出。

日志模块单测提示词：

```text
你是流水线日志与断点续跑模块开发助手。请实现一个统一日志模块，记录 PDF 接入、页面渲染、版面解析、版面标准化、样本生成、校验、去重、导出各阶段的状态。日志需要同时输出人类可读 pipeline.log 和机器可读 errors.jsonl、metrics.json、checkpoints.json。请设计状态枚举、日志字段、异常捕获装饰器和断点恢复逻辑。
```

## 12. 质量控制规则

样本生成后至少需要执行以下检查：

- 图片文件是否存在且可读取。
- 图片路径是否与 metadata 中的书名、页码、块 ID 匹配。
- OCR 证据是否能够支撑生成内容。
- 纯文本任务是否没有不必要的图片占位符。
- 多模态样本是否包含 `<image>` 和 `images` 字段。
- 跨页样本是否至少包含两个不同页码。
- 段落摘要是否忠实、简洁、保留关键术语。
- 标题-正文对齐是否使用标题控制范围内的正文证据。
- 页图结构化描述是否覆盖页面布局、阅读顺序和主要内容块。
- 图表解读是否区分可见信息、图注信息和正文解释信息。
- 页面重述是否保持页面知识顺序，且没有压缩成简单摘要。
- 章节结论是否每条都有证据来源、条件或适用范围。
- 证据链是否每一步都有来源支撑。
- 答案不能直接复述大量 OCR 原文，除非任务目标是页面重述且有必要保留术语。
- 输出不能高度模板化。
- 同一页面同一任务类型样本数量不能过多。
- 对低 OCR 置信度页面降低采样率。
- 表格、公式、图注类样本需要保留明确证据块。

## 13. 最终代码生成总提示词

```text
你是资深 Python 数据工程师和多模态数据集构建专家。请根据以下规格，为 book_cpt/data/ 目录中的航空书籍 PDF 实现一套书籍原始知识表达数据生成流水线。

输入：
- book_cpt/data/飞行员航空知识手册.pdf
- book_cpt/data/飞机的性能一动力学与控制.pdf
- book_cpt/data/飞机推进 [美法罗基著] 2011年版.pdf

输出：
- LLaMA-Factory ShareGPT 格式数据
- Alpaca 格式数据
- LLaMA-Factory stage: pt 纯文本 JSONL 数据
- 可追溯的页面图片、局部块图片、版面解析结果、中间结构、日志和指标

必须支持的任务类型：
1. paragraph_summary：段落摘要，基于连续正文段落生成忠实摘要、关键术语和覆盖点。
2. title_body_alignment：标题-正文对齐，判断标题与其下正文范围、语义覆盖和偏离风险。
3. page_to_structured_description：页图到结构化描述，基于整页图片、OCR 和版面结构生成页面结构描述。
4. chart_table_to_text：图表到文字解读，基于图表、图注、表注和正文引用生成文字化解读。
5. page_content_restatement：页面内容重述，基于单页内容按原知识顺序进行忠实重述。
6. cross_page_synthesis：跨页内容归纳，基于连续多页窗口归纳跨页延续的知识表达。
7. chapter_key_conclusions：章节到关键结论提炼，基于章节或小节提炼关键结论、条件和证据。
8. evidence_to_conclusion_chain：页面证据到结论链条，生成证据片段到中间解释再到结论的可追溯链条。
9. domain_knowledge_corpus：领域知识预训练语料，基于解析后的书籍文本进行清洗、去噪、保结构切块，生成只包含 text 字段的 stage: pt JSONL 语料。

任务生成机制：
- 默认生成全部任务类型。
- 支持通过命令行 `--task` 或 `--tasks` 指定只生成部分任务。
- 指定任务后，路由、LLM/VLM provider 白名单、中间样本文件和导出文件都只保留指定任务。
- 未指定任务时，不应因为新增任务而关闭任何默认任务。

必须实现的模块：
- PDF 接入模块
- 页面渲染模块
- 版面解析模块
- 版面标准化模块
- 图像与局部块裁剪模块
- 任务路由模块
- 九类任务生成模块
- 样本校验模块
- 去重模块
- ShareGPT 导出模块
- Alpaca 导出模块
- PT 导出模块
- 日志与断点续跑模块

工程要求：
1. 使用多进程、异步队列或批处理提升处理速度。
2. 使用 JSONL 流式保存中间结果和样本，避免大内存占用。
3. 每个阶段都要支持断点续跑。
4. 文件名、目录名、样本 ID 和 metadata 必须包含 book_id、章节、页码、block_id、task_type 等溯源信息。
5. 每个模块都要能单独测试。
6. 样本生成必须经过格式校验、证据校验、去重和质量评分。
7. 代码应配置化，支持调整 DPI、跨页窗口大小、任务开关、任务适配规则、每页样本数、并发数、导出格式和命令行自定义任务选择。
8. 对版面解析器、LLM/VLM 调用和导出格式设计适配器接口，方便后续替换实现。
9. 失败页面必须记录错误并跳过，不得中断全局任务。
10. 需要识别和保存标题-正文、图表-图注、图表-正文解释、跨页承接关系。
11. 所有任务类型不设置固定比例；只要页面、段落、图表、章节片段或跨页窗口适合生成某类任务，就允许生成该类任务。
12. 同一本书的各任务类型必须在该书目录下分别保存为独立文件，不要合并保存到同一个样本文件或同一个 train.jsonl。
13. domain_knowledge_corpus 的最终导出必须是 `exports/pt/domain_knowledge_corpus.jsonl`，每行只包含 `text` 字段，不包含问答、对话、图片或 metadata 字段。
14. 当前版本不要求实现数据集划分和版权处理；OCR 置信度、页面类型过滤、人工抽检、版本管理等可以作为可选增强项存在。
15. 最终请给出项目目录结构、核心代码、配置文件、命令行入口、测试用例和运行示例。

请先输出整体架构和数据流，再逐步实现每个模块。实现时优先保证数据可追溯、结果可复现、格式可被 LLaMA-Factory 直接使用，并让模型尽量学习书籍原始知识表达，而不是只学习问答结果。
```
