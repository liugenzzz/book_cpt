from __future__ import annotations

import json


TASK_TYPES = [
    "paragraph_summary",
    "title_body_alignment",
    "page_to_structured_description",
    "chart_table_to_text",
    "page_content_restatement",
    "cross_page_synthesis",
    "chapter_key_conclusions",
    "evidence_to_conclusion_chain",
    "domain_knowledge_corpus",
]

PROMPTS = {
    "system": "你是多模态书籍数据处理专家、数据工程架构师和航空领域教材数据集构建专家。所有输出必须忠实基于输入证据，不引入外部知识。",
    "paragraph_summary": """
你是航空书籍段落摘要专家。请根据输入中的一个或多个连续正文段落生成忠实、短小、覆盖关键概念的摘要。
要求：只使用输入证据；保留关键术语、公式名、变量名和限定条件；不要写成问答；证据不足时返回 []。
输出 JSON 数组，每条包含 summary、key_terms、coverage_notes、source_pages、evidence_block_ids、evidence_text、confidence。
""",
    "title_body_alignment": """
你是书籍标题-正文对齐审校专家。请根据标题块及其控制范围内的正文，判断标题语义、正文范围、覆盖关系和不匹配风险。
要求：说明标题是否能概括正文；指出偏题、范围过宽/过窄、上下文缺失等风险；只使用输入证据；不要生成问答。
输出 JSON 数组，每条包含 title_text、body_scope、alignment_judgement、coverage_relation、mismatch_risks、source_pages、evidence_block_ids、evidence_text、confidence。
""",
    "page_to_structured_description": """
你是书籍页面结构化描述专家。请根据整页页面图片、OCR 和版面结构，生成页面的结构化描述。
要求：覆盖标题、正文块、图表块、公式块、阅读顺序和页面主旨；区分可见版面信息与 OCR 文本信息；不要生成问答。
输出 JSON 数组，每条包含 page_main_topic、reading_order、content_blocks、figures_tables_formulas、layout_notes、source_pages、evidence_block_ids、evidence_text、visual_evidence、confidence。
""",
    "chart_table_to_text": """
你是航空教材图表解读专家。请根据目标图、表、曲线、示意图、公式或表格，以及图注、表注和正文引用，生成文字化解读。
要求：区分 visible_information、caption_information 和 text_explanation；强调变量关系、趋势、对比和书中解释；证据不足时返回 []。
输出 JSON 数组，每条包含 target_type、target_description、visible_information、caption_information、text_explanation、interpretation、source_pages、evidence_block_ids、evidence_text、visual_evidence、confidence。
""",
    "page_content_restatement": """
你是书籍页面内容重述专家。请基于单页完整内容，用不同表达方式忠实重述该页知识。
要求：保持原页面知识顺序；保留必要术语、公式名、图表编号、变量名和条件限制；不要写成摘要或问答；不要逐字大段照抄。
输出 JSON 数组，每条包含 restatement、preserved_terms、content_order、coverage_notes、source_pages、evidence_block_ids、evidence_text、confidence。
""",
    "cross_page_synthesis": """
你是跨页书籍内容归纳专家。请根据连续多页页面内容，归纳跨页延续的概念、图文关系、步骤、论述链或互补信息。
要求：必须依赖至少两页证据；说明每页在连续论述中的作用；多页没有清晰关联时返回 []；不要生成问答。
输出 JSON 数组，每条包含 synthesis_topic、page_roles、cross_page_summary、continuity_relations、key_points_by_page、source_pages、evidence_text_by_page、visual_evidence_by_page、confidence。
""",
    "chapter_key_conclusions": """
你是航空书籍章节结论提炼专家。请根据章节、小节或较长片段内容提炼关键结论、适用条件和证据。
要求：每条结论都要有原文、图表、公式或页码支持；说明条件、假设、参数限制和概念依赖；不要改写成问答。
输出 JSON 数组，每条包含 chapter_scope、key_conclusions、supporting_evidence、conditions_or_assumptions、related_figures_tables_formulas、concept_dependencies、source_pages、confidence。
""",
    "evidence_to_conclusion_chain": """
你是书籍证据链构建专家。请根据页面或跨页证据，生成从证据片段到中间解释再到结论的可追溯链条。
要求：每一步都标注使用的原文、图表、公式或页面信息；不引入外部知识；证据不足时返回 []。
输出 JSON 数组，每条包含 conclusion、evidence_chain、chain_steps、source_pages、evidence_block_ids、evidence_text、visual_evidence、reasoning_scope、confidence。
""",
    "domain_knowledge_corpus": """
你是一个领域书籍数据清洗与预训练语料构建助手。你的任务是把解析后的书籍文本处理成适合大语言模型继续预训练（LLaMA-Factory stage: pt）的高质量纯文本语料。

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
章节：{章节标题或页码范围}
小节：{小节标题，可选}

{清洗后的正文}

注意：
- 不要在输出外层添加 Markdown 代码块。
- 不要输出解释说明，只输出 JSONL 数据。
- 如果输入片段主要是目录、版权、索引、参考文献、空白页或无法恢复的噪声，请不输出任何行。
""",
    "validator": "你是书籍原始知识表达训练样本质量校验专家。请检查样本是否符合 task_type 目标、是否有证据支持、是否存在幻觉或格式错误。",
}

PROMPTS.update(
    {
        "paragraph_summary": "你是段落摘要生成专家。请基于 paragraph_text 生成摘要样本。输出 JSON 数组，每条必须包含 instruction、summary，可选 keywords；instruction 只能围绕段落摘要任务做多样化表达。",
        "title_body_alignment": "你是标题-正文对齐判断专家。请基于 title 和 body_text 判断是否对齐。输出 JSON 数组，每条必须包含 instruction、label、reason；label 应为“对齐”“不对齐”或“部分对齐”等明确判断。",
        "page_to_structured_description": "你是页面结构化描述专家。请基于 page_image 及辅助 OCR/版面证据生成结构化页面信息。输出 JSON 数组，每条必须包含 instruction、page_type、title、sections、main_topics、key_elements。",
        "chart_table_to_text": "你是图表文字化解读专家。请基于 chart_image 及辅助上下文说明图中主要趋势和结论。输出 JSON 数组，每条必须包含 instruction、chart_subject、trends、conclusion。回答字段中不得出现图号、章节号、本文、文中、书中、本书、该书、本章、本节、章节等来源位置词，也不要写图1、表2、figure 3 等编号表达。",
        "page_content_restatement": "你是页面内容重述专家。请基于 page_image 及辅助 OCR/版面证据，在不改变原意的前提下重述页面内容。输出 JSON 数组，每条必须包含 instruction、restatement。",
        "cross_page_synthesis": "你是跨页内容归纳专家。请综合 page_images 中所有页面内容进行归纳。输出 JSON 数组，每条必须包含 instruction、topic、page1_focus、page2_focus、combined_summary。",
        "chapter_key_conclusions": "你是章节关键结论提炼专家。请基于 chapter_title 和 chapter_text 提炼章节核心结论。输出 JSON 数组，每条必须包含 instruction、chapter_topic、key_conclusions。",
        "evidence_to_conclusion_chain": "你是证据到结论链条构建专家。请基于 page_image 及辅助证据生成证据、推理步骤和结论。输出 JSON 数组，每条必须包含 instruction、evidence、reasoning、conclusion。",
        "domain_knowledge_corpus": PROMPTS["domain_knowledge_corpus"],
    }
)

CFG = {
    "pipeline_version": "book_prompt_v1",
    "encoding": "utf-8",
    "sharegpt_image_token": "<image>",
    "pdf_suffix": ".pdf",
    "pdf_page_regex": r"/Type\s*/Page\b",
    "output_dir_name": "outputs",
    "default_chapter_no": "unknown",
    "default_block_id": "page",
    "default_block_type": "page",
    "unknown_book_id": "untitled_book",
    "timestamp_format": "%Y-%m-%dT%H:%M:%SZ",
    "hash_chunk_size": 1048576,
    "logger_name": "book_cpt",
    "statuses": {"ready": "ready", "done": "done"},
    "paths": {
        "skipped_books": "skipped_books.jsonl",
        "manifest": "manifest.jsonl",
        "pages_manifest": "pages/page_index.jsonl",
        "page_images": "images/pages/ch{chapter_no}/p{page_no:03d}.png",
        "block_images": "images/blocks/ch{chapter_no}/p{page_no:03d}/{block_id}_{block_type}.png",
        "extracted_images": "images/extracted",
        "image_map": "mineru/image_map.json",
        "mineru_raw": "mineru/raw/{book_id}.json",
        "mineru_parsed": "mineru/parsed/{book_id}.json",
        "mineru_status": "mineru/status/{book_id}.json",
        "normalized_page": "normalized/ch{chapter_no}/p{page_no:03d}.json",
        "sample_cache_state": "samples/cache_state.json",
        "sample_generation_state": "samples/generation_state.json",
        "sample_raw": "samples/raw/{task_type}.jsonl",
        "sample_validated": "samples/validated/{task_type}.jsonl",
        "sample_deduped": "samples/deduped/{task_type}.jsonl",
        "export_sharegpt": "exports/sharegpt/{task_type}.jsonl",
        "export_alpaca": "exports/alpaca/{task_type}.jsonl",
        "export_pt": "exports/pt/{task_type}.jsonl",
        "pipeline_log": "logs/pipeline.log",
        "errors": "logs/errors.jsonl",
        "metrics": "logs/metrics.json",
        "checkpoints": "logs/checkpoints.json",
    },
    "extracted_image_naming": {
        "template": "p{page_label}_img{image_no:03d}{suffix}",
        "unknown_page_label": "unknown",
        "allowed_suffixes": [".png", ".jpg", ".jpeg", ".webp", ".bmp"],
    },
    "task_types": TASK_TYPES,
    "export_formats": {
        "sharegpt": [
            "page_to_structured_description",
            "chart_table_to_text",
            "page_content_restatement",
            "cross_page_synthesis",
            "evidence_to_conclusion_chain",
        ],
        "alpaca": [
            "paragraph_summary",
            "title_body_alignment",
            "chapter_key_conclusions",
        ],
        "pt": [
            "domain_knowledge_corpus",
        ],
    },
    "block_types": {
        "title": "title",
        "section": "section_title",
        "section_title": "section_title",
        "text": "text",
        "paragraph": "text",
        "image": "figure",
        "figure": "figure",
        "figure_caption": "figure_caption",
        "caption": "figure_caption",
        "table": "table",
        "table_caption": "table_caption",
        "formula": "formula",
        "equation": "formula",
        "list": "list",
        "header": "header",
        "footer": "footer",
        "page_number": "page_number",
        "reference": "reference",
        "unknown": "unknown",
    },
    "runtime": {
        "input_dir": "book_cpt/data",
        "input_books": [],
        "output_root": "book_cpt/outputs",
        "cache_only": False,
        "recursive": False,
        "book_workers": 4,
        "max_workers": 8,
        "page_workers": 4,
        "crop_workers": 4,
        "vlm_max_pending": 8,
        "vlm_min_interval_seconds": 0.0,
        # provider 冷却状态跨进程共享，读缓存的 TTL（秒）。
        "cooldown_refresh_seconds": 1.0,
        # 生成阶段每完成多少个 job（或间隔多少秒）打一条带 ETA 的进度日志。
        "generation_progress_every": 10,
        "generation_progress_seconds": 60.0,
        # 断点 checkpoint 攒批落盘的阈值，避免每个 job 都整份重写。
        "generation_state_flush_every": 20,
        "generation_state_flush_seconds": 10.0,
        "reuse_mineru": False,
        "reuse_normalized": True,
        "reuse_samples": True,
        "reuse_exports": True,
        "force_rebuild": False,
        "skip_vlm": False,
        "render_pages": True,
        # 页面图已存在时默认跳过重渲染；去水印改写了 PDF 时流水线会自动置 True。
        "rerender_pages": False,
        "crop_blocks": True,
        "log_level": "INFO",
        "progress": True,
    },
    "watermark": {
        # 书籍版式比期刊杂，去水印有误删正文可选内容组的风险，默认关闭。
        # 打开后会把整册重复出现的水印 Form XObject 调用剔掉，另存一份干净 PDF 再送 MinerU。
        "enabled": False,
        "form_names": [],
        "image_sizes": [],
        "min_page_coverage": 0.6,
        "cleaned_pdf_path": "preprocessed/{book_id}_cleaned.pdf",
        "report_path": "preprocessed/watermark_cleaning.json",
    },
    "render": {"dpi": 180, "image_format": "png", "max_side": 2200, "retry_count": 2},
    "crop_filter": {
        "enabled": True,
        "min_width": 28,
        "min_height": 32,
        "min_area": 3000,
        "min_text_chars": 18,
        "padding": 10,
        "text_block_types": ["title", "section_title", "text", "figure_caption", "table_caption", "list"],
        "visual_block_types": ["figure", "table", "formula"],
        "min_text_content_margin": 3,
        "edge_ink_band": 2,
        "max_edge_ink_ratio": 0.01,
        "min_visual_container_coverage": 0.75,
        "visual_container_overlap_ratio": 0.9,
        "white_pixel_threshold": 245,
        "max_blank_ratio": 0.985,
        "min_non_white_ratio": 0.005,
        "min_intensity_stddev": 3.0,
        "skip_block_types": ["header", "footer", "page_number", "unknown"],
        "skip_text_patterns": [r"^\s*第?\s*\d+\s*页\s*$", r"^.{1,30}\s+\d{1,4}\s*$"],
    },
    "mineru": {
        "url": "http://192.168.78.36:7086",
        "backend": "hybrid-auto-engine",
        "parse_method": "ocr",
        "lang_list": ["ch"],
        "timeout": 3600,
        "formula_enable": True,
        "table_enable": True,
        "return_md": True,
        "return_middle_json": True,
        "return_content_list": True,
        "return_images": True,
        "response_format_zip": False,
        "return_original_file": False,
        "max_concurrency": 2,
        "slot_poll_seconds": 2,
        "slot_stale_seconds": 7200,
        "retry_count": 3,
        "retry_backoff_seconds": 5,
        "retry_backoff_multiplier": 2,
        "cooldown_seconds": 300,
        "min_content_items": 1,
        "min_page_coverage": 0.8,
        "min_text_chars": 100,
        # 多实例时在这里逐个列出；每条继承上面的通用配置，只覆盖 url / server_url /
        # max_concurrency / weight 这类实例相关字段。留空则退回单实例（用上面的 url）。
        "providers": [
            {
                "name": "mineru_1",
                "url": "http://192.168.78.36:7086",
                "max_concurrency": 2,
                "weight": 1,
            },
        ],
    },
    "vlm_pool": {
        "strategy": "least_busy_weighted_fallback",
        "fallback": {"enabled": True, "max_attempts": 2, "cooldown_seconds": 300},
        "providers": [
            {
                "name": "fx_q3_235",
                "url": "https://mcc-pre.3xmt.com/gateway/ai-service/v1/chat/completions",
                "model": "fx-q3-235",
                "api_key": "sk-2h1RwMjqYcdh6F5Fs5",
                "stream": False,
                "temperature": 0.4,
                "max_tokens": 8192,
                "timeout": 240,
                "chat_template_kwargs": {"enable_thinking": False},
                "capabilities": ["text", "image"],
                "task_types": TASK_TYPES,
                "weight": 1,
                "max_concurrency": 4,
            },
            {
                "name": "Qwen3.6-27B",
                "url": "http://192.168.78.36:3012/v1/chat/completions",
                "model": "Qwen3.6-27B",
                "api_key": "sk-bveYeVn6NAdRRElTWCqhtyJbkTL5XwweedczV9FJ05kDqhqX",
                "stream": False,
                "temperature": 0.6,
                "max_tokens": 8192,
                "timeout": 240,
                "capabilities": ["text", "image"],
                "task_types": TASK_TYPES,
                "weight": 2,
                "max_concurrency": 4,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            {
                "name": "InternVL3_5-38B",
                "url": "http://192.168.78.35:8879/intern-vl/v1/chat/completions",
                "model": "InternVL3_5-38B",
                "api_key": "",
                "stream": False,
                "temperature": 0.6,
                "max_tokens": 8192,
                "timeout": 240,
                "chat_template_kwargs": {},
                "capabilities": ["text", "image"],
                "task_types": TASK_TYPES,
                "weight": 3,
                "max_concurrency": 6,
            },
        ],
    },
    "routing": {
        "enabled_tasks": {task_type: True for task_type in TASK_TYPES},
        "min_page_text_chars": 200,
        "min_block_text_chars": 20,
        "min_chapter_text_chars": 1000,
        "min_domain_corpus_text_chars": 300,
        "cross_page_window": 3,
        "chapter_window": 6,
        "domain_corpus_window": 2,
        "domain_corpus_target_input_chars": 3600,
        "visual_block_types": ["figure", "table", "formula"],
        "text_block_types": ["text", "paragraph", "list"],
        "title_block_types": ["title", "section_title"],
        "ocr_block_types": ["title", "section_title", "text", "figure_caption", "table_caption", "table", "formula", "list"],
    },
    "generation": {
        "samples_per_job": {task_type: 1 for task_type in TASK_TYPES},
        "max_page_context_chars": 5000,
        "max_block_context_chars": 3600,
        "max_neighbor_context_chars": 5000,
        "max_pt_context_chars": 9000,
        "max_image_bytes": 2097152,
        "max_image_side": 1600,
        "image_jpeg_quality": 85,
    },
    "validation": {
        "min_question_chars": 4,
        "min_answer_chars": 8,
        "default_quality_score": 0.8,
        "low_quality_score": 0.35,
        "quality_review_enabled": True,
        "quality_review_min_score": 0.65,
        "quality_review_min_dimension_score": 0.60,
        "quality_review_fail_closed": True,
        "min_pt_source_coverage_score": 0.70,
        "min_pt_output_source_ratio": 0.65,
        "max_missing_pt_section_headings": 0,
        "max_samples_per_page_task": 8,
        "dedup_similarity_threshold": 0.92,
        "required_output_fields": {
            "paragraph_summary": ["summary"],
            "title_body_alignment": ["label", "reason"],
            "page_to_structured_description": ["page_type", "sections", "main_topics", "key_elements"],
            "chart_table_to_text": ["chart_subject", "trends", "conclusion"],
            "page_content_restatement": ["restatement"],
            "cross_page_synthesis": ["topic", "combined_summary"],
            "chapter_key_conclusions": ["chapter_topic", "key_conclusions"],
            "evidence_to_conclusion_chain": ["evidence", "reasoning", "conclusion"],
            "domain_knowledge_corpus": ["text"],
        },
        "min_pt_text_chars": 80,
        "image_required_tasks": [
            "page_to_structured_description",
            "chart_table_to_text",
            "page_content_restatement",
            "cross_page_synthesis",
            "evidence_to_conclusion_chain",
        ],
        "multi_page_tasks": ["cross_page_synthesis"],
    },
    "prompt_style_rules": {},
    "prompts": PROMPTS,
}

CONFIG_JSON = json.dumps(CFG, ensure_ascii=False, indent=2)
