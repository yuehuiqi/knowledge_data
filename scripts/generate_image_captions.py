#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成并插入图片描述到 Markdown 文件的脚本。
功能：
- 遍历指定目录下所有 `.md` 文件
- 识别 Markdown 图片语法 `![alt](path)` 和 HTML `<img src="...">`
- 对图片文件：进行 OCR（pytesseract）+ 可选的 VLM 调用（Qwen API）/本地 BLIP 模型回退
- 在图片引用下插入中文描述（保留原有链接），并备份原 md 为 `.bak`

使用：
    python scripts/generate_image_captions.py --root d:\\LLMagent\\knowledge_data

环境变量（可选）：
- QWEN_API_KEY: 如果设置，将尝试调用 Qwen VLM API（示例请求体，需按实际 API 文档调整）

注意：如果没有 Qwen API 或本地模型，脚本至少会尝试用 OCR 提取图片中的文字，并把 OCR 结果放入描述中。

"""

import re
import os
import sys
import argparse
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import base64
import concurrent.futures
import time

try:
    from PIL import Image
except Exception:
    Image = None

# Optional heavy deps
try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import requests
except Exception:
    requests = None

# local model optional
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
except Exception:
    BlipProcessor = None
    BlipForConditionalGeneration = None
    torch = None

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

IMG_MD_REGEX = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
IMG_HTML_REGEX = re.compile(r'<img[^>]*src=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)

# 标记描述的前缀，避免重复插入
CAPTION_MARKER_PREFIX = '**[图片详情：'
CAPTION_MARKER_SUFFIX = ']**'


def find_md_files(root: Path):
    return list(root.rglob('*.md'))


def resolve_image_path(md_path: Path, image_ref: str) -> Optional[Path]:
    # image_ref might be URL or relative path; only handle local files
    if image_ref.startswith('http://') or image_ref.startswith('https://'):
        return None
    # strip optional title after space like (path "title")
    image_ref = image_ref.split()[0]
    # remove surrounding < > if present
    image_ref = image_ref.strip('<>')
    candidate = (md_path.parent / image_ref).resolve()
    if candidate.exists():
        return candidate
    # try to handle absolute-like paths from workspace root
    if image_ref.startswith('/'):
        cand2 = (Path(os.getcwd()) / image_ref.lstrip('/')).resolve()
        if cand2.exists():
            return cand2
    return None


def already_has_caption(md_lines, insert_line_index):
    # look at next non-empty line after the image and see if it starts with our marker
    i = insert_line_index + 1
    while i < len(md_lines) and md_lines[i].strip() == '':
        i += 1
    if i < len(md_lines):
        return md_lines[i].strip().startswith(CAPTION_MARKER_PREFIX)
    return False


def ocr_image(image_path: Path) -> str:
    if pytesseract is None or Image is None:
        return ''
    try:
        img = Image.open(str(image_path))
        text = pytesseract.image_to_string(img, lang='chi_sim+eng')
        return text.strip()
    except Exception as e:
        logging.warning(f'OCR 失败: {image_path} -> {e}')
        return ''


def call_qwen_vlm_api(image_path: Path, ocr_text: str = '') -> Optional[str]:
    """
    示例调用 Qwen VLM 的占位函数。具体 API 路径和请求体需按 Qwen 的文档调整。
    要使用请设置环境变量 `QWEN_API_KEY`。
    """
    # 该函数已被新实现替代。保留以兼容历史调用。
    return None


def call_dashscope_vlm_api(image_path: Path, ocr_text: str = '', api_key: Optional[str] = None, base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1', model: str = 'qwen3-vl-plus') -> Optional[str]:
    """
    使用阿里云 DashScope 的 OpenAI 兼容接口调用 qwen3-vl-plus 等 VLM 模型。
    我们把本地图片编码为 data URI（base64）并放入 message 的 image_url 字段中。

    要么传入 api_key，要么在环境变量 `DASHSCOPE_API_KEY` 中配置。
    """
    try:
        from openai import OpenAI
    except Exception:
        logging.warning('缺少 openai (DashScope) SDK：请安装 `pip install openai`)')
        return None

    api_key = api_key or os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        logging.warning('未提供 DASHSCOPE_API_KEY，跳过远程 VLM 调用')
        return None

    # read image and make data url
    try:
        with open(str(image_path), 'rb') as f:
            b = f.read()
        mime = 'image/jpeg'
        suffix = image_path.suffix.lower()
        if suffix in ('.png', '.gif', '.bmp', '.webp'):
            if suffix == '.png':
                mime = 'image/png'
            elif suffix == '.gif':
                mime = 'image/gif'
            elif suffix == '.bmp':
                mime = 'image/bmp'
            elif suffix == '.webp':
                mime = 'image/webp'
        data_b64 = base64.b64encode(b).decode('ascii')
        data_url = f'data:{mime};base64,{data_b64}'
    except Exception as e:
        logging.warning(f'读取或编码图片失败: {image_path} -> {e}')
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt_text = (
        '你是一个病理专家。请详细描述这张病理图片。重点关注：细胞形态、排列结构、染色特点、是否有核分裂象或坏死等。'
        '如果图片包含文字（如各种图表），请把文字 OCR 出来并标注。'
    )

    # 构造 messages：使用 image_url data URI 与文本提示
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': data_url
                    }
                },
                {
                    'type': 'text',
                    'text': prompt_text
                }
            ]
        }
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )
        # completion.choices[0].message.content 可能是字符串或结构
        first = completion.choices[0].message.content
        if isinstance(first, str):
            return first
        # 有些兼容实现返回 list of dict
        if isinstance(first, list):
            # join text parts
            parts = []
            for p in first:
                if isinstance(p, dict) and 'type' in p and p['type'] == 'text' and 'text' in p:
                    parts.append(p['text'])
                elif isinstance(p, str):
                    parts.append(p)
            return '\n'.join(parts).strip()
        return str(first)
    except Exception as e:
        logging.warning(f'调用 DashScope VLM 失败: {e}')
        return None


def local_blip_caption(image_path: Path) -> Optional[str]:
    if BlipProcessor is None or BlipForConditionalGeneration is None or torch is None:
        return None
    try:
        processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        raw_image = Image.open(str(image_path)).convert('RGB')
        inputs = processor(raw_image, return_tensors='pt').to(device)
        out = model.generate(**inputs, max_length=256)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        logging.warning(f'本地 BLIP 推理失败: {e}')
        return None


def generate_caption(image_path: Path, prefer_api: bool = True, api_key: Optional[str] = None, base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1', model: str = 'qwen3-vl-plus') -> str:
    # 先 OCR
    ocr_text = ocr_image(image_path)
    parts: List[str] = []
    if ocr_text:
        parts.append('OCR 识别文字：' + ocr_text)

    caption = None
    if prefer_api:
        caption = call_dashscope_vlm_api(image_path, ocr_text=ocr_text, api_key=api_key, base_url=base_url, model=model)
    if not caption:
        caption = local_blip_caption(image_path)
    if not caption:
        if ocr_text:
            caption = '仅有 OCR 文本，未生成自然语言描述。'
        else:
            caption = '未能自动生成描述（请手动补充）。'
    parts.insert(0, caption)
    return '\n'.join([p for p in parts if p])


def process_md_file(md_path: Path, dry_run: bool, prefer_api: bool, force: bool, api_key: Optional[str], base_url: str, model: str, executor: concurrent.futures.Executor, concurrency: int):
    text = md_path.read_text(encoding='utf-8')
    lines = text.splitlines()
    # collect image occurrences in this file
    occurrences: List[Tuple[int, str, Path]] = []  # (line_index, ref, image_path)
    for idx, line in enumerate(lines):
        for m in IMG_MD_REGEX.finditer(line):
            alt, ref = m.groups()
            img_path = resolve_image_path(md_path, ref)
            if img_path is None:
                logging.info(f'无法解析图片路径: {ref} in {md_path}')
                continue
            if not force and already_has_caption(lines, idx):
                logging.info(f'已存在描述，跳过：{md_path} -> {ref}')
                continue
            occurrences.append((idx, ref, img_path))
        for m in IMG_HTML_REGEX.finditer(line):
            ref = m.group(1)
            img_path = resolve_image_path(md_path, ref)
            if img_path is None:
                logging.info(f'无法解析图片路径: {ref} in {md_path}')
                continue
            if not force and already_has_caption(lines, idx):
                logging.info(f'已存在描述，跳过：{md_path} -> {ref}')
                continue
            occurrences.append((idx, ref, img_path))

    if not occurrences:
        logging.info(f'未检测到需要修改的图片：{md_path}')
        return

    # generate captions concurrently for occurrences
    futures: Dict[int, concurrent.futures.Future] = {}
    for i, (_idx, _ref, img_path) in enumerate(occurrences):
        futures[i] = executor.submit(generate_caption, img_path, prefer_api, api_key, base_url, model)

    results: Dict[int, str] = {}
    for i, fut in futures.items():
        try:
            results[i] = fut.result()
        except Exception as e:
            logging.error(f'生成描述失败: {e}')
            results[i] = '未能生成描述（出错）。'

    # rebuild file with inserted captions — iterate lines and insert captions after corresponding images
    new_lines: List[str] = []
    i = 0
    occ_map: Dict[int, List[Tuple[int, str, Path]]] = {}
    for idx, ref, path in occurrences:
        occ_map.setdefault(idx, []).append((idx, ref, path))

    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        if i in occ_map:
            for j, (idx, ref, img_path) in enumerate(occ_map[i]):
                # find occurrence index
                occ_index = None
                for k, occ in enumerate(occurrences):
                    if occ[0] == idx and occ[2] == img_path and occ[1] == ref:
                        occ_index = k
                        break
                caption = results.get(occ_index, '未能生成描述（内部映射失败）。')
                caption_md = f"{CAPTION_MARKER_PREFIX}{caption}{CAPTION_MARKER_SUFFIX}"
                new_lines.append('')
                new_lines.append(caption_md)
        i += 1

    modified = True
    backup = md_path.with_suffix(md_path.suffix + '.bak')
    if not dry_run:
        if not backup.exists():
            shutil.copy2(md_path, backup)
        md_path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
        logging.info(f'已更新并备份：`{md_path}` -> 备份 `{backup}`')
    else:
        logging.info(f'dry-run: 文件 `{md_path}` 将被修改（但未写入）')


def insert_caption_to_md(md_path: Path, dry_run: bool = False, prefer_api: bool = True, force: bool = False):
    text = md_path.read_text(encoding='utf-8')
    lines = text.splitlines()
    modified = False
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        # check markdown image
        for m in IMG_MD_REGEX.finditer(line):
            alt, ref = m.groups()
            img_path = resolve_image_path(md_path, ref)
            if img_path is None:
                logging.info(f'无法解析图片路径: {ref} in {md_path}')
                continue
            # check next lines for existing caption
            if not force and already_has_caption(lines, len(new_lines)-1):
                logging.info(f'已存在描述，跳过：{md_path} -> {ref}')
                continue
            logging.info(f'生成描述：{md_path} -> {ref}')
            caption = generate_caption(img_path, prefer_api=prefer_api)
            caption_md = f"{CAPTION_MARKER_PREFIX}{caption}{CAPTION_MARKER_SUFFIX}"
            # insert a blank line then caption
            new_lines.append('')
            new_lines.append(caption_md)
            modified = True
        # check html img tags in the line
        for m in IMG_HTML_REGEX.finditer(line):
            ref = m.group(1)
            img_path = resolve_image_path(md_path, ref)
            if img_path is None:
                logging.info(f'无法解析图片路径: {ref} in {md_path}')
                continue
            if not force and already_has_caption(lines, len(new_lines)-1):
                logging.info(f'已存在描述，跳过：{md_path} -> {ref}')
                continue
            logging.info(f'生成描述：{md_path} -> {ref}')
            caption = generate_caption(img_path, prefer_api=prefer_api)
            caption_md = f"{CAPTION_MARKER_PREFIX}{caption}{CAPTION_MARKER_SUFFIX}"
            new_lines.append('')
            new_lines.append(caption_md)
            modified = True
        i += 1

    if modified:
        backup = md_path.with_suffix(md_path.suffix + '.bak')
        if not dry_run:
            if not backup.exists():
                shutil.copy2(md_path, backup)
            md_path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
            logging.info(f'已更新并备份：`{md_path}` -> 备份 `{backup}`')
        else:
            logging.info(f'dry-run: 文件 `{md_path}` 将被修改（但未写入）')
    else:
        logging.info(f'未检测到需要修改的图片：{md_path}')


def main():
    parser = argparse.ArgumentParser(description='为 Markdown 中的图片生成中文描述并插入文档')
    parser.add_argument('--root', '-r', type=str, default='.', help='要处理的根目录')
    parser.add_argument('--dry-run', action='store_true', help='仅显示将要修改的文件，不写回')
    parser.add_argument('--no-api', action='store_true', help='禁用远程 API，强制使用本地模型或 OCR')
    parser.add_argument('--force', action='store_true', help='即使已有描述也覆盖')
    parser.add_argument('--api-key', type=str, default=None, help='DashScope API Key，若未提供则使用环境变量 DASHSCOPE_API_KEY')
    parser.add_argument('--base-url', type=str, default='https://dashscope.aliyuncs.com/compatible-mode/v1', help='DashScope base_url')
    parser.add_argument('--model', type=str, default='qwen3-vl-plus', help='模型名称，例如 qwen3-vl-plus')
    parser.add_argument('--concurrency', type=int, default=4, help='并发生成图片描述的线程数')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    md_files = find_md_files(root)
    logging.info(f'在 `{root}` 下找到 {len(md_files)} 个 md 文件')

    # 使用 ThreadPoolExecutor 处理图片生成（并发）
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        for md in md_files:
            try:
                process_md_file(md, dry_run=args.dry_run, prefer_api=not args.no_api, force=args.force, api_key=args.api_key, base_url=args.base_url, model=args.model, executor=executor, concurrency=args.concurrency)
            except Exception as e:
                logging.error(f'处理 `{md}` 时出错: {e}')


if __name__ == '__main__':
    main()
