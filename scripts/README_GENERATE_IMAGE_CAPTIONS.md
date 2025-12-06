**说明**: 本脚本用于把 Markdown 中的图片翻译（Captioning）为可检索的中文文本描述，便于上传到 Nexent 知识库并进行向量化检索。

- **脚本路径**: `scripts/generate_image_captions.py`
- **备份策略**: 修改文件前会生成同名 `.bak` 备份（如果不存在）。

**快速开始**

1. 安装依赖（可选按需安装）:

```powershell
cd `D:\LLMagent\knowledge_data`
python -m pip install -r requirements.txt
```

2. （可选）安装 Tesseract OCR（Windows）并确保 `tesseract.exe` 在 PATH 中。
   下载地址: https://github.com/tesseract-ocr/tesseract

3. （可选）设置 Qwen API Key（如果你有 Nexent/Qwen VLM 的访问）:

```powershell
setx QWEN_API_KEY "你的_api_key"
# 重新打开终端以使环境变量生效
```

4. 运行脚本（先做一次 dry-run 查看将要修改的文件）:

```powershell
python scripts/generate_image_captions.py --root D:\LLMagent\knowledge_data --dry-run
```

5. 真正写回修改：

```powershell
python scripts/generate_image_captions.py --root D:\LLMagent\knowledge_data
```

**可选参数**
- `--no-api`: 禁用远程 API，强制使用本地模型或仅 OCR
- `--force`: 即使已有描述也覆盖
- `--dry-run`: 不写回，仅显示将要修改的文件

**如何与 Nexent 平台配合**
- 运行脚本完成图片描述后，将更新的 Markdown 上传到 Nexent 的文档上传功能，或将这些文件作为知识库外挂导入。向量化阶段请使用描述文本与 OCR 文本作为图片对应的 metadata/embedding input，提高检索命中率与时效性。

**注意事项**
- 远程 VLM API 的 endpoint 与返回字段在不同厂商之间不同，`generate_image_captions.py` 中 `call_qwen_vlm_api` 是占位实现，请根据你的实际 API 文档修改 `endpoint` 与请求/响应解析逻辑。
- 如果处理大量图片，建议先用 `--dry-run` 检查，然后并行化处理或按目录分批处理。
