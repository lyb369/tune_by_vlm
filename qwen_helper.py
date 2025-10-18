import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
    Qwen2_5_VLForConditionalGeneration,
)


class QwenLocal:
    def __init__(self, model_path: str, device: Optional[str] = None, max_new_tokens: int = 256):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._is_vl = False
        self._model = None
        self._tokenizer = None
        self._processor = None

    def load(self):
        if self._model is not None:
            return
        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        model_type = getattr(cfg, 'model_type', '') or ''
        if 'qwen2_5_vl' in model_type or 'vl' in model_type:
            # Vision-language model path
            self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map='auto',
                trust_remote_code=True,
            )
            self._is_vl = True
        else:
            # Text-only causal LM path
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map='auto',
                trust_remote_code=True,
            )
            self._is_vl = False

    def _generate_text(self, prompt: str) -> str:
        self.load()
        if self._is_vl:
            # Use VL chat template in text-only mode
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text_prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text_prompt], images=None, padding=True, return_tensors="pt")
            # Keep tensors on model device via .to(self._model.device) if available
            model_inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            outputs = self._model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
            input_len = model_inputs['input_ids'].shape[1]
            text = self._processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            return text
        else:
            inputs = self._tokenizer(prompt, return_tensors='pt')
            model_inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            outputs = self._model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            # For some tokenizers, generated text includes the prompt; trim if present
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text

    def suggest_params(self, history_summaries: List[str], obs_text: str) -> Tuple[float, float]:
        # Model should already be loaded externally
        if self._model is None:
            self.load()
        
        # 改进的系统指令，更清晰易懂
        sys_inst = (
            "你是一个图像去模糊算法的参数优化专家。\n\n"
            "算法背景：\n"
            "- 这是一个基于稀疏性的图像去模糊算法\n"
            "- 需要优化两个关键参数：fidelity（保真度）和sparsity（稀疏性）\n\n"
            "参数含义：\n"
            "- fidelity：控制算法对原始模糊图像的保真程度，值越大越保持原图特征，但可能保留更多噪声\n"
            "- sparsity：控制去噪和细节保持的平衡，值越大去噪越强，但可能过度平滑细节\n\n"
            "优化目标：\n"
            "- 提高PSNR（峰值信噪比）：数值越大越好\n"
            "- 提高SSIM（结构相似性）：数值越大越好\n"
            "- 平衡去噪和细节保持\n\n"
            "输出要求：\n"
            "请分析历史记录和当前结果，给出下一轮的建议参数。\n"
            "必须严格按照以下格式输出，不要添加任何其他文字：\n"
            "fidelity=数值, sparsity=数值\n\n"
            "示例输出：\n"
            "fidelity=120.5, sparsity=8.3\n"
        )
        
        # 构建更详细的提示词
        prompt = sys_inst + "\n历史记录:\n" + "\n".join(history_summaries[-5:]) + "\n\n当前结果:\n" + obs_text + "\n\n请给出下一轮参数建议:"
        
        print(f"发送给Qwen的提示词:\n{prompt}\n")
        
        try:
            out = self._generate_text(prompt)
            print(f"Qwen原始回复: {out}")
        except Exception as e:
            print(f"Qwen模型生成失败: {e}")
            return 150.0, 10.0  # 返回默认值
        
        # 提取参数，使用更宽松的匹配模式
        lam = 150.0  # 默认fidelity
        hes = 10.0   # 默认sparsity
        
        # 更宽松的数字匹配模式
        num_pat = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
        
        # 尝试多种匹配模式
        patterns = [
            rf"fidelity\s*[=:]\s*{num_pat}",
            rf"fidelity\s*{num_pat}",
            rf"fidelity\s*=\s*{num_pat}",
        ]
        
        for pattern in patterns:
            m1 = re.search(pattern, out, re.IGNORECASE)
            if m1:
                try:
                    lam = float(m1.group(1))
                    print(f"成功提取fidelity: {lam}")
                    break
                except ValueError:
                    continue
        
        patterns = [
            rf"sparsity\s*[=:]\s*{num_pat}",
            rf"sparsity\s*{num_pat}",
            rf"sparsity\s*=\s*{num_pat}",
        ]
        
        for pattern in patterns:
            m2 = re.search(pattern, out, re.IGNORECASE)
            if m2:
                try:
                    hes = float(m2.group(1))
                    print(f"成功提取sparsity: {hes}")
                    break
                except ValueError:
                    continue
        
        # 验证参数合理性
        if lam <= 0 or hes <= 0:
            print(f"警告: 提取的参数不合理 (fidelity={lam}, sparsity={hes})，使用默认值")
            return 150.0, 10.0
        
        print(f"最终提取参数: fidelity={lam}, sparsity={hes}")
        return lam, hes


def ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_memory(memory_dir: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.isdir(memory_dir):
        return records
    for fname in sorted(os.listdir(memory_dir)):
        if fname.endswith('.json'):
            with open(os.path.join(memory_dir, fname), 'r', encoding='utf-8') as f:
                try:
                    records.append(json.load(f))
                except Exception:
                    pass
    return records


def append_memory(memory_dir: str, iteration: int, data: Dict[str, Any]) -> None:
    ensure_dirs([memory_dir])
    path = os.path.join(memory_dir, f'iter_{iteration:03d}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_history_summaries(records: List[Dict[str, Any]]) -> List[str]:
    summaries: List[str] = []
    for r in records:
        it = r.get('iteration')
        lam = r.get('fidelity')
        hes = r.get('sparsity')
        psnr = r.get('metrics', {}).get('psnr')
        ssim = r.get('metrics', {}).get('ssim')
        mg = r.get('metrics', {}).get('mean_grad_mag')
        summaries.append(
            f"[iter {it}] fidelity={lam}, sparsity={hes}, PSNR={psnr}, SSIM={ssim}, mean_grad_mag={mg}"
        )
    return summaries


