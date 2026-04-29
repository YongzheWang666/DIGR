# auto_label_mfnet_optimized.py
# -*- coding: utf-8 -*-

import os
import json
import time
import base64
import re
from typing import Dict, Any, Optional, List


from openai import OpenAI

# =========================
# 1) 配置区
# =========================
ROOT = r"E:\my_fusion\dataset"   # dataset根目录
DATASET = "potsdam"
MODEL = "qwen3.5-plus"                 # 便宜+可看图
SLEEP_SEC = 0.15

# mfnet:train:3136,test:1572
# Potsdam: train:10830,test:2527
# whu:     train:17280，test:4320
TRAIN_N = 10830
TEST_N = 2527


# MFNet、Potsdam、whu vocab list（优先使用）
if DATASET == "mfnet":
    DATASET_VOCAB = [
        "car",
        "person",
        "bike",
        "curve",
        "car_stop",
        "guardrail",
        "color_cone",
        "bump",
        "background",
    ]
elif DATASET == "potsdam":
    DATASET_VOCAB = [
        "impervious_surface",
        "building",
        "low_vegetation",
        "tree",
        "car",
        "clutter_background",
    ]
elif DATASET == "whu":
    DATASET_VOCAB = [
        "farmland",
        "city",
        "village",
        "water",
        "forest",
        "road",
        "others",
    ]
else:
    raise ValueError(f"Unsupported DATASET: {DATASET}")

# 允许的“可见光结构/纹理”补充词表（不是强制，只是引导更稳定）
if DATASET == "mfnet":
    VISIBLE_BIAS_WORDS = [
        "road", "building", "tree", "sky", "sidewalk", "lane", "street", "wall",
        "sign", "light", "lamp", "pole", "fence", "window", "door", "vegetation"
    ]
elif DATASET == "potsdam":
    VISIBLE_BIAS_WORDS = [
        "building", "road", "tree", "vegetation", "shadow",
        "roof", "surface", "car", "background"
    ]
elif DATASET == "whu":
    VISIBLE_BIAS_WORDS = [
        "farmland", "city", "village", "water", "forest",
        "road", "vegetation", "background"
    ]

# 用于归一化的同义词映射（非常关键：稳定输出）
CANONICAL_MAP = {
    # mfnet / 通用目标
    "pedestrian": "person",
    "people": "person",
    "man": "person",
    "woman": "person",
    "human": "person",

    "vehicle": "car",
    "auto": "car",
    "automobile": "car",
    "truck": "car",
    "bus": "car",

    "bicycle": "bike",
    "cyclist": "bike",

    "trees": "tree",
    "buildings": "building",
    "roads": "road",

    # potsdam
    "impervious surface": "impervious_surface",
    "impervioussurface": "impervious_surface",
    "low vegetation": "low_vegetation",
    "clutter": "clutter_background",
    "background clutter": "clutter_background",

    # whu
    "farmlands": "farmland",
    "cities": "city",
    "villages": "village",
    "forests": "forest",
}
# =========================
# 2) 工具函数
# =========================
client = OpenAI(
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
) # 创建客户端

# 创建目录
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

#将图片转base64编码，然后解码成字符串，方便塞进JSON请求里
def b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 找数据集图片，返回图片地址
def find_image_file(folder: str, idx: int) -> Optional[str]:
    # 你数据是 1.png 这种
    for ext in ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]:
        p = os.path.join(folder, f"{idx}.{ext}")
        if os.path.exists(p):
            return p
    return None

# 读取JSON文件，如果存在返回解析后的数据；若不存在返回空字典
def safe_json_load(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 将Dict JSON标注写进path地址文件中
def safe_json_dump(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# 在模型输出里提取JSON
def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip() # 去掉字符串前后空白
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S) # re.search()作用是在字符串中寻找符合某种“模式”的子字符串
    if m:
        return json.loads(m.group(0))
    raise ValueError("Model output is not valid JSON.")

def _token_clean(s: str) -> str:
    # 只保留小写字母/数字/下划线/空格，避免奇怪符号
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_ ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

#归一化列表，去重，限制数量
def normalize_list(xs: Any, max_items: int = 6) -> List[str]:
    if xs is None:
        return []
    if isinstance(xs, str):
        # 如果模型返回 "a, b, c"
        xs = [x.strip() for x in xs.split(",") if x.strip()]
    if not isinstance(xs, list):
        return []

    out = []
    seen = set()
    for x in xs:
        if not isinstance(x, str):
            continue
        t = _token_clean(x)
        if not t:
            continue
        # 映射到规范词
        t = CANONICAL_MAP.get(t, t)
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max_items:
            break
    return out

# 生成最终prompt
def build_prompt_text(scene: str, ir_objs: List[str], vis_objs: List[str]) -> dict[str,str]:
    """
    生成两个独立的prompt：
    -infrared_prompt:给IR分支
    -visible_prompt:给VIS分支
    """
    ir_text = ", ".join(ir_objs) if ir_objs else "none"
    vis_text = ", ".join(vis_objs) if vis_objs else "none"

    infrared_prompt = f"{scene}. Infrared highlights:{ir_text}."
    visible_prompt = f"{scene}. Visible highlights:{vis_text}."
    return {
        "infrared_prompt": infrared_prompt,
        "visible_prompt":visible_prompt
    }

# 构造提示词模板
def build_instruction() -> str:
    vocab = ", ".join(DATASET_VOCAB)
    vis_bias = ", ".join(VISIBLE_BIAS_WORDS[:12])

    # 关键：把“热目标/结构纹理”的偏好写清楚，输出会稳很多
    return f"""
You label an RGB-visible image and an infrared-thermal image pair.

Return ONLY a valid JSON object with exactly these keys:
- "scene": one short English sentence (<= 18 words).
- "infrared_objects": 0-6 salient objects/cues more informative in infrared.
- "visible_objects": 0-6 salient objects/cues more informative in visible RGB.

Guidelines:
1) Prefer dataset class names when applicable: [{vocab}].
2) Infrared usually emphasizes salient targets and regions with strong thermal contrast.
3) Visible RGB usually emphasizes structure/texture/background (examples: {vis_bias}). Use such words if helpful.
4) If salient cues are not in the dataset vocabulary, you may add extra concise lowercase words.
5) Use lowercase words; no duplicates; no long phrases (<= 2 words per item).
6) Output JSON only. No markdown, no explanation.
""".strip()

# =========================
# 3) 调用 API：更严格的输出控制
# =========================
def label_pair(rgb_path: str, ir_path: str, max_retries: int = 3) -> Dict[str, Any]:
    instruction = build_instruction()
    rgb_b64 = b64_image(rgb_path)
    ir_b64 = b64_image(ir_path)

    last_err = None
    for attempt in range(1, max_retries + 1): #每一对图像最多尝试调用三次
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},

                            {"type": "text", "text": "Visible (RGB) image:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{rgb_b64}"}},

                            {"type": "text", "text": "Infrared (Thermal) image:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ir_b64}"}},
                        ],
                    }
                ],
                temperature=0,
            )
            text = resp.choices[0].message.content
            data = extract_json(text)

            scene = str(data.get("scene", "")).strip()
            scene = re.sub(r"\s+", " ", scene)

            ir_objs = normalize_list(data.get("infrared_objects"), max_items=6)
            vis_objs = normalize_list(data.get("visible_objects"), max_items=6)

            # 再做一个“小策略”：如果红外为空但可见光有 car/person/bike，就把它们拉回红外（更符合你的先验）
            if DATASET == "mfnet" and not ir_objs:
                for t in ["person", "car", "bike"]:
                    if t in vis_objs and t not in ir_objs:
                        ir_objs.append(t)
                ir_objs = ir_objs[:6]

            prompts = build_prompt_text(scene, ir_objs, vis_objs)

            return {
                "scene": scene,
                "infrared_objects": ir_objs,
                "visible_objects": vis_objs,
                "infrared_prompt": prompts["infrared_prompt"],
                "visible_prompt":prompts["visible_prompt"]
            }

        except Exception as e:
            last_err = e
            time.sleep(1.0 * attempt)

    raise RuntimeError(f"Failed after retries: {last_err}")

# =========================
# 4) 批量跑
# =========================
def run_split(split: str, n: int) -> None:
    if split == "train":
        rgb_dir = os.path.join(ROOT, split, DATASET, "rgb","input")
        ir_dir = os.path.join(ROOT, split, DATASET, "ir","input")
    elif split == "test":
        rgb_dir = os.path.join(ROOT, split, DATASET, "rgb")
        ir_dir = os.path.join(ROOT, split, DATASET, "ir")
    else:
        raise ValueError(f"Unsupported split: {split}")

    ann_dir = os.path.join(ROOT, split, DATASET, "annotations")
    ensure_dir(ann_dir)

    out_json = os.path.join(ann_dir, f"{split}.json")
    ann = safe_json_load(out_json)

    print(f"[{split}] loaded: {len(ann)} items -> {out_json}")

    missing = 0
    new_count = 0

    for idx in range(1, n + 1):
        key = str(idx)
        if key in ann: # 说明已经标注过了，跳过
            continue

        rgb = find_image_file(rgb_dir, idx)
        ir = find_image_file(ir_dir, idx)
        if not rgb or not ir:
            missing += 1
            print(f"[{split}] MISSING id={idx} rgb={rgb} ir={ir}")
            continue

        try:
            item = label_pair(rgb, ir)
            ann[key] = item
            new_count += 1

            if new_count % 20 == 0:
                safe_json_dump(ann, out_json)
                print(f"[{split}] saved {len(ann)} items...")

            print(f"[{split}] id={idx} OK | {item['infrared_prompt']}  {item['visible_prompt']}")

        except Exception as e:
            print(f"[{split}] id={idx} FAILED: {e}")

        time.sleep(SLEEP_SEC)

    safe_json_dump(ann, out_json)
    print(f"[{split}] DONE. total={len(ann)} missing={missing}")

if __name__ == "__main__":
    run_split("train", TRAIN_N)
    run_split("test", TEST_N)