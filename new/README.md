# Market Signal AI Agent Platform

YouTube 评论区 Demand Signal 检测平台 — Streamlit Dashboard。

## 快速开始

### 1. 安装依赖

```bash
cd new
pip install -r requirements.txt
```

### 2. 配置 API Keys

在 Streamlit 侧边栏输入：

- **YouTube Data API Key** — 从 [Google Cloud Console](https://console.cloud.google.com/apis/credentials) 获取
- **DeepSeek API Key** — 从 [DeepSeek Platform](https://platform.deepseek.com/) 获取

### 3. 启动

```bash
cd new
streamlit run app.py
```

## 项目结构

```
new/
├── app.py                   # Streamlit 主应用（入口）
├── requirements.txt         # Python 依赖
├── README.md               # 本文件
│
├── src/                    # 核心逻辑模块（保持与原始 notebook 一致）
│   ├── __init__.py
│   ├── config.py            # 所有配置常量（分类、关键词、正则模式等）
│   ├── step1_keywords.py    # Step 1: 关键词生成
│   ├── step2_collect.py     # Step 2: YouTube 数据采集
│   ├── step3_merge.py       # Step 3: 数据合并
│   ├── step4_link.py        # Step 4: 评论-视频关联
│   ├── step5_clean.py       # Step 5: 数据清洗
│   ├── step6_demand_signal.py # Step 6: DeepSeek LLM 信号分类
│   └── pipeline.py          # 全流程一键运行
│
└── data/                   # 数据输出目录（自动创建）
    ├── linked_data/
    ├── cleaned_data/
    └── demand_signals/
        └── checkpoints/
```

## 功能模块

| 步骤 | 页面 | 说明 |
|------|------|------|
| Step 1 | 🔑 Keywords | 生成 18 个产品类别 × 4 种查询类型 = 864+ 搜索关键词 |
| Step 2 | 📡 Collection | YouTube Data API v3 采集视频 + 评论，支持断点续传 |
| Step 3 | 🔗 Merge | 合并所有采集文件，去重生成 master 表 |
| Step 4 | 📊 Link | 评论关联视频元数据，分类产品类别，提取视频上下文标签 |
| Step 5 | 🧹 Clean | 多级过滤：低互动视频、低价值评论、产品关键词筛选 |
| Step 6 | 🤖 LLM Signals | DeepSeek AI 分类 7 种需求信号，含置信度与理由 |
| Dashboard | 📈 Results | 可视化、信号分析、数据导出 |

## Demand Signal 分类标签

| 标签 | 说明 |
|------|------|
| `purchase_intent` | 明确表达购买意愿 |
| `problem_complaint` | 抱怨损坏或缺乏保护 |
| `comparison_research` | 正在比较或研究 |
| `usage_scenario` | 描述保护使用场景 |
| `wishful_thinking` | 希望已有保护措施 |
| `supply_recommendation` | 推荐或好评某款产品 |

## 数据流

```
关键词生成 (Step 1)
  ↓
YouTube API 采集视频+评论 (Step 2)
  ↓
数据合并去重 (Step 3)
  ↓
评论-视频关联 (Step 4) → linked_data/comments_video_linked.parquet
  ↓
数据清洗过滤 (Step 5) → cleaned_data/cleaned_comments_linked.parquet
  ↓
DeepSeek LLM 分类 (Step 6) → demand_signals/demand_signals_only_*.parquet
  ↓
结果可视化 (Dashboard)
```

## 注意事项

- **YouTube API 配额**：每个 API Key 每日有固定配额，平台会优雅地停止并保存进度
- **断点续传**：每个步骤的结果自动保存，重新运行会从断点继续
- **不修改原代码**：所有新文件生成在 `./new/` 目录，原有 notebook 代码不受影响
- **原始数据复用**：Step 3 支持加载原始 notebook 的 staging/master 数据目录

## API 费用估算

- **YouTube Data API**：免费配额 10,000 units/天
- **DeepSeek**：约 $0.001 / 100 条评论（batch=20，每批 $0.001）
