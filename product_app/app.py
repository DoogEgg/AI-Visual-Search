from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from PIL import Image

from product_app.core.clip_encoder import CLIPEncoder
from product_app.core.config import (
    DEFAULT_MODEL_NAME,
    EMBEDDINGS_PATH,
    METADATA_PATH,
    ensure_product_directories,
)
from product_app.core.index_builder import build_photo_index
from product_app.core.vector_store import VectorStore


st.set_page_config(
    page_title="AI智能相册语义检索",
    page_icon="🔎",
    layout="wide",
)
ensure_product_directories()


@st.cache_resource(show_spinner=False)
def load_encoder(model_name: str) -> CLIPEncoder:
    return CLIPEncoder(model_name=model_name)


def load_saved_store() -> VectorStore | None:
    try:
        return VectorStore.load(EMBEDDINGS_PATH, METADATA_PATH)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        st.warning("本地索引损坏或格式不兼容，请重新建立索引。原始图片不会被修改。")
        return None


def get_store() -> VectorStore | None:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_saved_store()
    return st.session_state.vector_store


def render_results(results: list[dict], columns: int = 4) -> None:
    if not results:
        st.info("没有找到结果。请尝试更换描述或降低检索数量。")
        return

    grid = st.columns(columns)
    for index, result in enumerate(results):
        with grid[index % columns]:
            path = Path(result["path"])
            if path.exists():
                st.image(str(path), use_container_width=True)
                st.caption(f'{path.name} · 相似度 {result["score"]:.3f}')
            else:
                st.warning(f"文件已移动：{path.name}")


st.title("AI智能相册与语义检索平台")
st.caption("使用CLIP理解图片与文字语义，通过自然语言或参考图片检索本地相册。")

with st.sidebar:
    st.header("系统状态")
    model_name = st.text_input("CLIP模型", value=DEFAULT_MODEL_NAME)
    store = get_store()
    if store is None:
        st.warning("尚未建立图片索引")
    else:
        st.success(f"已索引 {len(store)} 张图片")
        st.write(f"检索后端：{store.backend}")
        st.caption(f"索引模型：{store.model_name or '未知（请重新建库）'}")
    st.info("当前默认CLIP对英文描述最稳定，例如：a person wearing red clothes。")
    st.caption("本地推理 · 图片与索引默认保存在本机 · 请勿公开个人相册")

import_tab, text_tab, image_tab, insight_tab, evaluation_tab = st.tabs(
    ["导入与建库", "自然语言搜图", "以图搜图", "图片洞察", "系统评测"]
)

with import_tab:
    st.subheader("导入本地照片并建立语义索引")
    default_photo_dir = "examples/images"
    photo_dir_text = st.text_input("照片文件夹路径", value=default_photo_dir)
    batch_size = st.slider("批处理大小", 1, 64, 16)
    st.write("程序会递归扫描子文件夹，跳过损坏图片，并通过SHA-256排除重复文件。")
    st.caption("路径属于运行程序的电脑。重新建库会替换当前索引，不会删除或移动原照片。")

    if st.button("开始建立索引", type="primary"):
        photo_root = Path(photo_dir_text).expanduser()
        if not photo_root.is_absolute():
            photo_root = PROJECT_ROOT / photo_root
        if not photo_root.exists() or not photo_root.is_dir():
            st.error("文件夹不存在，请检查路径。")
        else:
            progress_bar = st.progress(0.0)
            status = st.empty()

            def update_progress(message: str, value: float) -> None:
                status.write(message)
                progress_bar.progress(min(max(value, 0.0), 1.0))

            try:
                with st.spinner("首次使用可能需要加载CLIP模型，请稍候……"):
                    encoder = load_encoder(model_name)
                    new_store, stats = build_photo_index(
                        photo_root=photo_root,
                        encoder=encoder,
                        embeddings_path=EMBEDDINGS_PATH,
                        metadata_path=METADATA_PATH,
                        batch_size=batch_size,
                        progress=update_progress,
                    )
                st.session_state.vector_store = new_store
                st.success(
                    f'索引完成：{stats["indexed"]} 张有效图片，'
                    f'{stats["rejected"]} 张重复或无效图片，后端为 {stats["backend"]}。'
                )
                if stats["rejected_items"]:
                    with st.expander("查看跳过的文件"):
                        st.dataframe(pd.DataFrame(stats["rejected_items"]), use_container_width=True)
            except Exception as exc:
                st.exception(exc)

with text_tab:
    st.subheader("使用自然语言搜索图片")
    query = st.text_input(
        "描述你想找的图片",
        placeholder="例如：a person wearing red clothes / sunset by the sea",
    )
    top_k = st.slider("返回图片数量", 1, 40, 12, key="text_top_k")
    if st.button("搜索图片", type="primary"):
        store = get_store()
        if store is None:
            st.warning("请先在“导入与建库”页面建立图片索引。")
        elif not query.strip():
            st.warning("请输入搜索描述。")
        else:
            try:
                store.validate_model(model_name)
                encoder = load_encoder(model_name)
                query_vector = encoder.encode_texts([query])[0]
                results = store.search(query_vector, top_k=top_k)
                render_results(results)
            except Exception as exc:
                st.exception(exc)

with image_tab:
    st.subheader("上传参考图片，查找相似照片")
    uploaded_image = st.file_uploader(
        "上传一张参考图片",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
    )
    image_top_k = st.slider("返回图片数量", 1, 40, 12, key="image_top_k")
    if uploaded_image is not None:
        try:
            with Image.open(uploaded_image) as source:
                reference_image = source.convert("RGB")
        except (OSError, ValueError):
            st.warning("参考图片无法读取，请上传有效图片。")
        else:
            try:
                st.image(reference_image, caption="参考图片", width=320)
                if st.button("查找相似图片", type="primary"):
                    store = get_store()
                    if store is None:
                        st.warning("请先建立图片索引。")
                    else:
                        try:
                            store.validate_model(model_name)
                            encoder = load_encoder(model_name)
                            query_vector = encoder.encode_images([reference_image])[0]
                            render_results(store.search(query_vector, top_k=image_top_k))
                        except Exception as exc:
                            st.exception(exc)
            finally:
                reference_image.close()

with insight_tab:
    st.subheader("图片语义标签分析")
    store = get_store()
    if store is None or not len(store):
        st.info("建立图片索引后，可在这里查看单张图片的语义标签。")
    else:
        selected_index = st.selectbox(
            "选择图片",
            range(len(store.metadata)),
            format_func=lambda index: store.metadata[index]["name"],
        )
        selected = store.metadata[selected_index]
        selected_path = Path(selected["path"])
        if selected_path.exists():
            left, right = st.columns([1, 1])
            with left:
                st.image(str(selected_path), use_container_width=True)
            with right:
                st.json(
                    {
                        "文件名": selected["name"],
                        "分辨率": f'{selected["width"]} × {selected["height"]}',
                        "大小(MB)": round(selected["size_bytes"] / 1024 / 1024, 3),
                        "路径": (
                            str(selected_path.relative_to(PROJECT_ROOT))
                            if selected_path.is_relative_to(PROJECT_ROOT)
                            else "本地文件（路径已隐藏）"
                        ),
                    }
                )

            candidate_text = st.text_area(
                "候选语义标签（每行一个英文标签）",
                value="a person\nan animal\nfood\na vehicle\nnature\na building\na screenshot\na document\nred clothes\na group photo",
            )
            if st.button("分析语义标签"):
                labels = [line.strip() for line in candidate_text.splitlines() if line.strip()]
                if labels:
                    try:
                        store.validate_model(model_name)
                        encoder = load_encoder(model_name)
                        image_vector = store.embeddings[selected_index]
                        text_vectors = encoder.encode_texts(labels)
                        scores = text_vectors @ image_vector
                        order = scores.argsort()[::-1]
                        chart_data = pd.DataFrame(
                            {
                                "标签": [labels[index] for index in order],
                                "语义相似度": [float(scores[index]) for index in order],
                            }
                        )
                        st.dataframe(chart_data, use_container_width=True, hide_index=True)
                        st.bar_chart(chart_data.set_index("标签"))
                        st.caption("分数是候选标签与图片的余弦相似度，不是识别准确率或概率。")
                    except Exception as exc:
                        st.exception(exc)

with evaluation_tab:
    st.subheader("索引与模型状态")
    store = get_store()
    if store is None:
        st.info("尚无可评测的图片索引。")
    else:
        sizes = [item["size_bytes"] for item in store.metadata]
        st.metric("已索引图片", len(store))
        st.metric("向量维度", store.embeddings.shape[1])
        st.metric("图片总大小", f"{sum(sizes) / 1024 / 1024:.1f} MB")
        st.write(f"检索后端：{store.backend}")
        st.write(f"索引模型：{store.model_name or '未知'}")
        st.info(
            "下一阶段将加入带标准答案的查询集，并计算 Recall@5、Recall@10、mAP 和平均响应时间。"
        )
