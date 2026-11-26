"""Streamlit dashboard for the road sign detector."""

from collections import Counter
from io import BytesIO
from pathlib import Path
from time import perf_counter

import pandas as pd
import streamlit as st
from PIL import UnidentifiedImageError

from inference import CLASS_LABELS, DEFAULT_CONFIDENCE, IMAGE_SIZE, detect, load_model, read_image

st.set_page_config(page_title="Розпізнавання дорожніх знаків", page_icon="🚦", layout="wide")
st.html(Path(__file__).resolve().parent / "assets" / "dashboard.css")


@st.cache_resource
def get_model():
    return load_model()


def show_comparison(image=None, annotated=None):
    with st.container(key="comparison"):
        st.subheader("Порівняння зображень")
        for column, key, title, photo, placeholder in zip(
            st.columns(2),
            ("original-photo", "annotated-photo"),
            ("Оригінал", "Результат розпізнавання"),
            (image, annotated),
            ("Завантажене фото з’явиться тут", "Тут з’явиться фото з позначеними об’єктами"),
        ):
            with column:
                with st.container(border=True, key=key):
                    st.markdown(f"#### {title}")
                    if photo is None:
                        st.html(f'<div class="image-placeholder"><span>{placeholder}</span></div>')
                    else:
                        st.image(photo, width="stretch")


with st.sidebar:
    st.html(
        '<div class="dashboard-brand">'
        '<span class="brand-mark" aria-hidden="true"></span>'
        '<span>Road Sign<br>Detection</span>'
        '</div>'
    )
    st.divider()
    st.header("Налаштування")
    confidence_percent = st.slider(
        "Поріг впевненості", 5, 95, round(DEFAULT_CONFIDENCE * 100), format="%d%%"
    )
    st.caption(f"Рекомендований поріг — {DEFAULT_CONFIDENCE:.0%}.")
    with st.container(border=True, key="sidebar-model"):
        st.markdown("**YOLOv8n**")
        st.caption(f"Розмір входу · {IMAGE_SIZE} × {IMAGE_SIZE} пкс")
    st.divider()
    st.subheader("Класи об’єктів")
    labels = "".join(f"<li>{label}</li>" for label in CLASS_LABELS.values())
    st.html(f'<ul class="class-list">{labels}</ul>')
    st.divider()
    st.caption("Навчальний прототип. Не призначено для керування транспортом.")

st.html("""
    <section class="dashboard-hero" aria-labelledby="dashboard-title">
        <div class="hero-copy">
            <p class="hero-category">Комп’ютерний зір</p>
            <h1 id="dashboard-title">Розпізнавання<br>дорожніх знаків</h1>
            <p>Завантажте фото дороги. YOLOv8n позначить знаки й світлофори
                та покаже впевненість для кожного об’єкта.</p>
        </div>
        <div class="hero-graphic" aria-hidden="true">
            <div class="scan-frame">
                <span class="scan-sign">STOP</span>
            </div>
        </div>
    </section>
""")

demo_tab, model_tab = st.tabs(["Розпізнавання", "Про модель"])

with demo_tab:
    with st.container(border=True, key="upload-panel"):
        st.subheader("Завантажте фото дороги")
        uploaded = st.file_uploader(
            "Фотографія дороги",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="road-photo",
        )
    if uploaded is None:
        show_comparison()
    else:
        try:
            if uploaded.size > 20 * 1024 * 1024:
                raise ValueError("Максимальний розмір файлу — 20 МБ.")
            image = read_image(uploaded)
            with st.spinner("Модель аналізує зображення…"):
                model = get_model()
                started = perf_counter()
                annotated, detections = detect(model, image, confidence_percent / 100)
                elapsed = (perf_counter() - started) * 1000
        except UnidentifiedImageError:
            st.error("Не вдалося відкрити фото. Спробуйте завантажити інший файл JPG, JPEG або PNG.")
        except (OSError, ValueError) as exc:
            st.error(f"Не вдалося обробити зображення або модель. {exc}")
        except Exception as exc:
            st.error("Не вдалося виконати розпізнавання. Перевірте модель і залежності.")
            st.caption(str(exc))
        else:
            with st.container(key="result-metrics"):
                stats = (
                    ("Виявлено об’єктів", len(detections)),
                    ("Розпізнано класів", len({item.name for item in detections})),
                    ("Час обробки", f"{elapsed:.0f} мс"),
                )
                for column, (label, value) in zip(st.columns(3), stats):
                    column.metric(label, value, border=True)

            show_comparison(image, annotated)
            if detections:
                counts = Counter(item.name for item in detections)
                with st.container(border=True, key="detection-details"):
                    st.subheader("Виявлені об’єкти")
                    st.dataframe(
                        [
                            {
                                "№": index,
                                "Клас": CLASS_LABELS[item.name],
                                "Впевненість": item.confidence * 100,
                                "Координати (x1, y1, x2, y2)": ", ".join(
                                    f"{value:.0f}" for value in item.xyxy
                                ),
                            }
                            for index, item in enumerate(detections, 1)
                        ],
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "Впевненість": st.column_config.ProgressColumn(
                                min_value=0, max_value=100, format="%.1f%%"
                            ),
                        },
                    )
                    st.subheader("Кількість об’єктів за класами")
                    st.bar_chart(
                        pd.DataFrame(
                            [(CLASS_LABELS[name], count) for name, count in counts.items()],
                            columns=["Клас", "Кількість"],
                        ),
                        x="Клас",
                        y="Кількість",
                        horizontal=True,
                        width="stretch",
                        color="#087f79",
                    )
            else:
                st.info("За обраного порогу об’єктів не виявлено.")

            output = BytesIO()
            annotated.save(output, format="PNG")
            st.download_button(
                "Завантажити результат · PNG",
                output.getvalue(),
                "road-sign-result.png",
                "image/png",
                type="primary",
                on_click="ignore",
            )

with model_tab:
    st.subheader("Якість моделі на тестовій вибірці")
    with st.container(key="model-metrics"):
        metrics = (
            ("Точність (Precision)", "97,6%"),
            ("Повнота (Recall)", "88,4%"),
            ("mAP@0.5", "90,2%"),
            ("mAP@0.5:0.95", "75,1%"),
        )
        for column, (label, value) in zip(st.columns(4), metrics):
            column.metric(label, value, border=True)

    with st.container(border=True, key="model-details"):
        st.markdown("### Архітектура YOLOv8n")
        st.write("YOLOv8n обрано після порівняння з YOLOv8s. Найскладніший клас — світлофори.")
        st.caption(
            f"Вхід {IMAGE_SIZE} × {IMAGE_SIZE} пкс · поріг за замовчуванням {DEFAULT_CONFIDENCE:.0%}"
        )
        st.markdown("### Датасет · 877 зображень")
        for column, (label, value) in zip(
            st.columns(3), (("Навчання", 613), ("Валідація", 176), ("Тестування", 88))
        ):
            column.metric(label, value)
