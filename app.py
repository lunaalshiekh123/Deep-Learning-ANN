import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import os

# 1. إعدادات الصفحة الأساسية
st.set_page_config(page_title="AI Deep Learning Studio", layout="wide")

# --- دالات النظام تظل كما هي ---
def load_dataset_builtin(name, normalize=True):
    name = name.lower()
    if name == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, -1); x_test = np.expand_dims(x_test, -1)
        num_classes = 10
    elif name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        num_classes = 10
    elif name == 'fashion-mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = np.expand_dims(x_train, -1); x_test = np.expand_dims(x_test, -1)
        num_classes = 10
    
    x_train = x_train.astype('float32')
    if normalize: x_train /= 255.0
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    return x_train, y_train, x_train.shape[1:], num_classes

def build_model(input_shape, num_classes, layers_config):
    model = models.Sequential()
    is_flattened = False
    for i, cfg in enumerate(layers_config):
        if cfg['type'] == 'Conv2D':
            if i == 0: model.add(layers.Conv2D(cfg['units'], (3,3), activation=cfg['act'], input_shape=input_shape))
            else: model.add(layers.Conv2D(cfg['units'], (3,3), activation=cfg['act']))
        elif cfg['type'] == 'MaxPool2D':
            model.add(layers.MaxPooling2D((2,2)))
        elif cfg['type'] == 'Dense':
            if not is_flattened:
                model.add(layers.Flatten())
                is_flattened = True
            model.add(layers.Dense(cfg['units'], activation=cfg['act']))
    if not is_flattened: model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# ==========================================
# 2. تصميم الواجهة (القائمة الجانبية العمودية)
# ==========================================

# --- القائمة الجانبية (Sidebar) ---
with st.sidebar:
    st.header("⚙️ إعدادات النموذج")
    dataset_name = st.selectbox("اختر مجموعة البيانات", ["MNIST", "CIFAR10", "Fashion-MNIST"])
    epochs = st.slider("عدد دورات التدريب", 1, 10, 3)
    batch_size = st.selectbox("حجم الدفعة", [32, 64, 128])
    
    st.divider()
    st.subheader("🛠️ هيكلية الشبكة")
    num_layers = st.number_input("عدد الطبقات", 1, 6, 3)
    
    configs = []
    for i in range(num_layers):
        with st.expander(f"الطبقة {i+1}", expanded=(i == 0)):
            l_type = st.selectbox("النوع", ["Conv2D", "MaxPool2D", "Dense"], key=f"t{i}")
            units = st.number_input("الوحدات", 8, 256, 32, key=f"u{i}")
            act = st.selectbox("التفعيل", ["relu", "tanh", "sigmoid"], key=f"a{i}")
            configs.append({'type': l_type, 'units': units, 'act': act})

# --- منطقة العمل الرئيسية (Main Area) ---
st.title("🧠 AI Deep Learning Studio")
st.caption("صمم شبكتك العصبية، درّبها، واختبرها في مكان واحد")

main_col1, main_col2 = st.columns([1.5, 1])

with main_col1:
    st.header("🚀 التدريب والتحليل")
if st.button("ابدأ التدريب الآن", use_container_width=True, type="primary"):
        with st.status("جاري بناء النموذج والتدريب...", expanded=True) as status:
            x_train, y_train, in_shape, n_classes = load_dataset_builtin(dataset_name)
            model = build_model(in_shape, n_classes, configs)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
            
            st.session_state['trained_model'] = model
            st.session_state['ds_name'] = dataset_name
            status.update(label="✅ اكتمل التدريب!", state="complete", expanded=False)
            
        st.subheader("📈 منحنى الدقة")
        st.line_chart(history.history['accuracy'])
        st.success(f"أعلى دقة تم الوصول إليها: {max(history.history['accuracy'])*100:.2f}%")

with main_col2:
    st.header("🔍 اختبار حي")
    uploaded_file = st.file_uploader("ارفع صورة للاختبار", type=["png", "jpg", "jpeg"])
    
    if uploaded_file and 'trained_model' in st.session_state:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="الصورة المرفوعة", use_column_width=True)
        
        if st.button("توقع النتيجة"):
            model = st.session_state['trained_model']
            img_arr = np.array(img)
            # معالجة الصورة حسب الداتا سيت
            if st.session_state['ds_name'] in ["MNIST", "Fashion-MNIST"]:
                img_t = tf.image.rgb_to_grayscale(img_arr)
                img_t = tf.image.resize(img_t, (28, 28))
            else:
                img_t = tf.image.resize(img_arr, (32, 32))
            
            img_t = img_t / 255.0
            pred = model.predict(np.expand_dims(img_t, 0))
            st.metric("الفئة المتوقعة", np.argmax(pred))
            st.write(f"نسبة الثقة: {np.max(pred)*100:.2f}%")