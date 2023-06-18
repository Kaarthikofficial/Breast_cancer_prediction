import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Breast cancer prediction", layout="wide")
st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Breast cancer prediction</h1>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))


@st.cache_resource
def scaled():
    return pickle.load(open('scale.pkl', 'rb'))


@st.cache_resource
def pca():
    return pickle.load(open('pca.pkl', 'rb'))


model = load_model()
scale = scaled()
pca_d = pca()

c1, c2, c3 = st.columns([5, 5, 5])
# Define a function to update the dataframe
def update_result():
    radius_mean_v = st.session_state.radius_mean_vl
    texture_mean_v = st.session_state.texture_mean_vl
    perimeter_mean_v = st.session_state.perimeter_mean_vl
    area_mean_v = st.session_state.area_mean_vl
    smoothness_mean_v = st.session_state.smoothness_mean_vl
    compactness_mean_v = st.session_state.compactness_mean_vl
    concavity_mean_v = st.session_state.concavity_mean_vl
    concave_points_mean_v = st.session_state.concave_points_mean_vl
    symmetry_mean_v = st.session_state.symmetry_mean_vl
    fractal_dimension_mean_v = st.session_state.fractal_dimension_mean_vl
    radius_se_v = st.session_state.radius_se_vl
    texture_se_v = st.session_state.texture_se_vl
    perimeter_se_v = st.session_state.perimeter_se_vl
    area_se_v = st.session_state.area_se_vl
    smoothness_se_v = st.session_state.smoothness_se_vl
    compactness_se_v = st.session_state.compactness_se_vl
    concavity_se_v = st.session_state.concavity_se_vl
    concave_points_se_v = st.session_state.concave_points_se_vl
    symmetry_se_v = st.session_state.symmetry_se_vl
    fractal_dimension_se_v = st.session_state.fractal_dimension_se_vl
    radius_worst_v = st.session_state.radius_worst_vl
    texture_worst_v = st.session_state.texture_worst_vl
    perimeter_worst_v = st.session_state.perimeter_worst_vl
    area_worst_v = st.session_state.area_worst_vl
    smoothness_worst_v = st.session_state.smoothness_worst_vl
    compactness_worst_v = st.session_state.compactness_worst_vl
    concavity_worst_v = st.session_state.concavity_worst_vl
    concave_points_worst_v = st.session_state.concave_points_worst_vl
    symmetry_worst_v = st.session_state.symmetry_worst_vl
    fractal_dimension_worst_v = st.session_state.fractal_dimension_worst_vl

    df = pd.DataFrame({
        'radius_mean': [radius_mean_v],
        'texture_mean': [texture_mean_v],
        'perimeter_mean': [perimeter_mean_v],
        'area_mean': [area_mean_v],
        'smoothness_mean': [smoothness_mean_v],
        'compactness_mean': [compactness_mean_v],
        'concavity_mean': [concavity_mean_v],
        'concave points_mean': [concave_points_mean_v],
        'symmetry_mean': [symmetry_mean_v],
        'fractal_dimension_mean': [fractal_dimension_mean_v],
        'radius_se': [radius_se_v],
        'texture_se': [texture_se_v],
        'perimeter_se': [perimeter_se_v],
        'area_se': [area_se_v],
        'smoothness_se': [smoothness_se_v],
        'compactness_se': [compactness_se_v],
        'concavity_se': [concavity_se_v],
        'concave points_se': [concave_points_se_v],
        'symmetry_se': [symmetry_se_v],
        'fractal_dimension_se': [fractal_dimension_se_v],
        'radius_worst': [radius_worst_v],
        'texture_worst': [texture_worst_v],
        'perimeter_worst': [perimeter_worst_v],
        'area_worst': [area_worst_v],
        'smoothness_worst': [smoothness_worst_v],
        'compactness_worst': [compactness_worst_v],
        'concavity_worst': [concavity_worst_v],
        'concave points_worst': [concave_points_worst_v],
        'symmetry_worst': [symmetry_worst_v],
        'fractal_dimension_worst': [fractal_dimension_worst_v]
    })

    # Scale the dataframe using the scaling parameters
    scaled_df = scale.transform(df)

    # Apply PCA to the scaled dataframe
    pca_df = pca_d.transform(scaled_df)
    print(pca_df)
    # Make the prediction
    prediction = model.predict(pca_df)
    probability = model.predict_proba(pca_df)[0][1] * 100

    print(prediction)
    print(probability)
    # Display the prediction and probability
    st.subheader('Results 📋')
    st.write('Prediction:', 'BREAST CANCER' if prediction == 1 else 'NO BREAST CANCER', '🙌')
    healthy_prob = 100 - probability
    cancer_prob = probability
    healthy = f"{healthy_prob:.2f}%"
    cancer = f"{cancer_prob:.2f}%"

    st.table(pd.DataFrame({'healthy': healthy, 'has breast cancer': cancer}, index=['probability']))


with c1:
    st.write("""
    <div style='text-align:center'>
        <h2 style='color:#009999;'>Mean</h2>
    </div>
    """, unsafe_allow_html=True)
    radius_mean = st.slider('Radius mean', min_value=5.0, max_value=40.0, value=6.2, step=0.01, key='radius_mean_s')
    st.session_state.radius_mean_vl = radius_mean
    texture_mean = st.slider('Texture mean', min_value=5.0, max_value=50.0, value=8.0, step=0.01, key='texture_mean_s')
    st.session_state.texture_mean_vl = texture_mean
    perimeter_mean = st.slider('Perimeter mean', min_value=40.0, max_value=200.0, value=50.0, step=0.1, key='perimeter_mean_s')
    st.session_state.perimeter_mean_vl = perimeter_mean
    area_mean = st.slider('Area mean', min_value=100.0, max_value=2600.0, value=150.0, step=0.1, key='area_mean_s')
    st.session_state.area_mean_vl = area_mean
    smoothness_mean = st.slider('Smoothness mean', min_value=0.0, max_value=0.20, value=4.0, step=0.01, key='smoothness_mean_s')
    st.session_state.smoothness_mean_vl = smoothness_mean
    compactness_mean = st.slider('Compactness mean', min_value=0.0, max_value=0.40, value=0.35, step=0.01, key='compactness_mean_s')
    st.session_state.compactness_mean_vl = compactness_mean
    concavity_mean = st.slider('Concavity mean', min_value=0.0, max_value=0.50, value=0.2, step=0.01, key='concavity_mean_s')
    st.session_state.concavity_mean_vl = concavity_mean
    concave_points_mean = st.slider('Concave points mean', min_value=0.0, max_value=0.3, value=0.25, step=0.01, key='concave_points_mean_s')
    st.session_state.concave_points_mean_vl = concave_points_mean
    symmetry_mean = st.slider('Symmetry mean', min_value=0.0, max_value=0.4, value=0.2, step=0.01, key='symmetry_mean_s')
    st.session_state.symmetry_mean_vl = symmetry_mean
    fractal_dimension_mean = st.slider('Fractal dimension mean', min_value=0.0, max_value=0.15, value=0.02,
                                       step=0.01, key='fractal_dimension_mean_s')
    st.session_state.fractal_dimension_mean_vl = fractal_dimension_mean
with c2:
    st.write("""
    <div style='text-align:center'>
        <h2 style='color:#009999;'>Standard Error</h2>
    </div>
    """, unsafe_allow_html=True)
    radius_se = st.slider('Radius se', min_value=0.0, max_value=4.0, value=1.5, step=0.01, key='radius_se_s')
    st.session_state.radius_se_vl = radius_se
    texture_se = st.slider('Texture se', min_value=0.0, max_value=6.0, value=4.2, step=0.01, key='texture_se_s')
    st.session_state.texture_se_vl = texture_se
    perimeter_se = st.slider('Perimeter se', min_value=0.0, max_value=30.0, value=11.0, step=0.01, key='perimeter_se_s')
    st.session_state.perimeter_se_vl = perimeter_se
    area_se = st.slider('Area se', min_value=5.0, max_value=70.0, value=36.0, step=0.01, key='area_se_s')
    st.session_state.area_se_vl = area_se
    smoothness_se = st.slider('Smoothness se', min_value=0.0, max_value=0.08, value=0.04, step=0.01, key='smoothness_se_s')
    st.session_state.smoothness_se_vl = smoothness_se
    compactness_se = st.slider('Compactness se', min_value=0.0, max_value=28.0, value=6.2, step=0.01, key='compactness_se_s')
    st.session_state.compactness_se_vl = compactness_se
    concavity_se = st.slider('Concavity se', min_value=0.0, max_value=0.2, value=0.07, step=0.01, key='concavity_se_s')
    st.session_state.concavity_se_vl = concavity_se
    concave_points_se = st.slider('Concave points se', min_value=0.0, max_value=0.09, value=0.02, step=0.01, key='concave_points_se_s')
    st.session_state.concave_points_se_vl = concave_points_se
    symmetry_se = st.slider('Symmetry se', min_value=0.0, max_value=0.1, value=0.05, step=0.01, key='symmetry_se_s')
    st.session_state.symmetry_se_vl = symmetry_se
    fractal_dimension_se = st.slider('Fractal dimension se', min_value=0.0, max_value=0.06, value=0.01, step=0.01, key='fractal_dimension_se_s')
    st.session_state.fractal_dimension_se_vl = fractal_dimension_se
with c3:
    st.write("""
        <div style='text-align:center'>
            <h2 style='color:#009999;'>worst</h2>
        </div>
        """, unsafe_allow_html=True)
    radius_worst = st.slider('Radius worst', min_value=6.0, max_value=50.0, value=28.0, step=0.1, key='radius_worst_s')
    st.session_state.radius_worst_vl = radius_worst
    texture_worst = st.slider('Texture worst', min_value=8.0, max_value=60.0, value=18.0, step=0.1, key='texture_worst_s')
    st.session_state.texture_worst_vl = texture_worst
    perimeter_worst = st.slider('Perimeter worst', min_value=40.0, max_value=280.0, value=185.0, step=0.1, key='perimeter_worst_s')
    st.session_state.perimeter_worst_vl = perimeter_worst
    area_worst = st.slider('Area worst', min_value=180.0, max_value=5000.0, value=1352.0, step=0.1, key='area_worst_s')
    st.session_state.area_worst_vl = area_worst
    smoothness_worst = st.slider('Smoothness worst', min_value=0.0, max_value=0.04, value=0.01, step=0.001, key='smoothness_worst_s')
    st.session_state.smoothness_worst_vl = smoothness_worst
    compactness_worst = st.slider('Compactness worst', min_value=0.0, max_value=2.0, value=0.045, step=0.01, key='compactness_worst_s')
    st.session_state.compactness_worst_vl = compactness_worst
    concavity_worst = st.slider('Concavity worst', min_value=0.0, max_value=2.0, value=0.069, step=0.01, key='concavity_worst_s')
    st.session_state.concavity_worst_vl = concavity_worst
    concave_points_worst = st.slider('Concave points worst', min_value=0.0, max_value=0.05, value=0.02, step=0.001, key='concave_points_worst_s')
    st.session_state.concave_points_worst_vl = concave_points_worst
    symmetry_worst = st.slider('Symmetry worst', min_value=0.0, max_value=0.1, value=0.08, step=0.001, key='symmetry_worst_s')
    st.session_state.symmetry_worst_vl = symmetry_worst
    fractal_dimension_worst = st.slider('Fractal dimension worst', min_value=0.0, max_value=0.04,
                                        value=0.02, step=0.001, key='fractal_dimension_worst_s')
    st.session_state.fractal_dimension_worst_vl = fractal_dimension_worst


if st.button('Predict'):
    update_result()