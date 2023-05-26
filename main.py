import streamlit as st

st.set_page_config(page_title="Breast cancer prediction", layout="wide")
st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Breast cancer prediction</h1>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

model = load_model()
c1, c2, c3 = st.columns([5, 5, 5])
with c1:
    st.write("""
<div style='text-align:center'>
    <h2 style='color:#009999;'>Mean</h2>
</div>
""", unsafe_allow_html=True)
    radius_mean = st.slider('Radius mean', min_value=5, max_value=40, value=6.2, step=0.01)
    texture_mean = st.slider('Texture mean', min_value=5, max_value=50, value=8, step=0.01)
    perimeter_mean = st.slider('Perimeter mean', min_value=40, max_value=200, value=50, step=0.1)
    area_mean = st.slider('Area mean', min_value=100, max_value=2600, value=150, step=0.1)
    smoothness_mean = st.slider('Smoothness mean', min_value=0, max_value=0.20, value=4, step=0.01)
    compactness_mean = st.slider('Compactness mean', min_value=0, max_value=0.40, value=0.35, step=0.01)
    concavity_mean = st.slider('Concavity mean', min_value=0, max_value=0.50, value=0.2, step=0.01)
    concave_points_mean = st.slider('Concave points mean', min_value=0, max_value=0.3, value=0.25, step=0.01)
    symmetry_mean = st.slider('Symmetry mean', min_value=0, max_value=0.4, value=0.2, step=0.01)
    fractal_dimension_mean = st.slider('Fractal dimension mean', min_value=0, max_value=0.15, value=0.02, step=0.01)
with c2:
    st.write("""
    <div style='text-align:center'>
        <h2 style='color:#009999;'>Standard Error</h2>
    </div>
    """, unsafe_allow_html=True)
    radius_se = st.slider('Radius se', min_value=0, max_value=4, value=1.5, step=0.01)
    texture_se = st.slider('Texture se', min_value=0, max_value=6, value=4.2, step=0.01)
    perimeter_se = st.slider('Perimeter se', min_value=0, max_value=30, value=11, step=0.01)
    area_se = st.slider('Area se', min_value=5, max_value=70, value=36, step=0.01)
    smoothness_se = st.slider('Smoothness se', min_value=0, max_value=0.08, value=0.04, step=0.01)
    compactness_se = st.slider('Compactness se', min_value=0, max_value=28, value=6.2, step=0.01)
    concavity_se = st.slider('Concavity se', min_value=0, max_value=0.2, value=0.07, step=0.01)
    concave_points_se = st.slider('Concave points se', min_value=0, max_value=0.09, value=0.02, step=0.01)
    symmetry_se = st.slider('Symmetry se', min_value=0, max_value=0.1, value=0.05, step=0.01)
    fractal_dimension_se = st.slider('Fractal dimension se', min_value=0, max_value=0.06, value=0.01, step=0.01)
with c3:
    st.write("""
        <div style='text-align:center'>
            <h2 style='color:#009999;'>worst</h2>
        </div>
        """, unsafe_allow_html=True)
    radius_worst = st.slider('Radius worst', min_value=6, max_value=50, value=28, step=0.1)
    texture_worst = st.slider('Texture worst', min_value=8, max_value=60, value=18, step=0.1)
    perimeter_worst = st.slider('Perimeter worst', min_value=40, max_value=280, value=185, step=0.1)
    area_worst = st.slider('Area worst', min_value=180, max_value=5000, value=1352, step=0.1)
    smoothness_worst = st.slider('Smoothness worst', min_value=0, max_value=0.04, value=0.01, step=0.001)
    compactness_worst = st.slider('Compactness worst', min_value=0, max_value=2, value=0.045, step=0.01)
    concavity_worst = st.slider('Concavity worst', min_value=0, max_value=2, value=0.069, step=0.01)
    concave_points_worst = st.slider('Concave points worst', min_value=0, max_value=0.05, value=0.02, step=0.001)
    symmetry_worst = st.slider('Symmetry worst', min_value=0, max_value=0.1, value=0.08, step=0.001)
    fractal_dimension_worst = st.slider('Fractal dimension worst', min_value=0, max_value=0.04, value=0.02, step=0.001)

dataframe = pd.DataFrame(
    {'radius_mean': radius_mean, 'texture_mean': texture_mean, 'perimeter_mean': perimeter_mean,
     'area_mean': area_mean, 'smoothness_mean': smoothness_mean, 'compactness_mean': compactness_mean,
     'concavity_mean': concavity_mean, 'concave_points_mean': concave_points_mean, 'symmetry_mean': symmetry_mean,
     'fractal_dimension_mean': fractal_dimension_mean, 'radius_se': radius_se, 'texture_se': texture_se,
     'perimeter_se': perimeter_se, 'area_se': area_se, 'smoothness_se': smoothness_se,
     'compactness_se': compactness_se, 'concavity_se': concavity_se, 'concave_points_se': concave_points_se,
     'symmetry_se': symmetry_se, 'fractal_dimension_se': fractal_dimension_se, 'radius_worst': radius_worst,
     'texture_worst': texture_worst, 'perimeter_worst': perimeter_worst, 'area_worst': area_worst,
     'smoothness_worst': smoothness_worst, 'compactness_worst': compactness_worst, 'concavity_worst': concavity_worst,
     'concave_points_worst': concave_points_worst, 'symmetry_worst': symmetry_worst,
     'fractal_dimension_worst': fractal_dimension_worst
     }, index=[0])


