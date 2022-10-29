import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import statsmodels.api as sm
import numpy as np
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from scipy.stats import zscore

df = pd.read_csv('data/PCOS_data.csv')
#df = pd.read_csv('/Users/almgacis/Documents/MSU/CMSE_830/HW/data/PCOS_data.csv')

#Replacing nans with median
df['Marriage Status (Yrs)'] = df['Marriage Status (Yrs)'].fillna(df['Marriage Status (Yrs)'].median())
df['II    beta-HCG(mIU/mL)'] = df['II    beta-HCG(mIU/mL)'].fillna(df['II    beta-HCG(mIU/mL)'].median())
df['AMH(ng/mL)'] = df['AMH(ng/mL)'].fillna(df['AMH(ng/mL)'].median())
df['Fast food (Y/N)'] = df['Fast food (Y/N)'].fillna(df['Fast food (Y/N)'].median())

#Dropping na, if any more
df2 = df.dropna()

#Averaging almost similar features
df2['Avg Follicle No'] = round((df2['Follicle No. (L)'] + df2['Follicle No. (R)'])/2)
df2['Avg F size (mm)'] = round((df2['Avg. F size (L) (mm)'] + df2['Avg. F size (R) (mm)'])/2)


st.sidebar.title("About the Data")
st.sidebar.markdown("Polycystic ovary syndrome (PCOS) is a condition involving irregular, missed, or prolonged periods, and most of the time, excess androgen levels. The ovaries develop follicles (small collections of fluid), and may fail to release eggs on a regular basis. The dataset contains physical and clinical variables that might help with determining PCOS diagnosis and infertility related issues. The data was collected from 10 hospitals across Kerala, India.")
st.sidebar.markdown('*The dataset entitled Polycystic ovary syndrome (PCOS) was made by Prasoon Kottarathil in 2020 and was published in [Kaggle](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos).*')

#-----Cat and Num--------------------------------------------------------------------------------------

st.title("""# Polycystic Ovarian Syndrome (PCOS) Diagnosis Data""")

measurements = df2.drop(labels=["PCOS (Y/N)"], axis=1).columns.tolist()

#Removing redundant features
df3 = df2.drop(['Sl. No', 'Patient File No.', 'Hip(inch)', 'Waist(inch)', 'BMI',
               'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)',
               'Avg. F size (R) (mm)', 'FSH/LH', 'II    beta-HCG(mIU/mL)'], axis=1)

df3_corr = df2.drop(['Sl. No', 'Patient File No.', 'Hip(inch)', 'Waist(inch)', 'BMI',
               'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)',
               'Avg. F size (R) (mm)', 'FSH/LH', 'II    beta-HCG(mIU/mL)',
               'Weight (Kg)', 'Marriage Status (Yrs)', 'Fast food (Y/N)'], axis=1)

df3copy = df3.copy()
categorical = df3copy.drop(labels=[' Age (yrs)',
 'Weight (Kg)',
 'Height(Cm) ',
 'Pulse rate(bpm) ',
 'RR (breaths/min)',
 'Hb(g/dl)',
 'Cycle(R/I)',
 'Cycle length(days)',
 'Marriage Status (Yrs)',
 'No. of abortions',
 '  I   beta-HCG(mIU/mL)',
 'FSH(mIU/mL)',
 'LH(mIU/mL)',
 'Waist:Hip Ratio',
 'TSH (mIU/L)',
 'AMH(ng/mL)',
 'PRL(ng/mL)',
 'Vit D3 (ng/mL)',
 'PRG(ng/mL)',
 'RBS(mg/dl)',
 'BP _Systolic (mmHg)',
 'BP _Diastolic (mmHg)',
 'Endometrium (mm)',
 'Avg Follicle No', 'Avg F size (mm)'], axis=1).columns.tolist()

for i in categorical:
    df3copy[i].replace([1, 0], ['Yes', 'No'], inplace=True)
    df3copy[i].replace([11, 12, 13, 14, 15, 16, 17, 18], ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB', 'AB-'], inplace=True)

cat_for_corr = df3.drop(labels=[' Age (yrs)',
 'Weight (Kg)',
 'Height(Cm) ',
 'Pulse rate(bpm) ',
 'RR (breaths/min)',
 'Hb(g/dl)',
 'Cycle(R/I)',
 'Cycle length(days)',
 'Marriage Status (Yrs)',
 'No. of abortions',
 '  I   beta-HCG(mIU/mL)',
 'FSH(mIU/mL)',
 'LH(mIU/mL)',
 'Waist:Hip Ratio',
 'TSH (mIU/L)',
 'AMH(ng/mL)',
 'PRL(ng/mL)',
 'Vit D3 (ng/mL)',
 'PRG(ng/mL)',
 'RBS(mg/dl)',
 'BP _Systolic (mmHg)',
 'BP _Diastolic (mmHg)',
 'Endometrium (mm)',
 'Avg Follicle No', 'Avg F size (mm)'], axis=1).columns.tolist()

numerical = df3.drop(labels=['PCOS (Y/N)',
 'Blood Group',
 'Pregnant(Y/N)',
 'Weight gain(Y/N)',
 'hair growth(Y/N)',
 'Skin darkening (Y/N)',
 'Hair loss(Y/N)',
 'Pimples(Y/N)',
 'Fast food (Y/N)',
 'Reg.Exercise(Y/N)'], axis=1).columns.tolist()

#Standardized with nominal vars for plots and standardized with binary vars for corr
df3z_nom = pd.concat([df3copy[categorical], df3[numerical].apply(zscore)], axis=1)
df3z_cor = pd.concat([df3[cat_for_corr], df3[numerical].apply(zscore)], axis=1).drop(['Weight (Kg)', 'Marriage Status (Yrs)', 'Fast food (Y/N)'], axis=1)


#Raw with removed outliers (for plots and corr)
np.random.seed(33454)

Q1 = df3[numerical].quantile(0.25)
Q3 = df3[numerical].quantile(0.75)
IQR = Q3 - Q1
LB = Q1 - 1.5 * IQR
UB = Q3 + 1.5 * IQR

out = pd.DataFrame(df3[numerical][((df3[numerical] < LB) | (df3[numerical] > UB)).any(axis=1)])
df2_outcon_nom = pd.concat([df3copy[categorical], df3[numerical], out], axis=1)
df2_outcon_cor = pd.concat([df3[cat_for_corr], df3[numerical], out], axis=1)

df2_wout_nom = df2_outcon_nom[df2_outcon_nom.isnull().any(1)].dropna(axis=1)
df2_wout_cor = df2_outcon_cor[df2_outcon_cor.isnull().any(1)].dropna(axis=1)

#Raw with removed outliers then standardized (for plots and corr)
df2z_wout_nom = pd.concat([df3copy[categorical], df2_wout_nom[numerical].apply(zscore)], axis=1).drop(['No. of abortions'], axis=1).dropna()
df2z_wout_cor = pd.concat([df3[cat_for_corr], df2_wout_cor[numerical].apply(zscore)], axis=1).drop(['No. of abortions'], axis=1).dropna()

trans = st.sidebar.multiselect("Transform data", ["Scale", "Remove outliers"])
if not trans:
    df4 = df3copy
    df4_corr = df3_corr
elif "Scale" in trans:
    if "Remove outliers" in trans:
        df4 = df2z_wout_nom
        df4_corr = df2z_wout_cor
    else:
        df4 = df3z_nom
        df4_corr = df3z_cor
elif "Remove outliers" in trans:
    if "Scale" in trans:
        df4 = df2z_wout_nom
        df4_corr = df2z_wout_cor
    else:
        df4 = df2_wout_nom
        df4_corr = df2_wout_cor

#---Summary Plots---------------------------------------------------------------------------------------

# with st.sidebar.form("key1"):
#     button1 = st.form_submit_button("Apply filters")
def filter_dataframe(df4: pd.DataFrame) -> pd.DataFrame:
        modify = st.sidebar.checkbox("Add filters for univariate and bivariate plots")
        if not modify:
            return df4

        df4 = df4.copy()

        modification_container = st.container()

        with modification_container:
            to_filter_columns = st.sidebar.multiselect("Filter dataframe on", df4.columns)
            for column in to_filter_columns:
                left, right = st.sidebar.columns((1, 20))
                left.write("↳")
                if is_categorical_dtype(df4[column]) or df4[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df4[column].unique(),
                        default=list(df4[column].unique()),
                    )
                    df4 = df4[df4[column].isin(user_cat_input)]
                elif is_numeric_dtype(df4[column]):
                    _min = float(df4[column].min())
                    _max = float(df4[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        min_value=_min,
                        max_value=_max,
                        value=(_min, _max),
                        step=step,
                    )
                    df4 = df4[df4[column].between(*user_num_input)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df4 = df4[df4[column].astype(str).str.contains(user_text_input)]
        return df4

df5 = filter_dataframe(df4)

data_show1 = st.checkbox('Show/hide data head')
if data_show1:
    st.dataframe(df5.head(5))

data_show2 = st.checkbox('Show/hide summary statistics of numerical features')
if data_show2:
    st.dataframe(df5.describe())

st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
col1, col2 = st.columns(2,gap='large')

with col1:
    st.subheader("""Bar Plot of categorical features by diagnosis""")
    cat_x1 = st.selectbox('Select symptom',categorical)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # fig1, ax = plt.subplots(figsize=(10,10))
    # sns.set(font_scale=2)

    fig1 = px.histogram(df5, x="PCOS (Y/N)", color=cat_x1, barmode='group')
    fig1.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    },font=dict(
        # family="Courier New, monospace",
        size=12,
        # color="RebeccaPurple"
        ),autosize=False,
        width=350,
        height=450,
    )
    st.write(fig1)

    if cat_x1 ==   'PCOS (Y/N)'    :
        st.caption('This indicates if the patient has PCOS or not.')        
    if cat_x1 ==    'Blood Group'   :
        st.caption("One study claims that females with a blood type O positive have the highest chance of developing PCOS, followed by blood type B positive, while Rh negative didn’t have any relationship with PCOS.")       
    if cat_x1 ==    'Weight gain(Y/N)'  :
        st.caption("Weight gain and obesity are closely related with PCOS. 38%-88% of women that have PCOS are either overweight or obese.")        
    if cat_x1 ==    'Pregnant(Y/N)' :
        st.caption("PCOS generates higher risks for women during pregnancy. Not only does it affect the mother, it also affects the child. Women with PCOS are more likely to miscarry in the first few months of pregnancy than those without.")       
    if cat_x1 ==    'hair growth(Y/N)'  :
        st.caption("One symptom of PCOS is hirsutism, which is excess facial and body hair.")       
    if cat_x1 ==    'Skin darkening (Y/N)'  :
        st.caption("PCOS can cause your skin to have dark patches. This is due to the insulin resistance experienced by those with PCOS.")      
    if cat_x1 ==    'Hair loss(Y/N)'    :
        st.caption("Some women with PCOS experience hair thinning and hair loss.")      
    if cat_x1 ==    'Pimples(Y/N)'  :
        st.caption("PCOS causes the ovaries to produce more androgens, which triggers the production of oil in the skin, which leads to acne.")     
    if cat_x1 ==    'Fast food (Y/N)'   :
        st.caption("Women with PCOS are advised to avoid saturated and trans fats, refined carbohydrates, sugar, dairy, and alcohol. A healthier diet can help manage the PCOS symptoms and reach or manage a healthy weight.")     
    if cat_x1 ==    'Reg.Exercise(Y/N)' :
        st.caption("Exercising has a lot of benefits for those who have PCOS. Reducing insulin resistance, stabilizing mood, improving fertility, and weight loss are just some of the benefits that can be gained through exercise.")            

with col2:
    st.subheader("""Violin Plot of numerical features by diagnosis""")
    num_y1 = st.selectbox('Compare diagnosis for what feature?', numerical)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    if num_y1:
        # dffig = plt.figure(figsize=(10,10))
        # dfax = dffig.add_subplot(111)

        fig2 = px.violin(df5, x="PCOS (Y/N)", y=num_y1, box=True, hover_data=df5.columns)
        fig2.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },font=dict(
            size=12
            ),autosize=False,
            width=450,
            height=450,
        )

    st.write(fig2)

    if num_y1 == ' Age (yrs)':
        st.caption('PCOS can be detected any time after puberty. Majority of women find out that they have PCOS when they are in their 20s and 30s.')
    if num_y1 == 'Weight (Kg)':
        st.caption('Weight gain and obesity are closely related with PCOS. 38%-88% of women that have PCOS are either overweight or obese.')
    if num_y1 == 'Height(Cm) ':
        st.caption('Height can be a factor when it comes to PCOS. A study has shown that height is positively related with PCOS. Girls that are taller have a higher chance of developing PCOS than girls of average height.')
    if num_y1 == 'Pulse rate(bpm) ':
        st.caption('The range that is considered as normal for adults is 60 to 100 beats per minute. Having a lower heart rate means that your heart is more efficient and you have better cardiovascular fitness.')
    if num_y1 == 'RR (breaths/min)':
        st.caption('The range that is considered as normal for adults is 12-16 breaths per minute. A study states that women with PCOS are about 10% more likely to have lower lung function.')
    if num_y1 == 'Hb(g/dl)':
        st.caption('This is the amount of hemoglobin in your blood. The normal range for female adults is 12.1 to 15.1 g/dL. It is reported that women with PCOS have hemoglobin levels that are significantly higher.')
    if num_y1 == 'Cycle(R/I)':
        st.caption('Many women with PCOS experience irregular periods. But some women with PCOS have regular periods.')
    if num_y1 == 'Cycle length(days)':
        st.caption('On average, the menstrual cycle is 21-35 days. Irregular periods are those that last below 21 days or longer than 35 days.')
    if num_y1 == 'Marriage Status (Yrs)':
        st.caption('PCOS can affect your relationship with your partner. Intimacy and pregnancy for someone with PCOS can be difficult and it can be reasons why some relationships struggle.')
    if num_y1 == 'No. of abortions':
        st.caption('Those who have PCOS have a high abortion rate of 30-50% in the first trimester, chances of recurrent early abortion is 36-82%, and habitual abortion is 58%.')
    if num_y1 == '  I   beta-HCG(mIU/mL)':
        st.caption('This is a hormone that the body produces during pregnancy. For adult women, a level of less than 5 mIU/mL is considered normal and it means that you are unlikely to be pregnant. Having a result of more than 20 mIU/mL means that you are likely to be pregnant.')
    if num_y1 == 'FSH(mIU/mL)':
        st.caption('This is a hormone that helps control the menstrual cycle. It also stimulates the growth of eggs in the ovaries. The normal levels for female are before puberty: 0 to 4.0 mIU/mL, during puberty: 0.3 to 10.0 mIU/mL, women who are still menstruating: 4.7 to 21.5 mIU/mL, and after menopause: 25.8 to 134.8 mIU/mL. High levels of FSH might suggest that you have PCOS.')
    if num_y1 == 'LH(mIU/mL)':
        st.caption('This is a hormone that the pituitary releases during ovulation. The normal levels for women depends on the phase of the menstrual cycle. The levels are follicular phase of menstrual cycle: 1.68 to 15 IU/mL, midcycle peak: 21.9 to 56.6 IU/mL, luteal phase: 0.61 to 16.3 IU/mL, postmenopausal: 14.2 to 52.3 IU/mL. Many women with PCOS have LH within the range of 5-20 mIU/mL.')
    if num_y1 == 'Waist:Hip Ratio':
        st.caption('Waist to hip ratio correspond to a high chance PCOS.')
    if num_y1 == 'TSH (mIU/L)':
        st.caption('This is a hormone that helps control the production of hormones and the metabolism of your body. The normal levels are from 0.4 to 4.0 mIU/L if you have no symptoms of having an under- or over-active thyroid. If you are receiving treatment for a thyroid disorder, TSH levels should be between 0.5 and 2.0 mIU/L. Women with PCOS generally have normal TSH levels.')
    if num_y1 == 'AMH(ng/mL)':
        st.caption('This is a hormone that is used to measure a woman’s ovarian reserve. The normal levels depends on the age of the woman -- under 33 years old: 2.1 – 6.8 ng/ml, 33 - 37 years old: 1.7 – 3.5 ng/ml, 38 - 40 years old: 1.1 – 3.0 ng/ml, 41+ years old: 0.5 – 2.5 ng/ml. An AMH above 6.8 ng/ml is considered high and is a potential sign of PCOS at any age.')
    if num_y1 == 'PRL(ng/mL)':
        st.caption('This is a hormone that triggers breast development and breast milk production in women. The normal range for non-pregnant women is less than 25 ng/mL , while it is 80 to 400 ng/mL for pregnant women. Women with PCOS usually have normal prolactin levels.')
    if num_y1 == 'Vit D3 (ng/mL)':
        st.caption('Vitamin D values are defined as normal : ≥20 ng/mL, vitamin D insufficiency: 12 to 20 ng/mL, and vitamin D deficiency: less than 12 ng/mL. Women with PCOS have a relatively high incidence of vitamin D deficiency. Vitamin D deficiency could aggravate some PCOS symptoms.')
    if num_y1 == 'PRG(ng/mL)':
        st.caption('Progesterone aids the uterus in getting ready so it can host a fertilized egg during pregnancy. The normal ranges are prepubescent girls: 0.1 to 0.3 ng/mL, follicular stage of the menstrual cycle: 0.1 to 0.7 ng/mL, luteal stage of the menstrual cycle: 2 to 25 ng/mL, first trimester of pregnancy: 10 to 44 ng/mL, second trimester of pregnancy: 19.5 to 82.5 ng/mL, third trimester of pregnancy: 65 to 290 ng/mL')
    if num_y1 == 'RBS(mg/dl)':
        st.caption('Blood sugar levels are defined normal: less than 140 mg/dL, prediabetes: between 140 and 199 mg/dL, diabetes: more than 200 mg/dL. In a random blood sugar test, a result of 200 mg/dL or higher would indicate diabetes. More than 50% of women with PCOS develop type 2 diabetes by 40 years old.')
    if num_y1 == 'BP _Systolic (mmHg)':
        st.caption('This is the maximum pressure your heart exerts while it is beating. A systolic pressure that is above 90 mm Hg and less than 120 mm Hg is the normal range. Many symptoms linked with PCOS can cause blood pressure to increase.')
    if num_y1 == 'BP _Diastolic (mmHg)':
        st.caption('This is the amount of pressure in the arteries between beats. A diastolic pressure that is above 60 mm Hg and less than 80 mm Hg is the normal range. Many symptoms linked with PCOS can cause blood pressure to increase.')
    if num_y1 == 'Endometrium (mm)':
        st.caption('Endometrial thickness of more than 8.5 mm could be linked with endometrial disease in women with PCOS.')
    if num_y1 == 'Avg Follicle No':
        st.caption('Having an antral follicle count of 6-10 means that a woman has a normal ovarian reserve. A count of less than 6 is considered low, while a count of greater than 12 is considered high. The threshold in determining PCOS is 12 or more follicles measuring 2-9 mm.')
    if num_y1 == 'Avg F size (mm)':
        st.caption('A regular ovary comprises of 8-10 follicles ranging from 2 to 28 mm. Follicles sized less than 18 mm are called antral follicles, while follicles sized 18 to 28 mm are called dominant follicles. The threshold in determining PCOS is 12 or more follicles measuring 2-9 mm.')


#---Bivariate---------------------------------------------------------------------------------------

st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
l1, m1, r1 = st.columns((2,5,1))
with m1:
    st.subheader("Bivariate scatterplot by diagnosis")

col3, col4, col5 = st.columns(3,gap='large')

with col3:
    alt_x = st.selectbox("Compare diagnosis for which features (X)?", numerical)
with col4:
    alt_y = st.selectbox("Compare diagnosis for which features? (Y)", numerical)
with col5:
    cat_hue = st.selectbox("Choose hue", categorical)

if alt_x and alt_y and cat_hue:
    fig3 = px.scatter(df5, alt_x, alt_y, color=cat_hue, trendline="ols")
    fig3.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    },font=dict(
        size=18
        )   
    )

st.write(fig3)

if alt_x == ' Age (yrs)':
    st.markdown('X-axis:')
    st.caption('PCOS can be detected any time after puberty. Majority of women find out that they have PCOS when they are in their 20s and 30s.')
if alt_x == 'Weight (Kg)':
    st.markdown('X-axis:')
    st.caption('Weight gain and obesity are closely related with PCOS. 38%-88% of women that have PCOS are either overweight or obese.')
if alt_x == 'Height(Cm) ':
    st.markdown('X-axis:')
    st.caption('Height can be a factor when it comes to PCOS. A study has shown that height is positively related with PCOS. Girls that are taller have a higher chance of developing PCOS than girls of average height.')
if alt_x == 'Pulse rate(bpm) ':
    st.markdown('X-axis:')
    st.caption('The range that is considered as normal for adults is 60 to 100 beats per minute. Having a lower heart rate means that your heart is more efficient and you have better cardiovascular fitness.')
if alt_x == 'RR (breaths/min)':
    st.markdown('X-axis:')
    st.caption('The range that is considered as normal for adults is 12-16 breaths per minute. A study states that women with PCOS are about 10% more likely to have lower lung function.')
if alt_x == 'Hb(g/dl)':
    st.markdown('X-axis:')
    st.caption('This is the amount of hemoglobin in your blood. The normal range for female adults is 12.1 to 15.1 g/dL. It is reported that women with PCOS have hemoglobin levels that are significantly higher.')
if alt_x == 'Cycle(R/I)':
    st.markdown('X-axis:')
    st.caption('Many women with PCOS experience irregular periods. But some women with PCOS have regular periods.')
if alt_x == 'Cycle length(days)':
    st.markdown('X-axis:')
    st.caption('On average, the menstrual cycle is 21-35 days. Irregular periods are those that last below 21 days or longer than 35 days.')
if alt_x == 'Marriage Status (Yrs)':
    st.markdown('X-axis:')
    st.caption('PCOS can affect your relationship with your partner. Intimacy and pregnancy for someone with PCOS can be difficult and it can be reasons why some relationships struggle.')
if alt_x == 'No. of abortions':
    st.markdown('X-axis:')
    st.caption('Those who have PCOS have a high abortion rate of 30-50% in the first trimester, chances of recurrent early abortion is 36-82%, and habitual abortion is 58%.')
if alt_x == '  I   beta-HCG(mIU/mL)':
    st.markdown('X-axis:')
    st.caption('This is a hormone that the body produces during pregnancy. For adult women, a level of less than 5 mIU/mL is considered normal and it means that you are unlikely to be pregnant. Having a result of more than 20 mIU/mL means that you are likely to be pregnant.')
if alt_x == 'FSH(mIU/mL)':
    st.markdown('X-axis:')
    st.caption('This is a hormone that helps control the menstrual cycle. It also stimulates the growth of eggs in the ovaries. The normal levels for female are before puberty: 0 to 4.0 mIU/mL, during puberty: 0.3 to 10.0 mIU/mL, women who are still menstruating: 4.7 to 21.5 mIU/mL, and after menopause: 25.8 to 134.8 mIU/mL. High levels of FSH might suggest that you have PCOS.')
if alt_x == 'LH(mIU/mL)':
    st.markdown('X-axis:')
    st.caption('This is a hormone that the pituitary releases during ovulation. The normal levels for women depends on the phase of the menstrual cycle. The levels are follicular phase of menstrual cycle: 1.68 to 15 IU/mL, midcycle peak: 21.9 to 56.6 IU/mL, luteal phase: 0.61 to 16.3 IU/mL, postmenopausal: 14.2 to 52.3 IU/mL. Many women with PCOS have LH within the range of 5-20 mIU/mL.')
if alt_x == 'Waist:Hip Ratio':
    st.markdown('X-axis:')
    st.caption('Waist to hip ratio correspond to a high chance PCOS.')
if alt_x == 'TSH (mIU/L)':
    st.markdown('X-axis:')
    st.caption('This is a hormone that helps control the production of hormones and the metabolism of your body. The normal levels are from 0.4 to 4.0 mIU/L if you have no symptoms of having an under- or over-active thyroid. If you are receiving treatment for a thyroid disorder, TSH levels should be between 0.5 and 2.0 mIU/L. Women with PCOS generally have normal TSH levels.')
if alt_x == 'AMH(ng/mL)':
    st.markdown('X-axis:')
    st.caption('This is a hormone that is used to measure a woman’s ovarian reserve. The normal levels depends on the age of the woman -- under 33 years old: 2.1 – 6.8 ng/ml, 33 - 37 years old: 1.7 – 3.5 ng/ml, 38 - 40 years old: 1.1 – 3.0 ng/ml, 41+ years old: 0.5 – 2.5 ng/ml. An AMH above 6.8 ng/ml is considered high and is a potential sign of PCOS at any age.')
if alt_x == 'PRL(ng/mL)':
    st.markdown('X-axis:')
    st.caption('This is a hormone that triggers breast development and breast milk production in women. The normal range for non-pregnant women is less than 25 ng/mL , while it is 80 to 400 ng/mL for pregnant women. Women with PCOS usually have normal prolactin levels.')
if alt_x == 'Vit D3 (ng/mL)':
    st.markdown('X-axis:')
    st.caption('Vitamin D values are defined as normal : ≥20 ng/mL, vitamin D insufficiency: 12 to 20 ng/mL, and vitamin D deficiency: less than 12 ng/mL. Women with PCOS have a relatively high incidence of vitamin D deficiency. Vitamin D deficiency could aggravate some PCOS symptoms.')
if alt_x == 'PRG(ng/mL)':
    st.markdown('X-axis:')
    st.caption('Progesterone aids the uterus in getting ready so it can host a fertilized egg during pregnancy. The normal ranges are prepubescent girls: 0.1 to 0.3 ng/mL, follicular stage of the menstrual cycle: 0.1 to 0.7 ng/mL, luteal stage of the menstrual cycle: 2 to 25 ng/mL, first trimester of pregnancy: 10 to 44 ng/mL, second trimester of pregnancy: 19.5 to 82.5 ng/mL, third trimester of pregnancy: 65 to 290 ng/mL')
if alt_x == 'RBS(mg/dl)':
    st.markdown('X-axis:')
    st.caption('Blood sugar levels are defined normal: less than 140 mg/dL, prediabetes: between 140 and 199 mg/dL, diabetes: more than 200 mg/dL. In a random blood sugar test, a result of 200 mg/dL or higher would indicate diabetes. More than 50% of women with PCOS develop type 2 diabetes by 40 years old.')
if alt_x == 'BP _Systolic (mmHg)':
    st.markdown('X-axis:')
    st.caption('This is the maximum pressure your heart exerts while it is beating. A systolic pressure that is above 90 mm Hg and less than 120 mm Hg is the normal range. Many symptoms linked with PCOS can cause blood pressure to increase.')
if alt_x == 'BP _Diastolic (mmHg)':
    st.markdown('X-axis:')
    st.caption('This is the amount of pressure in the arteries between beats. A diastolic pressure that is above 60 mm Hg and less than 80 mm Hg is the normal range. Many symptoms linked with PCOS can cause blood pressure to increase.')
if alt_x == 'Endometrium (mm)':
    st.markdown('X-axis:')
    st.caption('Endometrial thickness of more than 8.5 mm could be linked with endometrial disease in women with PCOS.')
if alt_x == 'Avg Follicle No':
    st.markdown('X-axis:')
    st.caption('Having an antral follicle count of 6-10 means that a woman has a normal ovarian reserve. A count of less than 6 is considered low, while a count of greater than 12 is considered high. The threshold in determining PCOS is 12 or more follicles measuring 2-9 mm.')
if alt_x == 'Avg F size (mm)':
    st.markdown('X-axis:')
    st.caption('A regular ovary comprises of 8-10 follicles ranging from 2 to 28 mm. Follicles sized less than 18 mm are called antral follicles, while follicles sized 18 to 28 mm are called dominant follicles. The threshold in determining PCOS is 12 or more follicles measuring 2-9 mm.')


if alt_y == ' Age (yrs)':
    st.markdown('Y-axis:')
    st.caption('PCOS can be detected any time after puberty. Majority of women find out that they have PCOS when they are in their 20s and 30s.')
if alt_y == 'Weight (Kg)':
    st.markdown('Y-axis:')
    st.caption('Weight gain and obesity are closely related with PCOS. 38%-88% of women that have PCOS are either overweight or obese.')
if alt_y == 'Height(Cm) ':
    st.markdown('Y-axis:')
    st.caption('Height can be a factor when it comes to PCOS. A study has shown that height is positively related with PCOS. Girls that are taller have a higher chance of developing PCOS than girls of average height.')
if alt_y == 'Pulse rate(bpm) ':
    st.markdown('Y-axis:')
    st.caption('The range that is considered as normal for adults is 60 to 100 beats per minute. Having a lower heart rate means that your heart is more efficient and you have better cardiovascular fitness.')
if alt_y == 'RR (breaths/min)':
    st.markdown('Y-axis:')
    st.caption('The range that is considered as normal for adults is 12-16 breaths per minute. A study states that women with PCOS are about 10% more likely to have lower lung function.')
if alt_y == 'Hb(g/dl)':
    st.markdown('Y-axis:')
    st.caption('This is the amount of hemoglobin in your blood. The normal range for female adults is 12.1 to 15.1 g/dL. It is reported that women with PCOS have hemoglobin levels that are significantly higher.')
if alt_y == 'Cycle(R/I)':
    st.markdown('Y-axis:')
    st.caption('Many women with PCOS experience irregular periods. But some women with PCOS have regular periods.')
if alt_y == 'Cycle length(days)':
    st.markdown('Y-axis:')
    st.caption('On average, the menstrual cycle is 21-35 days. Irregular periods are those that last below 21 days or longer than 35 days.')
if alt_y == 'Marriage Status (Yrs)':
    st.markdown('Y-axis:')
    st.caption('PCOS can affect your relationship with your partner. Intimacy and pregnancy for someone with PCOS can be difficult and it can be reasons why some relationships struggle.')
if alt_y == 'No. of abortions':
    st.markdown('Y-axis:')
    st.caption('Those who have PCOS have a high abortion rate of 30-50% in the first trimester, chances of recurrent early abortion is 36-82%, and habitual abortion is 58%.')
if alt_y == '  I   beta-HCG(mIU/mL)':
    st.markdown('Y-axis:')
    st.caption('This is a hormone that the body produces during pregnancy. For adult women, a level of less than 5 mIU/mL is considered normal and it means that you are unlikely to be pregnant. Having a result of more than 20 mIU/mL means that you are likely to be pregnant.')
if alt_y == 'FSH(mIU/mL)':
    st.markdown('Y-axis:')
    st.caption('This is a hormone that helps control the menstrual cycle. It also stimulates the growth of eggs in the ovaries. The normal levels for female are before puberty: 0 to 4.0 mIU/mL, during puberty: 0.3 to 10.0 mIU/mL, women who are still menstruating: 4.7 to 21.5 mIU/mL, and after menopause: 25.8 to 134.8 mIU/mL. High levels of FSH might suggest that you have PCOS.')
if alt_y == 'LH(mIU/mL)':
    st.markdown('Y-axis:')
    st.caption('This is a hormone that the pituitary releases during ovulation. The normal levels for women depends on the phase of the menstrual cycle. The levels are follicular phase of menstrual cycle: 1.68 to 15 IU/mL, midcycle peak: 21.9 to 56.6 IU/mL, luteal phase: 0.61 to 16.3 IU/mL, postmenopausal: 14.2 to 52.3 IU/mL. Many women with PCOS have LH within the range of 5-20 mIU/mL.')
if alt_y == 'Waist:Hip Ratio':
    st.markdown('Y-axis:')
    st.caption('Waist to hip ratio correspond to a high chance PCOS.')
if alt_y == 'TSH (mIU/L)':
    st.markdown('Y-axis:')
    st.caption('This is a hormone that helps control the production of hormones and the metabolism of your body. The normal levels are from 0.4 to 4.0 mIU/L if you have no symptoms of having an under- or over-active thyroid. If you are receiving treatment for a thyroid disorder, TSH levels should be between 0.5 and 2.0 mIU/L. Women with PCOS generally have normal TSH levels.')
if alt_y == 'AMH(ng/mL)':
    st.markdown('Y-axis:')
    st.caption('This is a hormone that is used to measure a woman’s ovarian reserve. The normal levels depends on the age of the woman -- under 33 years old: 2.1 – 6.8 ng/ml, 33 - 37 years old: 1.7 – 3.5 ng/ml, 38 - 40 years old: 1.1 – 3.0 ng/ml, 41+ years old: 0.5 – 2.5 ng/ml. An AMH above 6.8 ng/ml is considered high and is a potential sign of PCOS at any age.')
if alt_y == 'PRL(ng/mL)':
    st.markdown('Y-axis:')
    st.caption('This is a hormone that triggers breast development and breast milk production in women. The normal range for non-pregnant women is less than 25 ng/mL , while it is 80 to 400 ng/mL for pregnant women. Women with PCOS usually have normal prolactin levels.')
if alt_y == 'Vit D3 (ng/mL)':
    st.markdown('Y-axis:')
    st.caption('Vitamin D values are defined as normal : ≥20 ng/mL, vitamin D insufficiency: 12 to 20 ng/mL, and vitamin D deficiency: less than 12 ng/mL. Women with PCOS have a relatively high incidence of vitamin D deficiency. Vitamin D deficiency could aggravate some PCOS symptoms.')
if alt_y == 'PRG(ng/mL)':
    st.markdown('Y-axis:')
    st.caption('Progesterone aids the uterus in getting ready so it can host a fertilized egg during pregnancy. The normal ranges are prepubescent girls: 0.1 to 0.3 ng/mL, follicular stage of the menstrual cycle: 0.1 to 0.7 ng/mL, luteal stage of the menstrual cycle: 2 to 25 ng/mL, first trimester of pregnancy: 10 to 44 ng/mL, second trimester of pregnancy: 19.5 to 82.5 ng/mL, third trimester of pregnancy: 65 to 290 ng/mL')
if alt_y == 'RBS(mg/dl)':
    st.markdown('Y-axis:')
    st.caption('Blood sugar levels are defined normal: less than 140 mg/dL, prediabetes: between 140 and 199 mg/dL, diabetes: more than 200 mg/dL. In a random blood sugar test, a result of 200 mg/dL or higher would indicate diabetes. More than 50% of women with PCOS develop type 2 diabetes by 40 years old.')
if alt_y == 'BP _Systolic (mmHg)':
    st.markdown('Y-axis:')
    st.caption('This is the maximum pressure your heart exerts while it is beating. A systolic pressure that is above 90 mm Hg and less than 120 mm Hg is the normal range. Many symptoms linked with PCOS can cause blood pressure to increase.')
if alt_y == 'BP _Diastolic (mmHg)':
    st.markdown('Y-axis:')
    st.caption('This is the amount of pressure in the arteries between beats. A diastolic pressure that is above 60 mm Hg and less than 80 mm Hg is the normal range. Many symptoms linked with PCOS can cause blood pressure to increase.')
if alt_y == 'Endometrium (mm)':
    st.markdown('Y-axis:')
    st.caption('Endometrial thickness of more than 8.5 mm could be linked with endometrial disease in women with PCOS.')
if alt_y == 'Avg Follicle No':
    st.markdown('Y-axis:')
    st.caption('Having an antral follicle count of 6-10 means that a woman has a normal ovarian reserve. A count of less than 6 is considered low, while a count of greater than 12 is considered high. The threshold in determining PCOS is 12 or more follicles measuring 2-9 mm.')
if alt_y == 'Avg F size (mm)':
    st.markdown('Y-axis:')
    st.caption('A regular ovary comprises of 8-10 follicles ranging from 2 to 28 mm. Follicles sized less than 18 mm are called antral follicles, while follicles sized 18 to 28 mm are called dominant follicles. The threshold in determining PCOS is 12 or more follicles measuring 2-9 mm.')


if cat_hue ==   'PCOS (Y/N)'    :
    st.markdown('Hue:')
    st.caption('This indicates if the patient has PCOS or not.')        
if cat_hue ==   'Blood Group'   :
    st.markdown('Hue:')
    st.caption("One study claims that females with a blood type O positive have the highest chance of developing PCOS, followed by blood type B positive, while Rh negative didn’t have any relationship with PCOS.")       
if cat_hue ==   'Weight gain(Y/N)'  :
    st.markdown('Hue:')
    st.caption("Weight gain and obesity are closely related with PCOS. 38%-88% of women that have PCOS are either overweight or obese.")        
if cat_hue ==   'Pregnant(Y/N)' :
    st.markdown('Hue:')
    st.caption("PCOS generates higher risks for women during pregnancy. Not only does it affect the mother, it also affects the child. Women with PCOS are more likely to miscarry in the first few months of pregnancy than those without.")       
if cat_hue ==   'hair growth(Y/N)'  :
    st.markdown('Hue:')
    st.caption("One symptom of PCOS is hirsutism, which is excess facial and body hair.")       
if cat_hue ==   'Skin darkening (Y/N)'  :
    st.markdown('Hue:')
    st.caption("PCOS can cause your skin to have dark patches. This is due to the insulin resistance experienced by those with PCOS.")      
if cat_hue ==   'Hair loss(Y/N)'    :
    st.markdown('Hue:')
    st.caption("Some women with PCOS experience hair thinning and hair loss.")      
if cat_hue ==   'Pimples(Y/N)'  :
    st.markdown('Hue:')
    st.caption("PCOS causes the ovaries to produce more androgens, which triggers the production of oil in the skin, which leads to acne.")     
if cat_hue ==   'Fast food (Y/N)'   :
    st.markdown('Hue:')
    st.caption("Women with PCOS are advised to avoid saturated and trans fats, refined carbohydrates, sugar, dairy, and alcohol. A healthier diet can help manage the PCOS symptoms and reach or manage a healthy weight.")     
if cat_hue ==   'Reg.Exercise(Y/N)' :
    st.markdown('Hue:')
    st.caption("Exercising has a lot of benefits for those who have PCOS. Reducing insulin resistance, stabilizing mood, improving fertility, and weight loss are just some of the benefits that can be gained through exercise.")      

#---Correlation---------------------------------------------------------------------------------------

st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
l3, m3, r3 = st.columns((4,5,1))
with m3:
    st.subheader("Correlation")

with st.form("key2"):

    corr_range = st.slider("Select correlation magnitude range", value=[0.0, 1.0], step=0.05)

    so = pd.DataFrame(df4_corr.corr().unstack().sort_values(kind="quicksort"), columns=['corrcoeff'])
    so.reset_index(inplace=True)
    soc = so.rename(columns = {'level_0':'Var1', 'level_1':'Var2'})
    socorr = soc[soc['Var1'] != soc['Var2']]
    selected_corr = socorr.where(abs(socorr['corrcoeff']) >= min(corr_range)).where(abs(socorr['corrcoeff']) <= max(corr_range)).dropna()
    filtered_corr_vars = selected_corr['Var1'].unique().tolist()
    selected_corr_data = df4_corr[filtered_corr_vars]

    button2 = st.form_submit_button("Apply range")

corr_mat = st.checkbox('Show/hide correlation matrix')
cor_pal = (sns.color_palette("colorblind", as_cmap=True))
plt.tick_params(axis='both', which='major', labelsize=14)
if corr_mat:
    fig4 = px.imshow(round(selected_corr_data.corr(),2), text_auto=True, zmin=-1, zmax=1, color_continuous_scale=px.colors.sequential.Bluered)
    fig4.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    },font=dict(
        size=12
        ),autosize=False,
            width=750,
            height=750, 
    )
    st.write(fig4)

sources = st.sidebar.checkbox("Sources")
links = ["https://www.hopkinsmedicine.org/health/conditions-and-diseases/polycystic-ovary-syndrome-pcos#:~:text=PCOS%20is%20a%20very%20common,%2C%20infertility%2C%20and%20weight%20gain.",
"https://www.womenshealth.gov/a-z-topics/polycystic-ovary-syndrome#:~:text=However%2C%20their%20PCOS%20hormonal%20imbalance,with%20PCOS%20than%20those%20without.",
"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6734597/#:~:text=In%20women%20who%20are%20genetically,are%20either%20overweight%20or%20obese.", 
"https://pubmed.ncbi.nlm.nih.gov/33979806/#:~:text=Girls%20who%20were%20persistently%20tall,girls%20with%20average%20height%20growth.", 
"https://journals.sagepub.com/doi/abs/10.1177/00369330211043206", 
"https://www.verywellhealth.com/how-pcos-affects-your-relationships-2616703#:~:text=Infertility%2C%20or%20difficulty%20getting%20pregnant,comes%20with%20being%20a%20couple.", 
"https://www.mayoclinic.org/healthy-lifestyle/fitness/expert-answers/heart-rate/faq-20057979#:~:text=A%20normal%20resting%20heart%20rate%20for%20adults%20ranges%20from%2060,to%2040%20beats%20per%20minute.", 
"https://www.hopkinsmedicine.org/health/conditions-and-diseases/vital-signs-body-temperature-pulse-rate-respiration-rate-blood-pressure#:~:text=Normal%20respiration%20rates%20for%20an,to%2016%20breaths%20per%20minute.", 
"https://www.eurekalert.org/news-releases/521225#:~:text=Researchers%20used%20genetic%20variants%20associated,function%2C%20compared%20to%20other%20women.", 
"https://www.healthline.com/health/high-blood-pressure-hypertension/blood-pressure-reading-explained", 
"https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/expert-answers/pulse-pressure/faq-20058189#:~:text=Blood%20pressure%20readings%20are%20given,between%20beats%20(diastolic%20pressure).", 
"https://www.ahajournals.org/doi/full/10.1161/hypertensionaha.107.088138#:~:text=Many%20of%20the%20symptoms%20associated,resistance%20and%20type%202%20diabetes.", 
"https://www.healthline.com/health/high-blood-pressure-hypertension/blood-pressure-reading-explained", 
"https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/expert-answers/pulse-pressure/faq-20058189#:~:text=Blood%20pressure%20readings%20are%20given,between%20beats%20(diastolic%20pressure).", 
"https://www.ahajournals.org/doi/full/10.1161/hypertensionaha.107.088138#:~:text=Many%20of%20the%20symptoms%20associated,resistance%20and%20type%202%20diabetes.", 
"https://www.ijmrhs.com/medical-research/polycystic-ovary-syndrome-blood-group—diet-a-correlative-study-insouth-indian-females.pdf", 
"https://www.ucsfhealth.org/medical-tests/hemoglobin#:~:text=Normal%20Results&text=Female%3A%2012.1%20to%2015.1%20g,121%20to%20151%20g%2FL", 
"https://journals.sagepub.com/doi/10.1177/0300060520952282#:~:text=However%2C%20Han%20et%20al.10,dependent%20stimulatory%20effect%20on%20erythropoiesis.", 
"https://www.urmc.rochester.edu/encyclopedia/content.aspx?contenttypeid=167&contentid=hcg_urine", 
"https://medlineplus.gov/lab-tests/follicle-stimulating-hormone-fsh-levels-test/", 
"https://www.mountsinai.org/health-library/tests/follicle-stimulating-hormone-fsh-blood-test", 
"https://www.urmc.rochester.edu/encyclopedia/content.aspx?ContentTypeID=167&ContentID=luteinizing_hormone_blood#:~:text=Here%20are%20normal%20ranges%3A,21.9%20to%2056.6%20IU%2FmL", 
"https://medlineplus.gov/lab-tests/luteinizing-hormone-lh-levels-test/#:~:text=LH%20plays%20an%20important%20role,This%20is%20known%20as%20ovulation.", 
"https://www.contemporaryobgyn.net/view/hormone-levels-and-pcos", 
"https://www.uclahealth.org/medical-services/surgery/endocrine-surgery/patient-resources/patient-education/endocrine-surgery-encyclopedia/tsh-test#:~:text=When%20a%20thyroid%20disorder%20is,0.5%20and%202.0%20mIU%2FL.", 
"https://www.yourhormones.info/hormones/thyroid-stimulating-hormone/",
"https://www.contemporaryobgyn.net/view/hormone-levels-and-pcos",
"https://rmanetwork.com/blog/anti-mullerian-hormone-amh-testing-of-ovarian-reserve/", 
"https://www.whitelotusclinic.ca/amh-pcos-test/", 
"https://www.cancer.gov/publications/dictionaries/cancer-terms/def/anti-mullerian-hormone", 
"https://www.mountsinai.org/health-library/tests/prolactin-blood-test#:~:text=Test%20is%20Performed-,Prolactin%20is%20a%20hormone%20released%20by%20the%20pituitary%20gland.,and%20milk%20production%20in%20women.", 
"https://www.ucsfhealth.org/medical-tests/prolactin-blood-test#:~:text=Normal%20Results&text=Men%3A%20less%20than%2020%20ng,80%20to%20400%20%C2%B5g%2FL)", 
"https://www.contemporaryobgyn.net/view/hormone-levels-and-pcos", 
"https://www.uptodate.com/contents/vitamin-d-deficiency-beyond-the-basics#:~:text=%E2%97%8FA%20normal%20level%20of,30%20to%2050%20nmol%2FL)", 
"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6467740/", 
"https://www.verywellhealth.com/vitamin-d-more-than-just-a-vitamin-2616313#:~:text=Vitamin%20D%20deficiency%20might%20exacerbate,%2C%20weight%20gain%2C%20and%20anxiety.", 
"https://www.frontiersin.org/articles/10.3389/fendo.2020.00171/full#:~:text=PCOS%20women%20manifest%20a%20relatively,of%20PCOS%20women%20(21).", 
"https://www.urmc.rochester.edu/encyclopedia/content.aspx?ContentTypeID=167&ContentID=progesterone", 
"https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451", 
"https://www.cdc.gov/diabetes/basics/pcos.html#:~:text=Diabetes%E2%80%94more%20than%20half%20of,and%20risk%20increases%20with%20age", 
"https://www.jeanhailes.org.au/health-a-z/pcos/irregular-periods-management-treatment#:~:text=Although%20some%20women%20with%20PCOS,be%20irregular%2C%20or%20stop%20altogether.", 
"https://www.jeanhailes.org.au/health-a-z/pcos/irregular-periods-management-treatment#:~:text=If%20you%20have%20PCOS%2C%20your,fewer%20menstrual%20cycles%20per%20year", 
"https://www.nichd.nih.gov/health/topics/pcos/more_information/FAQs/pregnancy#:~:text=Pregnancy%20complications%20related%20to%20PCOS,as%20are%20women%20without%20PCOS.&text=Some%20research%20shows%20that%20metformin,in%20pregnant%20women%20with%20PCOS.", 
"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7744738/#:~:text=In%20addition%2C%20patients%20with%20PCOS,of%2058%25%20(7).", 
"https://www.healthline.com/health/pcos-hair-loss-2#_noHeaderPrefixedContent", 
"https://www.sepalika.com/pcos/pcos-symptom/pcos-dark-skin-patches/#:~:text=Apart%20from%20cystic%20acne%2C%20hirsutism,commonly%20seen%20in%20skin%20folds.", 
"https://www.healthline.com/health/pcos-hair-loss-2", 
"https://www.medicalnewstoday.com/articles/pcos-acne#diagnosis", 
"https://www.ccrmivf.com/news-events/food-pcos/#:~:text=%E2%80%9CWomen%20with%20PCOS%20should%20avoid,meats%20like%20fast%20food%20hamburgers)", 
"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6734597/#:~:text=In%20women%20who%20are%20genetically,are%20either%20overweight%20or%20obese.", 
"https://exerciseright.com.au/best-types-exercise-pcos/#:~:text=Moderate%20exercise%20like%20brisk%20walking,disease%20and%20type%202%20diabetes.", 
"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3283042/#:~:text=Also%2C%20our%20study%20showed%20that,associated%20with%20the%20endometrial%20disease.", 
"https://obgyn.onlinelibrary.wiley.com/doi/10.1002/uog.13402", 
"https://www.fertilityfamily.co.uk/blog/how-many-eggs-per-follicle-everything-you-need-to-know/#:~:text=A%20woman%20is%20considered%20to,reserve%20is%20greater%20than%2012.", 
"https://obgyn.onlinelibrary.wiley.com/doi/10.1002/uog.13402", 
"https://www.intechopen.com/chapters/45102#:~:text=A%20normal%20ovary%20consists%20of,are%20known%20as%20dominant%20follicles.", 
"https://www.kaggle.com/datasets/ayamoheddine/pcos-dataset", 
"https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos?select=PCOS_infertility.csv"]
s = ''
if sources:
    for i in links:
        s += "- " + i + "\n"
    st.sidebar.markdown(s)

#---Pending---------------------------------------------------------------------------------------

#Optional:
#> update sources as links





