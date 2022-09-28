import streamlit as st
import seaborn as sns



# DATA DISPLAY ELEMENTS
## interactive dataframe  
## st.dataframe(data=None, width=None, height=None)
df = sns.load_dataset('titanic')
## default
st.write("dataframe default size")
st.dataframe(df)
## 1000 px x 100 px
st.write("dataframe default size")
st.dataframe(df, 1000, 100)

## static table
## st.table(data=None)
st.write("table default")
#st.table(df)
#st.table(df.iloc[0:5,0:3])


## metric
## st.metric(label, value, delta=None, delta_color="normal", help=None)
st.metric(
    label = "My bills",
    value = "RM 60",
    delta = "8%"
)

## 3 metric in 1 row
col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "30 °C", "1.2 °C")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")