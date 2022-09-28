import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np


# load titanic data from seaborn
# Source: https://github.com/mwaskom/seaborn-data/blob/master/dowjones.csv
df1 = sns.load_dataset('dowjones')

# view dataframe
st.dataframe(df1)

# line chart
st.line_chart(df1.Price)

# area chart
st.area_chart(df1.Price)

# bar chart
chart_data = pd.DataFrame(
     np.random.randn(50, 3),
     columns=["a", "b", "c"])
st.dataframe(chart_data)
st.bar_chart(chart_data)



