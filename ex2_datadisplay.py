import streamlit as st
import pandas as pd


## import data from Google Sheet
sheet_url = "https://docs.google.com/spreadsheets/d/1Fx7f6rM5Ce331F9ipsEMn-xRjUKYiR3R_v9IDBusUUY/edit#gid=182521220"
url_1 = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
df = pd.read_csv(url_1)


## view dataframe 



## reproduce metric from this link: 
## https://data-science-at-swast-handover-poc-handover-yfa2kz.streamlitapp.com/
## st.metric(label, value, delta=None, delta_color="normal", help=None)












