import streamlit as st
import seaborn as sns


# WRITE
## title
st.title("My awesome Streamlit app!")

## write text
st.write("Hello World!")
st.write("My name is Fasha.")

## write 2 lines of text in 1 write command
#st.write("""
#Hello World! \n
#My name is Fasha.
#""")


## write dataframe
df = sns.load_dataset('titanic')
st.write(df)


# MAGIC
# Any time Streamlit sees either a variable or literal value on its own line, it automatically writes that to your app using st.write
"Helo Dunia"
"Nama saya Fasha"


