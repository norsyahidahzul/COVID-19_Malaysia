import streamlit as st


# TEXT ELEMENTS
## title
st.title("Welcome to My Awesome App !")

## caption
st.caption("Brought to you by myself")

## header
st.header("My first header")

## subheader
st.subheader("My first subheader")

## markdown
st.markdown("""
**Hello** _Fasha_. 
`How are you today?`

> Hint: Good or no good?

My favorite things:
- money
- money
- money

---

```python
fav_food = ["nasi lemak", "roti canai"]
print(fav_food)
```

$$
\gamma + \psi = 1000\eta
$$

""")


## text
st.text("""
IPK College Penang gives you the edge over others by providing a unique teaching package. With award-winning education system, our students will reach their full potential here with us. IPK College provides our students with the tools needed to adapt and excel in a complex and constantly changing world.
""")

