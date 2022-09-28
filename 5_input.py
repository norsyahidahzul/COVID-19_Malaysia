import streamlit as st
import datetime


text_content = '''Whales are a widely distributed and diverse group of fully aquatic placental marine mammals. They are an informal grouping within the infraorder Cetacea, which usually excludes dolphins and porpoises. Whales, dolphins and porpoises belong to the order Cetartiodactyla, which consists of even-toed ungulates. Their closest non-cetacean living relatives are the hippopotamuses, from which they and other cetaceans diverged about 54 million years ago. The two parvorders of whales, baleen whales (Mysticeti) and toothed whales (Odontoceti), are thought to have had their last common ancestor around 34 million years ago. Whales consist of eight extant families: Balaenopteridae (the rorquals), Balaenidae (right whales), Cetotheriidae (the pygmy right whale), Eschrichtiidae (the grey whale), Monodontidae (belugas and narwhals), Physeteridae (the sperm whale), Kogiidae (the dwarf and pygmy sperm whale), and Ziphiidae (the beaked whales).

Whales are fully aquatic, open-ocean creatures: they can feed, mate, give birth, suckle and raise their young at sea. Whales range in size from the 2.6 metres (8.5 ft) and 135 kilograms (298 lb) dwarf sperm whale to the 29.9 metres (98 ft) and 190 metric tons (210 short tons) blue whale, which is the largest known creature that has ever lived. The sperm whale is the largest toothed predator on earth. Several whale species exhibit sexual dimorphism, in that the females are larger than males.

Baleen whales have no teeth; instead they have plates of baleen, fringe-like structures that enable them to expel the huge mouthfuls of water they take in, while retaining the krill and plankton they feed on. Because their heads are enormous—making up as much as 40% of their total body mass—and they have throat pleats that enable then to expand their mouths, they are able to take huge quantities of water into their mouth at a time. Baleen whales also have a well developed sense of smell.

Toothed whales, in contrast, have conical teeth adapted to catching fish or squid. They also have such keen hearing—whether above or below the surface of the water—that some can survive even if they are blind. Some species, such as sperm whales, are particularly well adapted for diving to great depths to catch squid and other favoured prey.

Whales evolved from land-living mammals, and must regularly surface to breathe air, although they can remain under water for long periods of time. Some species, such as the sperm whale can stay underwater for up to 90 minutes [2] They have blowholes (modified nostrils) located on top of their heads, through which air is taken in and expelled. They are warm-blooded, and have a layer of fat, or blubber, under the skin. With streamlined fusiform bodies and two limbs that are modified into flippers, whales can travel at speeds of up to 20 knots, though they are not as flexible or agile as seals. Whales produce a great variety of vocalizations, notably the extended songs of the humpback whale. Although whales are widespread, most species prefer the colder waters of the northern and southern hemispheres, and migrate to the equator to give birth. Species such as humpbacks and blue whales are capable of travelling thousands of miles without feeding. Males typically mate with multiple females every year, but females only mate every two to three years. Calves are typically born in the spring and summer; females bear all the responsibility for raising them. Mothers in some species fast and nurse their young for one to two years.'''


# INPUT
## button
## st.button(label, key=None, help=None, on_click=None, args=None, kwargs=None, *, disabled=False)
if st.button('Click here for more info'):
    st.markdown(text_content)
     

# download button
st.download_button('⬇️ Click here to download the info', text_content)


# checkbox
agree = st.checkbox('I agree')
if agree:
     st.write('Great!')
        

# radio
genre = st.radio(
     "What's your favorite movie genre",
     ('Comedy', 'Drama', 'Documentary'))

if genre == 'Comedy':
     st.write('You selected comedy.')
else:
     st.write("You didn't select comedy.")


# select box
option = st.selectbox(
     'How would you like to be contacted?',
     ('Email', 'Home phone', 'Mobile phone'))

st.write('You selected:', option)


# multiselect
options = st.multiselect(
     'What are your favorite colors',
     ['Green', 'Yellow', 'Red', 'Blue'],
     ['Yellow', 'Red'])

st.write('You selected:', options)


# slider
age = st.slider('How old are you?', 0, 130, 25)
st.write("I'm ", age, 'years old')


# slider time
from datetime import time
appointment = st.slider(
     "Schedule your appointment:",
     value=(time(11, 30), time(12, 45)))
st.write("You're scheduled for:", appointment)


# slider date
from datetime import datetime
start_time = st.slider(
     "When do you start?",
     value=datetime(2020, 1, 1, 9, 30),
     format="MM/DD/YY - hh:mm")
st.write("Start time:", start_time)


# select slider
start_color, end_color = st.select_slider(
     'Select a range of color wavelength',
     options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
     value=('red', 'blue'))
st.write('You selected wavelengths between', start_color, 'and', end_color)


# text input
title = st.text_input('Movie title', 'Life of Brian')
st.write('The current movie title is', title)


# number input
number = st.number_input('Insert a number')
st.write('The current number is ', number)


# text area
txt = st.text_area('Your address', '''
Department of Computational and Theoretical Sciences,
Kulliyyah of Science,
International Islamic University Malaysia
     ''')
st.write(txt)


# file uploader
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)


# camera input
picture = st.camera_input("Take a picture")
if picture:
     st.image(picture)









