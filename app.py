import streamlit as st
from streamlit_option_menu import option_menu
from views import Home, About, Guide, Team
from utils import logo

def main():
    #logo()
    st.write("#")
   #add streamlit option menu to create options
    with st.sidebar: 
        st.image("omdena_logo2.png")
        choice = option_menu("Menu", ["Inicio", "Nuestro proyecto", "Guia", "Colaboradores"],
                            icons=['house', 'info', 'book', 'people'],
                            menu_icon="app-indicator", default_index=0,
                            styles={
            "container": {"padding": "5!important", "background-color": "#0A122A"},
            "icon": {"color": "#DOCCD0", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#000000"},
            "nav-link-selected": {"background-color": "#3399ff"},
        }
        )
    if choice == "Inicio":
        Home.main()
    if choice == "Nuestro proyecto":
        About.main()
    if choice == "Guia":
        Guide.main()
    if choice == "Colaboradores":
        Team.main()
        
    with st.sidebar:
        "# Falsa o Real?"
        "Detecta noticias falsas con ayuda de nuestra herramienta de IA"
        st.image("Fake_or_Real2.png")


if __name__=='__main__':
   main()