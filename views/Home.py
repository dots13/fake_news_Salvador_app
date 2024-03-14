import streamlit as st
from utils import logo
from streamlit_option_menu import option_menu

import streamlit as st
from langchain.llms import OpenAI

import torch
from torch import Tensor
from torch import nn
from transformers import BertModel

import gdown

#MODEL_SAVE_PATH = "/content/drive/MyDrive/models-path/last-epoch-model-2024-02-27-15_22_42_6.pth"

# https://drive.google.com/file/d/1-4NIx36LmRF2R5T8Eu5Zku_-CGvV07VE/view?usp=sharing

# https://drive.google.com/file/d/1-4NIx36LmRF2R5T8Eu5Zku_-CGvV07VE/view?usp=drive_link
def load_model():
    f_checkpoint = Path(f"models//bert.pth")
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        gdown.download_folder(id='1-4NIx36LmRF2R5T8Eu5Zku_-CGvV07VE', quiet=True, use_cookies=False)
MDFEND_MODEL.load_state_dict(torch.load(f="models//bert.pth" , map_location=torch.device('cpu')))

def main():
    #logo()
    st.write("#") #forces the page to load from top 
    st.image("omdena_logo.png", width=300, use_column_width=True)
    st.title(" :blue[Portal de detección de noticias falsas]")
    
    openai_api_key = st.text_input('OpenAI API Key', type='password')
    max_len, bert = 178 , 'dccuchile/bert-base-spanish-wwm-uncased'
    
    #added to center the image on the sidebar to make it look better
    st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
    )
    
    

    with st.container(border=True):
        st.selectbox("Seleccionar Fuente de Noticias.", ["Diario Contra Punto", "Diario El Salvador","Diario La Huella","El Salvador","Focos TV","La Prensa Gráfica","Mala Yerba","Revista Factum","Revista Gato Encerrado","Sivar News","Última Hora SV","Otros"])
        choice = st.radio("Selecciona una opción:", ['Tengo la URL del artículo de noticias', 'Voy a pegar el texto del artículo'], index=0,key='choice')
        
        if choice == "Tengo la URL del artículo de noticias":
            url = st.text_input('Ingrese la URL del artículo de noticias:')
        else:
            text = st.text_area('Pegue el texto del artículo de noticias:')
            
    
        
    with st.container(border=True):
        btn_summary = st.button('Obtener resumen y análisis de la noticia',type='primary')
        if btn_summary:
            summary = 'Este es el resumen'
            insights = 'Idea clave 1'
            st.text_area('Resumen:', disabled=True, value=summary)
            st.text_area('Conclusiones Clave', disabled=True,value=insights)
           
            
    
    with st.container(border=True):
        row1 = st.columns(2, gap='small')
        row2 = st.columns(2,gap='small')
        with row1[0]:
            st.write('##### Validación:')
        with row2[0]:   
            if st.button('Confirmar',type='primary'):
                pass
        with row1[1]:
            st.write('#####')
        with row2[1]:
            if st.button('Borrar'):
                pass
    
    with st.container(border=True):
        row1 = st.columns(2, gap='small')
        row2 = st.columns(2,gap='small')
        with row1[0]:
            st.write('##### Predicción:')
        with row2[0]:   
            st.toggle('Hecho',disabled=True,)
        with row1[1]:
            st.write('#####')
        with row2[1]:   
            conf = '95%'
            st.write(f'###### Confianza: {conf}')
            
            
    with st.container(border=True):
        st.text_area('##### Justificación:',disabled=True)
        
    with st.container(border=True):
        if st.button('Restablecer',type='primary'):
            pass
   
   
    

if __name__== '__main__':
    main()
