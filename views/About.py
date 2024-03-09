import streamlit as st
from utils import logo
def main():
   #logo()
   st.write("#") #forces the page to load from top 
   st.image("omdena_logo.png", width=300, use_column_width=True)
   st.title(" :blue[Portal de detección de noticias falsas]")
   
   st.image("word_cloud.png", width=700)
   
   
   st.markdown('## La desinformación: una amenaza para la sociedad en la era digital')
   st.markdown('''La desinformación, la difusión de información falsa o engañosa, tiene un impacto sociopolítico profundo en la era digital. Erosiona la confianza en las instituciones, afecta la armonía social y socava la democracia. A nivel individual, genera confusión y ansiedad, y en el ámbito económico, puede llevar a malas decisiones de inversión y pérdidas financieras. La pandemia de COVID-19 ha demostrado el peligro de la desinformación en la salud pública.

El auge de los medios digitales ha agravado el problema. Las redes globales de desinformación complican aún más la lucha contra este fenómeno.
''')
   
   st.markdown('## Combatir la desinformación con herramientas innovadoras')
   st.markdown('''Combatir la desinformación exige soluciones innovadoras. Una herramienta de IA para detectar noticias falsas puede ser una poderosa aliada. Esta herramienta, mediante el análisis de grandes cantidades de datos, puede identificar patrones y características asociadas con la desinformación, verificando la información con fuentes confiables. De esta manera, ayuda a proteger la integridad de la información, promover una sociedad más informada y resiliente, y fortalecer la democracia.''')
   
   
   st.markdown('## Nuestro Objetivo:')
   st.markdown('''En El Salvador, la lucha contra la desinformación es especialmente crucial. El país ha experimentado un aumento en la difusión de noticias falsas, lo que ha erosionado la confianza en las instituciones y ha polarizado a la sociedad, como ya se ha descrito, detectar las noticias falsas permitirá fortalecer la democracia y la cohesión social de la nación. Es por eso que nuestro objetivo es:''')
   st.markdown(''' - Desarrollar una herramienta de IA confiable que pueda identificar y mitigar eficientemente la propagación de la desinformación en El Salvador, protegiendo así la integridad de la información difundida entre el público.''')
               
if __name__=='__main__':
   main()