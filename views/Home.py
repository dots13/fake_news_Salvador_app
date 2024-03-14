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

####
class _MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 embed_dims: List[int],
                 dropout_rate: float,
                 output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): shared feature from domain and text, shape=(batch_size, embed_dim)

        """
        return self.mlp(x)


class _MaskAttentionLayer(torch.nn.Module):
    """
    Compute attention layer
    """
    def __init__(self, input_size: int):
        super(_MaskAttentionLayer, self).__init__()
        self.attention_layer = torch.nn.Linear(input_size, 1)

    def forward(self,
                inputs: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        weights = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            weights = weights.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(weights, dim=-1).unsqueeze(1)
        outputs = torch.matmul(weights, inputs).squeeze(1)
        return outputs, weights


class MDFEND(AbstractModel):
    r"""
    MDFEND: Multi-domain Fake News Detection, CIKM 2021
    paper: https://dl.acm.org/doi/10.1145/3459637.3482139
    code: https://github.com/kennqiang/MDFEND-Weibo21
    """
    def __init__(self,
                 pre_trained_bert_name: str,
                 domain_num: int,
                 mlp_dims: Optional[List[int]] = None,
                 dropout_rate=0.2,
                 expert_num=5):
        """

        Args:
            pre_trained_bert_name (str): the name or local path of pre-trained bert model
            domain_num (int): total number of all domains
            mlp_dims (List[int]): a list of the dimensions in MLP layer, if None, [384] will be taken as default, default=384
            dropout_rate (float): rate of Dropout layer, default=0.2
            expert_num (int): number of experts also called TextCNNLayer, default=5
        """
        super(MDFEND, self).__init__()
        self.domain_num = domain_num
        self.expert_num = expert_num
        self.bert = BertModel.from_pretrained(
            pre_trained_bert_name)
        self.embedding_size = self.bert.config.hidden_size
        self.loss_func = nn.BCELoss()
        if mlp_dims is None:
            mlp_dims = [384]

        filter_num = 64
        filter_sizes = [1, 2, 3, 5, 10]
        experts = [
            TextCNNLayer(self.embedding_size, filter_num, filter_sizes)
            for _ in range(self.expert_num)
        ]
        self.experts = nn.ModuleList(experts)

        self.gate = nn.Sequential(
            nn.Linear(self.embedding_size * 2, mlp_dims[-1]), nn.ReLU(),
            nn.Linear(mlp_dims[-1], self.expert_num), nn.Softmax(dim=1))

        self.attention = _MaskAttentionLayer(self.embedding_size)

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num,
                                            embedding_dim=self.embedding_size)
        self.classifier = _MLP(320, mlp_dims, dropout_rate)

    def forward(self, token_id: Tensor, mask: Tensor,
                domain: Tensor) -> Tensor:
        """

        Args:
            token_id (Tensor): token ids from bert tokenizer, shape=(batch_size, max_len)
            mask (Tensor): mask from bert tokenizer, shape=(batch_size, max_len)
            domain (Tensor): domain id, shape=(batch_size,)

        Returns:
            FloatTensor: the prediction of being fake, shape=(batch_size,)
        """
        text_embedding = self.bert(token_id,
                                   attention_mask=mask).last_hidden_state
        attention_feature, _ = self.attention(text_embedding, mask)

        domain_embedding = self.domain_embedder(domain.view(-1, 1)).squeeze(1)

        gate_input = torch.cat([domain_embedding, attention_feature], dim=-1)
        gate_output = self.gate(gate_input)

        shared_feature = 0
        for i in range(self.expert_num):
            expert_feature = self.experts[i](text_embedding)
            shared_feature += (expert_feature * gate_output[:, i].unsqueeze(1))

        label_pred = self.classifier(shared_feature)

        return torch.sigmoid(label_pred.squeeze(1))

    def calculate_loss(self, data) -> Tensor:
        """
        calculate loss via BCELoss

        Args:
            data (dict): batch data dict

        Returns:
            loss (Tensor): loss value
        """

        token_ids = data['text']['token_id']
        masks = data['text']['mask']
        domains = data['domain']
        labels = data['label']
        output = self.forward(token_ids, masks, domains)
        return self.loss_func(output, labels.float())

    def predict(self, data_without_label) -> Tensor:
        """
        predict the probability of being fake news

        Args:
            data_without_label (Dict[str, Any]): batch data dict

        Returns:
            Tensor: one-hot probability, shape=(batch_size, 2)
        """

        token_ids = data_without_label['text']['token_id']
        masks = data_without_label['text']['mask']
        domains = data_without_label['domain']

        # shape=(n,), data = 1 or 0
        round_pred = torch.round(self.forward(token_ids, masks,
                                              domains)).long()
        # after one hot: shape=(n,2), data = [0,1] or [1,0]
        one_hot_pred = torch.nn.functional.one_hot(round_pred, num_classes=2)
        return one_hot_pred

####

#MODEL_SAVE_PATH = "/content/drive/MyDrive/models-path/last-epoch-model-2024-02-27-15_22_42_6.pth"

# https://drive.google.com/file/d/1-4NIx36LmRF2R5T8Eu5Zku_-CGvV07VE/view?usp=sharing

# https://drive.google.com/file/d/1-4NIx36LmRF2R5T8Eu5Zku_-CGvV07VE/view?usp=drive_link
def load_model():
    f_checkpoint = Path(f"models//bert.pth")
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        gdown.download_folder(id='1-4NIx36LmRF2R5T8Eu5Zku_-CGvV07VE', quiet=True, use_cookies=False)
MDFEND_MODEL = MDFEND(bert, domain_num , expert_num=15 , mlp_dims = [2024 ,1012 ,606])
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
