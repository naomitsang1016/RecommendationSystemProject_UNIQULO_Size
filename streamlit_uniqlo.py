# Streamlit test

# Importing the libraries
import streamlit as st
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.spatial.distance import hamming
from collections import Counter
import os
import base64

st.set_page_config(layout="wide")

#import data
df = pd.read_csv('df_20220323ver.csv')
#change column names
df.columns=['ID', 'Gender', 'Age_Group', 'Height', 'Weight', 'Foot_Size',
       'Product_Size', 'Rating', 'Item_Code','Sum_Code','Item_Name','Tags','Dept']
# Data Cleaning 
df['Height']=df['Height'].apply(lambda x:str(x).replace('身長: ','')).apply(lambda x:str(x).replace('"',''))
df['Weight']=df['Weight'].apply(lambda x:str(x).replace('体重: ',''))
df['Product_Size']=df['Product_Size'].apply(lambda x:str(x).replace('購入サイズ: ',''))
df['Foot_Size']=df['Foot_Size'].apply(lambda x:str(x).replace('足のサイズ: ',''))
df=df.replace('回答しない', np.nan)
df=df.replace('その他', np.nan)
df=df.replace('nan', np.nan)

# making dataframe for non foot size
df_nfs=df.copy()
df_nfs.drop('Foot_Size',axis=1,inplace=True)
df_nfs.dropna(inplace=True)

# dropna for dataframe with foot size
df.dropna(inplace=True)

# Data Cleaning with functions

# Gender
def gender_eng(x):
    if x=='男性':
        return 'Male'
    if x=='女性':
        return 'Female'


df['Gender']=df['Gender'].apply(lambda x:gender_eng(x))
df_nfs['Gender']=df_nfs['Gender'].apply(lambda x:gender_eng(x))

df=df[(df['Age_Group']!='10歳以下')&(df['Age_Group']!='0 - 6ヶ月')&(df['Age_Group']!='0 - 3ヶ月')&(df['Age_Group']!='7 - 9ヶ月')&(df['Age_Group']!='10 - 12ヶ月')&(df['Age_Group']!='7 - 12ヶ月')&(df['Age_Group']!='4 - 6ヶ月')]
df_nfs=df_nfs[(df_nfs['Age_Group']!='10歳以下')&(df_nfs['Age_Group']!='0 - 6ヶ月')&(df_nfs['Age_Group']!='0 - 3ヶ月')&(df_nfs['Age_Group']!='7 - 9ヶ月')&(df_nfs['Age_Group']!='10 - 12ヶ月')&(df_nfs['Age_Group']!='7 - 12ヶ月')&(df_nfs['Age_Group']!='4 - 6ヶ月')]

# Age
def Age_impute(x,y,z):
    if x=='10代':
        if z=='Female':
            if y in ['141 - 150cm', '151 - 155cm', '131 - 140cm','121 - 130cm']:
                return '10 - 14歳'
            else:
                return '15 - 19歳'
        if z=='Male':
            if y in ['91 - 95cm','141 - 150cm', '151 - 155cm', '131 - 140cm','121 - 130cm','156 - 160cm']:
                return '10 - 14歳'
            else:
                return '15 - 19歳'
            
    if x=='10歳以下':
        if z=='Female':
            if y in ['141 - 150cm', '151 - 155cm', '131 - 140cm','121 - 130cm']:
                return '10 - 14歳'
            else:
                return '15 - 19歳'
        if z=='Male':
            if y in ['91 - 95cm','141 - 150cm', '151 - 155cm', '131 - 140cm','121 - 130cm','156 - 160cm']:
                return '10 - 14歳'
            else:
                return '15 - 19歳'
    
    
    
            
    elif x=='3 - 6歳':
        return '4 - 6歳'
    
    else:
        return x

df['Age_Group']=df.apply(lambda df:Age_impute(df['Age_Group'],df['Height'],df['Gender']), axis=1)
df_nfs['Age_Group']=df_nfs.apply(lambda df_nfs:Age_impute(df_nfs['Age_Group'],df_nfs['Height'],df_nfs['Gender']), axis=1)


def Age_eng(x):
    if x=='10 - 14歳':
        return '10-14'
    elif x=='7 - 9歳':
        return '7-9'
    elif x=='4 - 6歳':
        return '4-6'
    elif x=='30代':
        return '30-39'
    elif x=='2 - 3歳':
        return '2-3'
    elif x=='40代':
        return '40-49'
    elif x=='20代':
        return '20-29'
    elif x=='15 - 19歳':
        return '15-19'
    elif x=='50代':
        return '50-59'
    elif x=='60代以上':
        return '60 or above'
    elif x=='13 - 24ヶ月':
        return 'Below 2'
    else:
        return x
    
df['Age_Group']=df['Age_Group'].apply(lambda x:Age_eng(x))    
df_nfs['Age_Group']=df_nfs['Age_Group'].apply(lambda x:Age_eng(x))        


# Height 
df=df[(df['Height']!='81 - 85cm')&(df['Height']!='50cm以下')&(df['Height']!='65cm以下')&(df['Height']!='61 - 70cm')&(df['Height']!='51 - 60cm')&(df['Height']!='71 - 80cm')]
df_nfs=df_nfs[(df_nfs['Height']!='81 - 85cm')&(df_nfs['Height']!='50cm以下')&(df_nfs['Height']!='65cm以下')&(df_nfs['Height']!='61 - 70cm')&(df_nfs['Height']!='51 - 60cm')&(df_nfs['Height']!='71 - 80cm')]

def Height_impute(x):
    if x in ['91 - 100cm','81 - 90cm','96 - 100cm','91 - 95cm','86 - 90cm', '66 - 70cm','71 - 75cm']:
        return '100cm or below'
    elif x in ['181cm以上','191cm以上', '186 - 190cm','181 - 185cm']:
        return '>180 cm '
    elif x=='76 - 80cm': 
        return '176 - 180cm'
    else:
        return x
   
    
df['Height']=df['Height'].apply(lambda x: Height_impute(x))
df_nfs['Height']=df_nfs['Height'].apply(lambda x: Height_impute(x))

# Weight
df=df[(df['Weight']!='5kg以下')]
df_nfs=df_nfs[(df_nfs['Weight']!='5kg以下')]
def Weight_impute(x):
    if x =='9 - 12kg':
        return 'Below 13kg'
    else:
        return x
        
df['Weight']=df['Weight'].apply(lambda x: Weight_impute(x))
df_nfs['Weight']=df_nfs['Weight'].apply(lambda x: Weight_impute(x))

#Foot Size
def Foot_Size_impute(x):
    if x in ['21.5cm以下','22.0cm以下']:
        return 'Below 22.0cm'
    if x in ['28.0cm以上','30.0cm以上','29.0cm','28.5cm','29.5cm']:
        return 'Above 28.0cm'
    
    else:
        return x
    
df['Foot_Size']=df['Foot_Size'].apply(lambda x: Foot_Size_impute(x))

# reset_index for dataframe
df_nfs.reset_index(drop=True,inplace=True)
df.reset_index(drop=True,inplace=True)

# Create Item_Sex Columns
df['Item_Sex']=df['Item_Name'].apply(lambda x:x[:1])
df_nfs['Item_Sex']=df_nfs['Item_Name'].apply(lambda x:x[:1])

# one hot for tags

# df
Set_Tags=[]
for i in df['Tags'].unique():
    Set_Tags=Set_Tags+i.split(', ')
Set_Tags=list(set(Set_Tags))


dict_tag=dict()
list_tag=[]
for j in Set_Tags:
    dict_tag['{}'.format(j)]=[]
    list_tag.append(dict_tag['{}'.format(j)])

for i in df['Tags']:
    for j in i.split(', '):
        dict_tag['{}'.format(j)].append(1)
    for k in list_tag:
        if len(k)!=len(dict_tag['{}'.format(j)]):
            k.append(0)
df_tag=pd.DataFrame(dict_tag)
df=pd.concat([df,df_tag],axis=1)
df.drop('Tags',axis=1,inplace=True)

#df_nfs

dict_tag=dict()
list_tag=[]
for j in Set_Tags:
    dict_tag['{}'.format(j)]=[]
    list_tag.append(dict_tag['{}'.format(j)])

for i in df_nfs['Tags']:
    for j in i.split(', '):
        dict_tag['{}'.format(j)].append(1)
    for k in list_tag:
        if len(k)!=len(dict_tag['{}'.format(j)]):
            k.append(0)
df_tag=pd.DataFrame(dict_tag)
df_nfs=pd.concat([df_nfs,df_tag],axis=1)
df_nfs.drop('Tags',axis=1,inplace=True)

#st.title("UNIQLO - TOP100")

#sidebar
st.sidebar.image('logo.png')
st.sidebar.title("Size Predictor")

Gender_input = st.sidebar.selectbox("Gender:", ['Male', 'Female'])
Age_Group_input= st.sidebar.selectbox("Age Group:", ['Below 2','2-3','4-6','7-9', '10-14', '15-19',  '20-29', '30-39', '40-49',  '50-59', '60 or above' ])
Height_input=st.sidebar.selectbox("Height:", ['100cm or below','101 - 110cm','111 - 120cm', '121 - 130cm', '141 - 150cm', '131 - 140cm', '151 - 155cm', '156 - 160cm', 
       '161 - 165cm', '166 - 170cm', '171 - 175cm',       '176 - 180cm', '>180 cm '])
Weight_input=st.sidebar.selectbox("Weight:", ['Below 13kg','13 - 15kg', '16 - 20kg', '21 - 25kg', '26 - 30kg', '31 - 35kg', '36 - 40kg',
        '41 - 45kg', '46 - 50kg', '51 - 55kg', '56 - 60kg', '61 - 65kg','66 - 70kg', '71 - 75kg',  '76 - 80kg','81 - 85kg',  '86 - 90kg',  '91kg or above'])
bra_size=st.sidebar.selectbox("Under Bust and Cup Size:", ['NA', '65AA','65A','65B','65C','65D','65E','65F','70AA','70A','70B','70C','70D','70E','70F','75A','75B','75C','75D','75E','75F','80A','80B','80C','80D','80E','80F','85A','85B','85C','85D','85E','85F','90A','90B','90C','90D','90E','90F'])
foot_size_re=st.sidebar.selectbox("Foot Size:", ['Below 22.0cm', '22.0cm', '22.5cm', '23.0cm','23.5cm', '24.0cm', '24.5cm', '25.0cm',
        '25.5cm','26.0cm', '26.5cm','27.0cm', '27.5cm',  '28.0cm',  'Above 28.0cm'])
item_code_input=st.sidebar.text_input('Item Code')
if item_code_input!='':
    item_code_input=int(item_code_input)



#Bra and Shoes functions
           
def bra_size_rec(item_code_input,x):
    
    if item_code_input==445383:
        #['M', 'XL', 'L', 'S', 'XS', 'XXL']
    
        if x in ['65AA','70AA']:
            return 'XS'
        elif x in ['65A','65B','65C','70A']:
            return 'S'
        elif x in ['65D','70B','70C','70D','75A','75B']:
            return 'M'
        elif x in ['70E','75C','75D','75E','80B','80C']:
            return 'L'
        elif x in ['80D','80E','80F','85B','85C','85D']:
            return 'XL'
        elif x in ['85E','85F','80F','90B','90C','90D']:
            return 'XXL'
        else:
            return 'No suitable size yet'
    else:
        #438961
        if x in ['65AA','70AA']:
            return '65_70 AA'
        elif x in ['65A','65B','65C','70A','70B','70C']:
            return '65_70 ABC'
        elif x in ['65D','65E','65F','70D','70E','70F']:
            return '65_70 DEF'
        elif x in ['75A','75B','75C','80A','80B','80C']:
            return '75_80 ABC'
        elif x in ['75D','75E','75F','80D','80E','80F']:
            return '75_80 DEF'
        elif x in ['85A','85B','85C','90A','90B','90C']:
            return '85_90 ABC'
        elif x in ['85D','85E','85F','90D','90E','90F']:
            return '85_90 DEF'
        else:
            return 'No suitable size yet'


        
def shoes_size_rec(item_code_input,x):
    
    if item_code_input==445086:
        if x in ['22.5cm','23.0cm','23.5cm','24.0cm','24.5cm','25.0cm','25.5cm','26.0cm','26.5cm','27.0cm','27.5cm','28.0cm']:
            return re.search('[0-9]+(.5)*',x).group()
        else:
            return 'No suitable size yet'
    else:
        if x in ['22.5cm','23.0cm','23.5cm','24.0cm','24.5cm']:
            return 'RM'
        elif x in ['25.0cm','25.5cm','26.0cm','26.5cm','27.0cm','27.5cm','28.0cm']:
            return 'RL'
        elif x in ['Above 28.0cm']:
            return 'No suitable size yet'
        else:
            return 'No suitable size yet'

#initial prefill input for recommendation system

def general_rec(item_code_input,Gender_input,Age_Group_input,Height_input,Weight_input):

    #data frame for recommendation system
    df_nfs_nb=df_nfs[(df_nfs['bra']!=1)&(df_nfs['Rating']>3)]
    df_nfs_nb.drop('bra',axis=1,inplace=True)
    df_nfs_nb.drop('Rating',axis=1,inplace=True)
    df_nfs_nb.reset_index(drop=True,inplace=True)
    
    def prefill_data(item_code_input,Gender_input,Age_Group_input,Height_input,Weight_input):
        dict_input={}
        for i in df_nfs_nb.columns:
            dict_input['{}'.format(i)]=[]
        dict_input['ID'].append('PredictID')
        dict_input['Item_Code'].append(item_code_input)
        dict_input['Gender'].append(Gender_input)
        dict_input['Age_Group'].append(Age_Group_input)
        dict_input['Height'].append(Height_input)
        dict_input['Weight'].append(Weight_input)
        dict_input['Product_Size'].append('M')
        for j in ['Sum_Code', 'Item_Name', 'Dept', 'Item_Sex', 'cropped',  'easy shorts', 'chinos', 'rayon', 'room shoes', 'long', 'suw',       't-shirts', 'skinny', 'shirts', 'uv cut', 'blouse', 'outer', 'chino', '2way stretch', 'ultra light', 'oversized', 'lounge', 'bottoms',       'relaco', 'polo', 'shorts', 'short', 'legging pants', 'joggers','dry-ex', 'regular', 'cardigan', 'jeans', 'sweat', 'easy pants','ankle pants', 'trousers', 'ultra stretch', 'relaxed', 'outer,uv cut',       'nylon', 'airism cotton', 'leggings', 'jersey', 'fashion','lounge pants', 'cotton', 'formal', 'inner', 'wide', 'unisex', 'slim',       'inner bottoms', 'lounge set', 'boxers', 'airism', 'bratop','high rise', 'sneakers']:
            dict_input['{}'.format(j)].append(df_nfs_nb[df_nfs_nb['Item_Code']==item_code_input]['{}'.format(j)].unique().tolist()[0])

        return dict_input
    
    df_nfs_nb=pd.concat([df_nfs_nb,pd.DataFrame(prefill_data(item_code_input,Gender_input,Age_Group_input,Height_input,Weight_input) )],axis=0)
    df_nfs_nb.reset_index(drop=True,inplace=True)

    # get dummies
    dummies_list=df_nfs_nb[['Item_Code','Gender', 'Age_Group', 'Height', 'Weight','Sum_Code','Item_Sex','Dept']]
    df_nfs_nb_dummies=pd.get_dummies(dummies_list, prefix='',prefix_sep='')
    df_nfs_nb=pd.concat([df_nfs_nb,df_nfs_nb_dummies],axis=1)
    df_nfs_nb.drop(['Item_Code','Gender', 'Age_Group', 'Height', 'Weight','Sum_Code','Item_Sex','Dept'],axis=1,inplace=True)
    df_nfs_nb.reset_index(drop=True,inplace=True)
    df_nfs_nb.set_index('ID',inplace=True)

    df_Model=df_nfs_nb.copy()
    df_Model.drop(['Product_Size','Item_Name'],axis=1,inplace=True)
    
    
    def size_predictor(distance_method, ID, N):
        # create dataframe used to store distances between recipes
        df_distance = pd.DataFrame(data=df_Model.index)

        # remove rows where index is equal to the inputted recipe_id
        df_distance = df_distance[df_Model.index != ID]

        # add a distance column that states the inputted recipe's distance with every other recipe
        df_distance['distance'] = df_distance['ID'].apply(lambda x: distance_method(df_Model.loc[ID],df_Model.loc[x]))

        # sort the allRecipes by distance and take N closes number of rows to put in the TopNRecommendation as the recommendations
        df_distance.sort_values(by='distance',inplace=True)

        # for each recipe in TopNRecommendation, input to defined lists

        # return dataframe with the inputted recipe and the recommended recipe's normalized nutritional values
        return(df_distance.head(N))

    size_rec_id=[i for i in size_predictor(cosine, 'PredictID', 10)['ID']]

    size_rec=[]

    for i in size_rec_id:
        size_rec.append(df_nfs_nb.loc[[i]]['Product_Size'][0])
        
    def most_frequent(List):
        return max(set(List), key = List.count)


    return most_frequent(size_rec)

# Whole function
def size_recommender(item_code_input,Gender_input,Age_Group_input,Height_input,Weight_input):
    item_code_input=int(item_code_input)
    if df_nfs[df_nfs['Item_Code']==item_code_input]['bra'].unique()==1:
        return bra_size_rec(item_code_input,bra_size)
    elif (df[df['Item_Code']==item_code_input]['room shoes'].unique()==1)or(df[df['Item_Code']==item_code_input]['sneakers'].unique()==1):
        return shoes_size_rec(item_code_input,foot_size_re)   
    else:
        return general_rec(item_code_input,Gender_input,Age_Group_input,Height_input,Weight_input)
        
#img

df_img=pd.read_csv('Sample_img.csv')
df_img.drop_duplicates(subset=['item_code'],inplace=True)
df_img.reset_index(drop=True,inplace=True)
item_list=pd.read_csv('item_list.csv')
df_img['Item_Name']=df_img['item_code'].apply(lambda x:item_list[item_list['item code']==x]['item name'].unique()[0])
df_img['Item_Sex']=df_img['Item_Name'].apply(lambda x:x[:1])

df_kids = df_img[df_img['Item_Sex'].isin(['K','G'])]
df_kids.reset_index(inplace=True)
df_kids.drop(columns=['index','Unnamed: 0'],inplace=True)

df_women = df_img[df_img['Item_Sex']=='W']
df_women.reset_index(inplace=True)
df_women.drop(columns=['index','Unnamed: 0'],inplace=True)

df_men = df_img[df_img['Item_Sex']=='M']
df_men.reset_index(inplace=True)
df_men.drop(columns=['index','Unnamed: 0'],inplace=True)

def create_tab(df_tab):
    for n in range(0,len(df_tab),4):
        a,b,c,d = st.columns(4)
        with a:
            st.image(df_tab.iloc[n,1])
            st.caption(df_tab.iloc[n,5])
            st.caption(f'Item code: {df_tab.iloc[n,0]}')
        with b:
            try:
                st.image(df_tab.iloc[n+1,1])
                st.caption(df_tab.iloc[n+1,5])
                st.caption(f'Item code: {df_tab.iloc[n+1,0]}')
            except:
                pass
        with c:
            try:
                st.image(df_tab.iloc[n+2,1])
                st.caption(df_tab.iloc[n+2,5])
                st.caption(f'Item code: {df_tab.iloc[n+2,0]}')
            except:
                pass
        with d:
            try:
                st.image(df_tab.iloc[n+3,1])
                st.caption(df_tab.iloc[n+3,5])
                st.caption(f'Item code: {df_tab.iloc[n+3,0]}')
            except:
                pass
        st.write('\n')
        st.write('\n')
        st.write('\n')


if item_code_input == '':
    tab = st.selectbox('Please select interested category: ',('WOMEN','MEN','KIDS'))
    if tab == 'WOMEN':
        create_tab(df_women)    
    elif tab == 'MEN':
        create_tab(df_men)
    else:
        create_tab(df_kids)



if item_code_input!='':
    try:
        predicted_size=size_recommender(item_code_input,Gender_input,Age_Group_input,Height_input,Weight_input)    
        st.header((df_nfs[df_nfs['Item_Code']==int(item_code_input)]['Item_Name'].unique()[0]).upper())
        st.caption(f'Item code: {item_code_input}')
        st.header('Your suggested size is:')
        no_size = ['No suitable size yet','4XL']
        sneaker_size = ['23','24','25','26','27','28']
        m_duplicated_size = ['27','28','29','30']
        m_pants = [447652,439177,439176,444591,445086]

        col9, col10 = st.columns(2)
        with col9:
            if predicted_size in no_size:
                st.header('Sorry, no suitable size is available')
            elif item_code_input==445086 and (predicted_size in sneaker_size):
                size_path = './size/S'+predicted_size+'.png'
                st.image(size_path)
            elif item_code_input in m_pants and (predicted_size in m_duplicated_size):
                size_path = './size/M'+predicted_size+'.png'
                st.image(size_path)
            elif predicted_size == '2XL':
                size_path = './size/XXL.png'
                st.image(size_path)
            else:
                size_path = './size/'+predicted_size+'.png'
                st.image(size_path)
        with col10:
            link = 'https://www.uniqlo.com.hk/en_GB/search.html?description='+str(item_code_input)
            #@st.cache(allow_output_mutation=True)
            def get_base64_of_bin_file(bin_file):
                with open(bin_file, 'rb') as f:
                    data = f.read()
                return base64.b64encode(data).decode()

            #@st.cache(allow_output_mutation=True)
            def get_img_with_href(local_img_path, target_url):
                img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
                bin_str = get_base64_of_bin_file(local_img_path)
                html_code = f'''
                    <a href="{target_url}">
                        <img src="data:image/{img_format};base64,{bin_str}" />
                    </a>'''
                return html_code

            gif_html = get_img_with_href('buy.jpg', link)
            st.markdown(gif_html, unsafe_allow_html=True)

        st.write('\n')    
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(df_img[df_img['item_code']==int(item_code_input)]['img_1'].unique()[0])
        with col2:
            st.image(df_img[df_img['item_code']==int(item_code_input)]['img_2'].unique()[0])
        with col3:
            st.image(df_img[df_img['item_code']==int(item_code_input)]['img_3'].unique()[0])
        with col4:
            st.image(df_img[df_img['item_code']==int(item_code_input)]['img_4'].unique()[0])
    except:
        st.error('Please double check item code')


# product recommender

def product_recommender(item_code_input,Gender_input,Age_Group_input,Height_input,Weight_input,predicted_size):
    item_code_input=int(item_code_input)
    df_nfs_rec=df_nfs[(df_nfs['Rating']>3)]
    df_nfs_rec.drop('Rating',axis=1,inplace=True)
    
    df_nfs_rec.reset_index(drop=True,inplace=True)
    
    
    def prefill_data(item_code_input,Gender_input,Age_Group_input,Height_input,Weight_input,predicted_size):
        dict_input={}
        for i in df_nfs_rec.columns:
            dict_input['{}'.format(i)]=[]
        dict_input['ID'].append('PredictID')
        dict_input['Item_Code'].append(item_code_input)
        dict_input['Gender'].append(Gender_input)
        dict_input['Age_Group'].append(Age_Group_input)
        dict_input['Height'].append(Height_input)
        dict_input['Weight'].append(Weight_input)
        dict_input['Product_Size'].append(predicted_size)
        
        for j in ['Sum_Code', 'Item_Name', 'Dept', 'Item_Sex',
       'easy shorts', 'leggings', 'jersey', 'trousers', 'suw', 'airism',
       'regular', 'outer,uv cut', 'slim', 'fashion', 'lounge pants',
       '2way stretch', 'long', 'cardigan', 'lounge set', 'sneakers', 'joggers',
       'lounge', 'chinos', 'relaxed', 'skinny', 'dry-ex', 'short', 'unisex',
       't-shirts', 'ultra stretch', 'high rise', 'bratop', 'cotton', 'nylon',
       'room shoes', 'blouse', 'bra', 'formal', 'inner bottoms', 'shorts',
       'relaco', 'easy pants', 'shirts', 'airism cotton', 'chino',
       'ultra light', 'outer', 'boxers', 'oversized', 'wide', 'cropped',
       'uv cut', 'ankle pants', 'sweat', 'polo', 'bottoms', 'rayon', 'jeans',
       'legging pants', 'inner']:
            dict_input['{}'.format(j)].append(df_nfs_rec[df_nfs_rec['Item_Code']==item_code_input]['{}'.format(j)].unique().tolist()[0])

        return dict_input
    
    df_nfs_rec=pd.concat([df_nfs_rec,pd.DataFrame(prefill_data(item_code_input,Gender_input,Age_Group_input,Height_input,Weight_input,predicted_size) )],axis=0)
    df_nfs_rec.reset_index(drop=True,inplace=True)
    
    # get dummies
    dummies_list=df_nfs_rec[['Gender', 'Age_Group', 'Height', 'Weight','Item_Sex','Dept']]
    df_nfs_rec_dummies=pd.get_dummies(dummies_list, prefix='',prefix_sep='')
    df_nfs_rec=pd.concat([df_nfs_rec,df_nfs_rec_dummies],axis=1)
    df_pre_row=df_nfs_rec[(df_nfs_rec['ID']=='PredictID')].copy()
    df_nfs_rec=df_nfs_rec[(df_nfs_rec['Item_Code']!=item_code_input)]
    df_nfs_rec=pd.concat([df_nfs_rec,df_pre_row],axis=0)
    
    
    df_nfs_rec.drop(['Item_Code','Gender', 'Age_Group', 'Height', 'Weight','Sum_Code','Item_Sex','Dept'],axis=1,inplace=True)
    
    df_nfs_rec.reset_index(drop=True,inplace=True)
    df_nfs_rec.set_index('ID',inplace=True)

    df_Model=df_nfs_rec.copy()
    df_Model.drop(['Product_Size','Item_Name'],axis=1,inplace=True)
    
    
    
    def product_predictor(distance_method, ID, N):
        # create dataframe used to store distances between recipes
        df_distance = pd.DataFrame(data=df_Model.index)

        # remove rows where index is equal to the inputted recipe_id
        df_distance = df_distance[df_Model.index != ID]

        # add a distance column that states the inputted recipe's distance with every other recipe
        df_distance['distance'] = df_distance['ID'].apply(lambda x: distance_method(df_Model.loc[ID],df_Model.loc[x]))

        # sort the allRecipes by distance and take N closes number of rows to put in the TopNRecommendation as the recommendations
        df_distance.sort_values(by='distance',inplace=True)

        # for each recipe in TopNRecommendation, input to defined lists

        # return dataframe with the inputted recipe and the recommended recipe's normalized nutritional values
        return(df_distance.head(N))

    product_rec_id=[i for i in product_predictor(cosine, 'PredictID', 20)['ID']]
    
    product_rec=[]
    df_nfs.set_index('ID',inplace=True)
    for i in product_rec_id:
        product_rec.append(df_nfs.loc[[i]]['Item_Code'][0])
     
    #def most_frequent(List):
        #return max(set(List), key = List.count)

    #return most_frequent(product_rec)

    products = Counter(product_rec)
    rec2 = products.most_common(2)
    return rec2




if item_code_input!='':

    st.write('\n')
    st.header('You may also like:')
    
    predicted_product=product_recommender(item_code_input,Gender_input,Age_Group_input,Height_input,Weight_input,predicted_size) 

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.image(df_img[df_img['item_code']==predicted_product[0][0]]['img_1'].unique()[0])
        st.caption((df_nfs[df_nfs['Item_Code']==predicted_product[0][0]]['Item_Name'][0]).upper())
        #st.caption(f'Item code: {predicted_product[0][0]}')    
    with col6:
        st.image(df_img[df_img['item_code']==predicted_product[0][0]]['img_2'].unique()[0])
        #st.text((df_nfs[df_nfs['Item_Code']==predicted_product[0][0]]['Item_Name'][0]).upper())
        st.caption(f'Item code: {predicted_product[0][0]}') 
    with col7:
        try:
            st.image(df_img[df_img['item_code']==predicted_product[1][0]]['img_1'].unique()[0])
            st.caption((df_nfs[df_nfs['Item_Code']==predicted_product[1][0]]['Item_Name'][0]).upper())
        except:
            st.image(df_img[df_img['item_code']==predicted_product[0][0]]['img_3'].unique()[0])
            #st.caption(f'Item code: {predicted_product[1][0]}') 
    with col8:
        try:
            st.image(df_img[df_img['item_code']==predicted_product[1][0]]['img_2'].unique()[0])
            #st.text((df_nfs[df_nfs['Item_Code']==predicted_product[1][0]]['Item_Name'][0]).upper())
            st.caption(f'Item code: {predicted_product[1][0]}')
        except:
            st.image(df_img[df_img['item_code']==predicted_product[0][0]]['img_4'].unique()[0])


   
    st.write('\n')    
    st.header('Recently viewed')
        
    # save input
    try:
        dict_input={'Item_Code':item_code_input,'Searched_Product_Name':df_nfs[df_nfs['Item_Code']==item_code_input]['Item_Name'].unique()[0],'Gender':Gender_input,'Age_Group':Age_Group_input,'Height':Height_input,'Weight':Weight_input,'Bra_Size':bra_size,'Foot_Size':foot_size_re,'Predicted_Size':predicted_size,'Rec_P1_Code':predicted_product[0][0],'Rec_P1_Name':df_nfs[df_nfs['Item_Code']==predicted_product[0][0]]['Item_Name'][0],'Rec_P2_Code':predicted_product[1][0],'Rec_P2_Name':df_nfs[df_nfs['Item_Code']==predicted_product[1][0]]['Item_Name'][0]}
    except:
        dict_input={'Item_Code':item_code_input,'Searched_Product_Name':df_nfs[df_nfs['Item_Code']==item_code_input]['Item_Name'].unique()[0],'Gender':Gender_input,'Age_Group':Age_Group_input,'Height':Height_input,'Weight':Weight_input,'Bra_Size':bra_size,'Foot_Size':foot_size_re,'Predicted_Size':predicted_size,'Rec_P1_Code':predicted_product[0][0],'Rec_P1_Name':df_nfs[df_nfs['Item_Code']==predicted_product[0][0]]['Item_Name'][0],'Rec_P2_Code':'NA','Rec_P2_Name':'NA'}
    df_input_new=pd.DataFrame(dict_input, index=[0])
    #df_input_new.to_csv('Input_Data.csv',index=False)

    try:
        df_input=pd.read_csv('Input_Data.csv')

        if sum(sum([k==v for (k,v) in zip(df_input.iloc[-1:,2:6].to_numpy(),df_input_new.iloc[-1:,2:6].to_numpy())]))==4:
            df_input_new['customer_id']=df_input['customer_id'].iloc[-1]
        else:
            df_input_new['customer_id']=df_input['customer_id'].iloc[-1]+1
        df_input=pd.concat([df_input,df_input_new],axis=0)
        df_input.reset_index(drop=True, inplace=True)
        df_input.to_csv('Input_Data.csv',index=False)

    except:
        df_input_new['customer_id']=1
        df_input_new.to_csv('Input_Data.csv',index=False)
    try:
        his_list=df_input[df_input['customer_id']==df_input_new['customer_id'].iloc[-1]]['Item_Code']
        his_list=his_list.tolist()
        his_list=list(set(his_list))
        his_list.remove(item_code_input)
        his_list.reverse()
        for i in range(0,len(set(his_list)),4):
            #if len(set(his_list))>=4:
            cols = st.columns(4)
            count=0
            for j in range(i,i+4):
                with cols[count]:
                    try:
                        st.image(df_img[df_img['item_code']==his_list[j]]['img_1'].unique()[0])
                        st.caption((df_nfs[df_nfs['Item_Code']==his_list[j]]['Item_Name'][0]).upper())
                        st.caption('Item code: {}'.format(his_list[j]))  
                        st.caption('Suggested Size: {}'.format(df_input[(df_input['customer_id']==df_input_new['customer_id'].iloc[-1])&(df_input['Item_Code']==int(his_list[j]))]['Predicted_Size'].unique()[0]))   
                    except:
                        st.text(' ')
                count+=1
            
    except:
        pass



    st.write('\n')
    st.text("To go back to product listing page, please delete the item code on sidebar and press 'ENTER'")
