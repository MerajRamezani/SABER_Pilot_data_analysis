import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats as ss
import json
import pathlib
from pycytominer import aggregate, normalize

# Load barcodes used in experiment
guide_df = pd.read_csv("SABER_Library_ngt_Included_Oligo_Sequences_Assiged.csv")
all_guides_list = list(guide_df.iloc[:,4].unique())

count_df = pd.read_csv('HeLa_SABER_screen_SABER_arm_cell_count.csv.gz')
count_df = count_df.rename(columns={'Metadata_Foci_Barcode_MatchedTo_GeneCode':'gene','Metadata_Foci_Barcode_MatchedTo_Barcode':'barcode'})
count_df['n_guides'] = np.nan

count_df_1000 = count_df.query('Cell_Count > 1000')
all_guides_list_1000 = list(count_df_1000.barcode.unique())
for guide in all_guides_list_1000:
    gene = list(count_df_1000.loc[count_df_1000['barcode']== guide,'gene'])[0]
    df_temp = count_df_1000.query('gene == @gene')
    count_df_1000.loc[count_df_1000['barcode']== guide,'n_guides'] = len(df_temp)
    
count_df_1000_4_guides = count_df_1000.query('n_guides >= 4')
selected_genes_list = list(count_df_1000_4_guides.gene.unique())
selected_guides_list = list(count_df_1000_4_guides.barcode.unique())

cp498_guide_profiles_df = pd.read_csv('20240202_6W_CP498_SABER_Pilot_HeLa_SABER_only_guide_normalized_merged_feature_select_median_ALLWELLS.csv.gz')
selected_features = list(cp498_guide_profiles_df.columns)

# Load and subset the single-cell profiles per plate for M059K plates
plates = ['SABER_Plate_1','SABER_Plate_2','SABER_Plate_4']
chunksize = 10 ** 5

df_plate_list = []
for plate in plates:
    filename = f'20240202_6W_CP498_SABER_Pilot_HeLa_SABER_only_single_cell_normalized_ALLBATCHES___{plate}___ALLWELLS.csv.gz' 
    chunks = []
    with pd.read_csv(filename ,usecols=selected_features ,chunksize=chunksize) as reader:
        for chunk in reader:
            sub_chunk = chunk[chunk['Metadata_Foci_Barcode_MatchedTo_GeneCode'].isin(selected_genes_list)]
            chunks.append(sub_chunk)
            print(chunk.shape,f'of plate {plate}')
            print(sub_chunk.shape,f'of plate {plate}')
    df = pd.concat(chunks)
    df_plate_list.append(df)

single_cell_df = pd.concat(df_plate_list)

subsample_ns = [100,200,300,400,500,750,1000]
subsample_dfs_dictionary = {} 
for n in subsample_ns:
    df_list_n = []
    for guide in selected_guides_list:
        df_temp = single_cell_df.query('Metadata_Foci_Barcode_MatchedTo_Barcode == @guide')
        if len(df_temp)<n:
            continue
        df_temp_sample = df_temp.sample(n=n,random_state=42)
        df_list_n.append(df_temp_sample)
    df_n = pd.concat(df_list_n)
    aggregate_df_n = aggregate(
        population_df=df_n,
        strata=['Metadata_Foci_Barcode_MatchedTo_GeneCode','Metadata_Foci_Barcode_MatchedTo_Barcode'],
        features='infer',
        operation='median'
    )
    subsample_dfs_dictionary[n]=aggregate_df_n
    aggregate_df_n.to_csv(f'CP498_SABER_arm_subsampled_aggregated_{n}_cells.csv',index=False)
    print(f'profiles at {n} representation level are aggregated.')

def cosine_to_df(df_temp, cosine_array, i):
    cosine_list = cosine_array[i]
    gene_list = list(df_temp.index)
    cosine_df = pd.DataFrame(index=gene_list)
    cosine_df['cosine'] = cosine_list
    cosine_df = cosine_df.sort_values('cosine',ascending=False)   
    return cosine_df

def ap_from_cosine_df(cosine_df,gene,n=10):    
    #print(cosine_df.iloc[:20])
    index_list = list(cosine_df.index)
    boolean = [1 if  i == gene else 0 for i in index_list ]
    grades_list=[]
    for i in range(2,n+2):
        pre_grade = sum(boolean[1:i])/(i-1)
        grades_list.append(pre_grade*boolean[i-1])
    return sum(grades_list)/3

def calculate_map(df_guide, gene):
    df_temp = df_guide.query("Metadata_Foci_Barcode_MatchedTo_GeneCode == 'nontargeting' | Metadata_Foci_Barcode_MatchedTo_GeneCode == @gene")
    df_temp = df_temp.drop(['Metadata_Foci_Barcode_MatchedTo_Barcode'],axis=1)
    df_temp = df_temp.set_index("Metadata_Foci_Barcode_MatchedTo_GeneCode")
    #print(df_temp)
    ap_list = []
    cosine_array = cosine_similarity(df_temp)
    for guide in range(4):
        cosine_df = cosine_to_df(df_temp, cosine_array, guide)
        #print(cosine_df[:10])
        guide_ap = ap_from_cosine_df(cosine_df,gene,10)
        ap_list.append(guide_ap)
    return np.mean(ap_list)

# calculate the mAP values
rep_results = {}
for n in subsample_ns:
    genes_list = list(subsample_dfs_dictionary[n].Metadata_Foci_Barcode_MatchedTo_GeneCode.unique())
    map_list = []
    for i in range(len(genes_list)):
        gene = genes_list[i]
        gene_map = calculate_map(subsample_dfs_dictionary[n], gene)
        map_list.append([gene, gene_map])
    rep_results[n] = map_list
ys = []
for n in subsample_ns:
    y = [i[1] for i in rep_results[n] if i[1]]
    ys.append(np.mean(y))

mpl.rc('axes', linewidth=0.7)
mpl.rc('ytick', labelsize=12)
mpl.rc('xtick', labelsize=12)

fig, ax = plt.subplots(figsize=(7,5))

ax = sns.lineplot(x=subsample_ns, y=ys, errorbar=None, style=1, markers=True, legend=False)
ax.set_title('',size=14)
ax.set_xlabel('Guide level representation',size=12)
ax.set_ylabel('Average mAP',size=12)
ax.set_xticks(range(0,1001,100))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig('CP498_SABER_arm_subsample_average_mAP.png',dpi = 300,bbox_inches='tight')
plt.show()