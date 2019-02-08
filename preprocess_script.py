import os
import pandas as pd


# f = open(action_csv,'r')
# # train_lines = f.read().splitlines()
# df = pd.read_csv(csv_file)
# df.columns = ['1','2','3','4','5','6','7','8']
#
# ac_id_dic = get_action_dic()
# # print ac_idf
# actions = f.read().splitlines()
# action_list = set()
# for action in actions[1:]:
# 		tags = action.split(',')
# 		tags = tags[:-1]
# 		ac_id = int(tags[0])
# 		ac = ''.join(tags[1:])
# 		ac = ac.replace('"','')
# 		action_list.add(ac)
#
# print action_list
# final_df = pd.DataFrame()
# ac_vid = {}
# vids_list = []
# for ac in tqdm(action_list):
# 	# print ac
# 	id = ac_id_dic[ac]
# 	vids_df = df.loc[df['7'] == id]
# 	vid_name = vids_df.iloc[1,0]
# 	vids_list.append(vid_name)
# 	vid_df = df.loc[df['1']== vid_name]
#
# 	final_df = final_df.append(vid_df,ignore_index=True)
# 	df = df.drop(df[df['1']==vids_df.iloc[1,0]].index)
#
# final_df['2'] = final_df['2'].apply(lambda x: str(x).zfill(4))
#
# final_df.to_csv(csv_file_subset, header = False, index = False, float_format='%.3f')





def form_multi_data():
    dataset = open('/home/subha/hoi_vid/keras-kinetics-i3d/data/ava/ava_data_subset_new.txt','r')
    f = open('/home/subha/hoi_vid/keras-kinetics-i3d/data/ava/ava_data_subset_multi.txt','w+')
    # df = pd.read_csv(dataset)
    # df.columns = ['1','2','3','4','5','6']
    lines = dataset.read().splitlines()
    while lines:
        # print len(lines)

        ann = lines[0]
        tags =ann.split(',')
        bbx = ','.join(tags[:-1])
        anns = [l for l in lines if bbx in l]
        # print anns
        [lines.remove(l) for l in anns]
        # [f.write(l+'\n') for l in anns]
        for a in anns:
            ac = a.split(',')[-1]
            bbx = bbx+','+ac

        f.write(bbx+'\n')
        # print bbx

form_multi_data()
