from scipy.spatial.distance import cosine
import argparse
import json
import pdb
import torch
import torch.nn.functional as F
import numpy as np
import time
from collections import OrderedDict


class TWCClustering:
    def __init__(self):
        print("In Zscore  Clustering")

    def compute_matrix(self,embeddings):
        #print("Computing similarity matrix ...)")
        embeddings= np.array(embeddings)
        start = time.time()
        vec_a = embeddings.T #vec_a shape (1024,)
        vec_a = vec_a/np.linalg.norm(vec_a,axis=0) #Norm is along axis 0 - rows
        vec_a = vec_a.T #vec_a shape becomes (,1024)
        similarity_matrix = np.inner(vec_a,vec_a)
        end = time.time()
        time_val = (end-start)*1000
        #print(f"Similarity matrix computation complete. Time taken:{(time_val/(1000*60)):.2f}  minutes")
        return similarity_matrix
        
    def get_terms_above_threshold(self,matrix,embeddings,pivot_index,threshold):
        run_index = pivot_index
        picked_arr = []
        while (run_index < len(embeddings)):
            if (matrix[pivot_index][run_index] >= threshold):
                picked_arr.append(run_index)
            run_index += 1
        return picked_arr

    def update_picked_dict_arr(self,picked_dict,arr):
        for i in range(len(arr)):
            picked_dict[arr[i]] = 1

    def update_picked_dict(self,picked_dict,in_dict):
        for key in in_dict:
            picked_dict[key] = 1

    def find_pivot_subgraph(self,pivot_index,arr,matrix,threshold,strict_cluster = True):
        center_index = pivot_index
        center_score = 0
        center_dict = {}
        for i in range(len(arr)):
            node_i_index = arr[i]
            running_score = 0
            temp_dict = {}
            for j in range(len(arr)):
                node_j_index = arr[j]
                cosine_dist = matrix[node_i_index][node_j_index]
                if ((cosine_dist < threshold) and strict_cluster):
                    continue
                running_score += cosine_dist
                temp_dict[node_j_index] = cosine_dist
            if (running_score > center_score):
                center_index = node_i_index
                center_dict = temp_dict
                center_score = running_score
        sorted_d = OrderedDict(sorted(center_dict.items(), key=lambda kv: kv[1], reverse=True))
        return  {"pivot_index":center_index,"orig_index":pivot_index,"neighs":sorted_d}
         

    def update_overlap_stats(self,overlap_dict,cluster_info):
        arr = list(cluster_info["neighs"].keys())
        for val in arr:
            if (val not in overlap_dict):
                overlap_dict[val] = 1
            else:
                overlap_dict[val] += 1

    def bucket_overlap(self,overlap_dict):
        bucket_dict = {}
        for key in overlap_dict:
            if (overlap_dict[key] not in bucket_dict):
                bucket_dict[overlap_dict[key]] = 1
            else:
                bucket_dict[overlap_dict[key]] += 1
        sorted_d = OrderedDict(sorted(bucket_dict.items(), key=lambda kv: kv[1], reverse=False))
        return sorted_d

    def merge_clusters(self,ref_cluster,curr_cluster):
        dup_arr = ref_cluster.copy()
        for j in range(len(curr_cluster)):
            if (curr_cluster[j] not in dup_arr):
                ref_cluster.append(curr_cluster[j]) 
                

    def non_overlapped_clustering(self,matrix,embeddings,threshold,mean,std,cluster_dict):
        picked_dict = {}
        overlap_dict = {}
        candidates = []
    
        for i in range(len(embeddings)):
            if (i in picked_dict):
                continue
            zscore = mean + threshold*std
            arr = self.get_terms_above_threshold(matrix,embeddings,i,zscore)
            candidates.append(arr)
            self.update_picked_dict_arr(picked_dict,arr)
    
        # Merge arrays to create non-overlapping sets
        run_index_i = 0
        while (run_index_i < len(candidates)):
            ref_cluster = candidates[run_index_i]
            run_index_j = run_index_i + 1
            found = False
            while (run_index_j < len(candidates)): 
                curr_cluster = candidates[run_index_j]
                for k in range(len(curr_cluster)):
                    if (curr_cluster[k] in ref_cluster):
                        self.merge_clusters(ref_cluster,curr_cluster)
                        candidates.pop(run_index_j)
                        found = True
                        run_index_i = 0
                        break
                if (found):
                    break
                else:
                    run_index_j += 1
            if (not found):
                run_index_i += 1 
            
                
        zscore = mean + threshold*std
        for i in range(len(candidates)):
            arr = candidates[i]
            cluster_info = self.find_pivot_subgraph(arr[0],arr,matrix,zscore,strict_cluster = False)
            cluster_dict["clusters"].append(cluster_info)
        return  {}

    def overlapped_clustering(self,matrix,embeddings,threshold,mean,std,cluster_dict):
        picked_dict = {}
        overlap_dict = {}
    
        zscore = mean + threshold*std
        for i in range(len(embeddings)):
            if (i in picked_dict):
                continue
            arr = self.get_terms_above_threshold(matrix,embeddings,i,zscore)
            cluster_info = self.find_pivot_subgraph(i,arr,matrix,zscore,strict_cluster = True)
            self.update_picked_dict(picked_dict,cluster_info["neighs"])
            self.update_overlap_stats(overlap_dict,cluster_info)
            cluster_dict["clusters"].append(cluster_info)
        sorted_d = self.bucket_overlap(overlap_dict)
        return  sorted_d
        
        
    def cluster(self,output_file,texts,embeddings,threshold,clustering_type):
        is_overlapped = True if clustering_type == "overlapped" else False
        matrix = self.compute_matrix(embeddings)
        mean = np.mean(matrix)
        std = np.std(matrix)
        zscores = []
        inc = 0
        value = mean
        while (value < 1):
            zscores.append({"threshold":inc,"cosine":round(value,2)})
            inc += 1
            value = mean + inc*std
        #print("In clustering:",round(std,2),zscores)
        cluster_dict = {}
        cluster_dict["clusters"] = []
        if (is_overlapped):
            sorted_d = self.overlapped_clustering(matrix,embeddings,threshold,mean,std,cluster_dict) 
        else:
            sorted_d = self.non_overlapped_clustering(matrix,embeddings,threshold,mean,std,cluster_dict) 
        curr_threshold = f"{threshold} (cosine:{mean+threshold*std:.2f})"
        cluster_dict["info"] ={"mean":mean,"std":std,"current_threshold":curr_threshold,"zscores":zscores,"overlap":list(sorted_d.items())}
        return cluster_dict


