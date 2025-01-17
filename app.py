import time
import sys
import streamlit as st
import string
from io import StringIO 
import pdb
import json
from twc_embeddings import HFModel,SimCSEModel,SGPTModel,CausalLMModel,SGPTQnAModel
from twc_openai_embeddings import OpenAIModel
from twc_clustering import TWCClustering
import torch
import requests
import socket


MAX_INPUT = 5000

SEM_SIMILARITY="1"
DOC_RETRIEVAL="2"
CLUSTERING="3"


use_case = {"1":"Finding similar phrases/sentences","2":"Retrieving semantically matching information to a query. It may not be a factual match","3":"Clustering"}
use_case_url = {"1":"https://huggingface.co/spaces/taskswithcode/semantic_similarity","2":"https://huggingface.co/spaces/taskswithcode/semantic_search","3":""}



from transformers import BertTokenizer, BertForMaskedLM


APP_NAME = "twc/semantic_clustering"
INFO_URL = "https://www.taskswithcode.com/stats/"



        

def get_views(action):
    ret_val = 0
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    if ("view_count" not in st.session_state):
        try:
           app_info = {'name': APP_NAME,"action":action,"host":hostname,"ip":ip_address}
           res = requests.post(INFO_URL, json = app_info).json()
           print(res)
           data = res["count"]
        except:
           data = 0
        ret_val = data
        st.session_state["view_count"] = data
    else:
        ret_val = st.session_state["view_count"]
        if (action != "init"):
           app_info = {'name': APP_NAME,"action":action,"host":hostname,"ip":ip_address}
           res = requests.post(INFO_URL, json = app_info).json()
    return "{:,}".format(ret_val)
        



def construct_model_info_for_display(model_names):
    options_arr  = []
    markdown_str = f"<div style=\"font-size:16px; color: #2f2f2f; text-align: left\"><br/><b>Models evaluated ({len(model_names)})</b><br/><i>The selected models satisfy one or more of the following (1) state-of-the-art (2) the most downloaded models on Hugging Face (3) Large Language Models (e.g. GPT-3)</i></div>"
    markdown_str += f"<div style=\"font-size:2px; color: #2f2f2f; text-align: left\"><br/></div>"
    for node in model_names:
        options_arr .append(node["name"])
        if (node["mark"] == "True"):
            markdown_str += f"<div style=\"font-size:16px; color: #5f5f5f; text-align: left\">&nbsp;•&nbsp;Model:&nbsp;<a href=\'{node['paper_url']}\' target='_blank'>{node['name']}</a><br/>&nbsp;&nbsp;&nbsp;&nbsp;Code released by:&nbsp;<a href=\'{node['orig_author_url']}\' target='_blank'>{node['orig_author']}</a><br/>&nbsp;&nbsp;&nbsp;&nbsp;Model info:&nbsp;<a href=\'{node['sota_info']['sota_link']}\' target='_blank'>{node['sota_info']['task']}</a></div>"
            if ("Note" in node):
                markdown_str += f"<div style=\"font-size:16px; color: #a91212; text-align: left\">&nbsp;&nbsp;&nbsp;&nbsp;{node['Note']}<a href=\'{node['alt_url']}\' target='_blank'>link</a></div>"
            markdown_str += "<div style=\"font-size:16px; color: #5f5f5f; text-align: left\"><br/></div>"
        
    markdown_str += "<div style=\"font-size:12px; color: #9f9f9f; text-align: left\"><b>Note:</b><br/>•&nbsp;Uploaded files are loaded into non-persistent memory for the duration of the computation. They are not cached</div>"
    limit = "{:,}".format(MAX_INPUT)
    markdown_str += f"<div style=\"font-size:12px; color: #9f9f9f; text-align: left\">•&nbsp;User uploaded file has a maximum limit of {limit} sentences.</div>"
    return options_arr,markdown_str


st.set_page_config(page_title='TasksWihCode', page_icon="logo.png", layout='centered', initial_sidebar_state='auto',
            menu_items={
             'About': 'This app was created by taskswithcode. http://taskswithcode.com'
             
              })
col,pad = st.columns([85,15])

with col:
    st.image("long_form_logo_with_icon.png")


@st.experimental_memo
def load_model(model_name,model_class,load_model_name):
    try:
        ret_model = None
        obj_class = globals()[model_class]
        ret_model = obj_class()
        ret_model.init_model(load_model_name)
        assert(ret_model is not None)
    except Exception as e:
        st.error(f"Unable to load model class:{model_class} model_name: {model_name} load_model_name: {load_model_name}   {str(e)}")
        pass
    return ret_model


  
@st.experimental_memo
def cached_compute_similarity(input_file_name,sentences,_model,model_name,threshold,_cluster,clustering_type):
    texts,embeddings = _model.compute_embeddings(input_file_name,sentences,is_file=False)
    results = _cluster.cluster(None,texts,embeddings,threshold,clustering_type)
    return results


def uncached_compute_similarity(input_file_name,sentences,_model,model_name,threshold,cluster,clustering_type):
    with st.spinner('Computing vectors for sentences'):
        texts,embeddings = _model.compute_embeddings(input_file_name,sentences,is_file=False)
        results = cluster.cluster(None,texts,embeddings,threshold,clustering_type)
    #st.success("Similarity computation complete")
    return results

DEFAULT_HF_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
def get_model_info(model_names,model_name):
    for node in model_names:
        if (model_name == node["name"]):
            return node,model_name
    return get_model_info(model_names,DEFAULT_HF_MODEL)


def run_test(model_names,model_name,input_file_name,sentences,display_area,threshold,user_uploaded,custom_model,clustering_type):
    display_area.text("Loading model:" + model_name)
    #Note. model_name may get mapped to new name in the call below for custom models
    orig_model_name = model_name
    model_info,model_name = get_model_info(model_names,model_name)
    if (model_name != orig_model_name):
        load_model_name  = orig_model_name
    else:
        load_model_name = model_info["model"]
    if ("Note" in model_info):
        fail_link = f"{model_info['Note']} [link]({model_info['alt_url']})"
        display_area.write(fail_link)
    if (user_uploaded and "custom_load" in model_info and model_info["custom_load"] == "False"):
        fail_link = f"{model_info['Note']} [link]({model_info['alt_url']})"
        display_area.write(fail_link)
        return {"error":fail_link}
    model = load_model(model_name,model_info["class"],load_model_name)
    display_area.text("Model " + model_name  + " load complete")
    try:
            if (user_uploaded):
                results = uncached_compute_similarity(input_file_name,sentences,model,model_name,threshold,st.session_state["cluster"],clustering_type)
            else:
                display_area.text("Computing vectors for sentences")
                results = cached_compute_similarity(input_file_name,sentences,model,model_name,threshold,st.session_state["cluster"],clustering_type)
                display_area.text("Similarity computation complete")
            return results
            
    except Exception as e:
        st.error("Some error occurred during prediction" + str(e))
        st.stop()
    return {}



    

def display_results(orig_sentences,results,response_info,app_mode,model_name):
    main_sent = f"<div style=\"font-size:14px; color: #2f2f2f; text-align: left\">{response_info}<br/><br/></div>"
    main_sent += f"<div style=\"font-size:14px; color: #2f2f2f; text-align: left\">Showing results for model:&nbsp;<b>{model_name}</b></div>"
    score_text = "cosine distance"
    main_sent += f"<div style=\"font-size:14px; color: #6f6f6f; text-align: left\">Clustering by {score_text}.&nbsp;<b>{len(results['clusters'])} clusters</b>.&nbsp;&nbsp;mean:{results['info']['mean']:.2f};&nbsp;std:{results['info']['std']:.2f};&nbsp;current threshold:{results['info']['current_threshold']}<br/>Threshold hints:{str(results['info']['zscores'])}<br/>Overlap stats(overlap,freq):{str(results['info']['overlap'])}</div>"
    body_sent = []
    download_data = {}
    for i in range(len(results["clusters"])):
        pivot_index = results["clusters"][i]["pivot_index"]
        pivot_sent = orig_sentences[pivot_index]
        pivot_index +=  1
        d_cluster = {}
        download_data[i + 1] = d_cluster
        d_cluster["pivot"] = {"pivot_index":pivot_index,"sent":pivot_sent,"children":{}}
        body_sent.append(f"<div style=\"font-size:16px; color: #2f2f2f; text-align: left\">{pivot_index}]&nbsp;{pivot_sent}&nbsp;<b><i>(Cluster {i+1})</i></b>&nbsp;&nbsp;</div>")
        neighs_dict = results["clusters"][i]["neighs"]
        for key in neighs_dict:
            cosine_dist = neighs_dict[key]
            child_index = key
            sentence = orig_sentences[child_index]
            child_index += 1
            body_sent.append(f"<div style=\"font-size:16px; color: #2f2f2f; text-align: left\">{child_index}]&nbsp;{sentence}&nbsp;&nbsp;&nbsp;<b>{cosine_dist:.2f}</b></div>")
            d_cluster["pivot"]["children"][sentence] = f"{cosine_dist:.2f}" 
        body_sent.append(f"<div style=\"font-size:16px; color: #2f2f2f; text-align: left\">&nbsp;</div>")
    main_sent = main_sent + "\n" + '\n'.join(body_sent)
    st.markdown(main_sent,unsafe_allow_html=True)
    st.session_state["download_ready"] = json.dumps(download_data,indent=4)
    get_views("submit")


def init_session():
    if ("model_name" not in st.session_state):
        st.session_state["model_name"] = "ss_test"
        st.session_state["download_ready"] = None    
        st.session_state["model_name"] = "ss_test"
        st.session_state["threshold"] = 1.5
        st.session_state["file_name"] = "default"
        st.session_state["overlapped"] = "overlapped"
        st.session_state["cluster"] = TWCClustering()
    else:
        print("Skipping init session")
 
def app_main(app_mode,example_files,model_name_files,clus_types):
  init_session()
  with open(example_files) as fp:
        example_file_names = json.load(fp) 
  with open(model_name_files) as fp:
        model_names = json.load(fp)
  with open(clus_types) as fp:
        cluster_types = json.load(fp)
  curr_use_case = use_case[app_mode].split(".")[0]
  st.markdown("<h5 style='text-align: center;'>Compare popular/state-of-the-art models for semantic clustering using sentence embeddings</h5>", unsafe_allow_html=True)
  st.markdown(f"<p style='font-size:14px; color: #4f4f4f; text-align: center'><i>Or compare your own model with state-of-the-art/popular models</p>", unsafe_allow_html=True)
  st.markdown(f"<div style='color: #4f4f4f; text-align: left'>Use cases for sentence embeddings<br/>&nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;<a href=\'{use_case_url['1']}\' target='_blank'>{use_case['1']}</a><br/>&nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;<a href=\'{use_case_url['2']}\' target='_blank'>{use_case['2']}</a><br/>&nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;{use_case['3']}<br/><i>This app illustrates <b>'{curr_use_case}'</b> use case</i></div>", unsafe_allow_html=True)
  st.markdown(f"<div style='color: #9f9f9f; text-align: right'>views:&nbsp;{get_views('init')}</div>", unsafe_allow_html=True)


  try:
      
      
      with st.form('twc_form'):

        step1_line = "Upload text file(one sentence in a line) or choose an example text file below"
        if (app_mode ==  DOC_RETRIEVAL):
            step1_line += ". The first line is treated as the query"
        uploaded_file = st.file_uploader(step1_line, type=".txt")

        selected_file_index = st.selectbox(label=f'Example files ({len(example_file_names)})',  
                    options = list(dict.keys(example_file_names)), index=0,  key = "twc_file")
        st.write("")
        options_arr,markdown_str = construct_model_info_for_display(model_names)
        selection_label = 'Select Model'
        selected_model = st.selectbox(label=selection_label,  
                    options = options_arr, index=0,  key = "twc_model")
        st.write("")
        custom_model_selection = st.text_input("Model not listed above? Type any Hugging Face sentence embedding model name ", "",key="custom_model")
        hf_link_str = "<div style=\"font-size:12px; color: #9f9f9f; text-align: left\"><a href='https://huggingface.co/models?pipeline_tag=sentence-similarity' target = '_blank'>List of Hugging Face sentence embedding models</a><br/><br/><br/></div>"
        st.markdown(hf_link_str, unsafe_allow_html=True)
        threshold = st.number_input('Choose a zscore threshold (number of std devs from mean)',value=st.session_state["threshold"],min_value = 0.0,step=.01)
        st.write("")
        clustering_type = st.selectbox(label=f'Select type of clustering',  
                    options = list(dict.keys(cluster_types)), index=0,  key = "twc_cluster_types")
        st.write("")
        submit_button = st.form_submit_button('Run')

        
        input_status_area = st.empty()
        display_area = st.empty()
        if submit_button:
            start = time.time()
            if uploaded_file is not None:
                st.session_state["file_name"]  = uploaded_file.name
                sentences = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            else:
                st.session_state["file_name"]  = example_file_names[selected_file_index]["name"]
                sentences = open(example_file_names[selected_file_index]["name"]).read()
            sentences = sentences.split("\n")[:-1]
            if (len(sentences) > MAX_INPUT):
                st.info(f"Input sentence count exceeds maximum sentence limit. First {MAX_INPUT} out of {len(sentences)} sentences chosen")
                sentences = sentences[:MAX_INPUT]
            if (len(custom_model_selection) != 0):
                run_model = custom_model_selection
            else:
                run_model = selected_model
            st.session_state["model_name"] = selected_model
            st.session_state["threshold"] = threshold
            st.session_state["overlapped"] = cluster_types[clustering_type]["type"]
            results = run_test(model_names,run_model,st.session_state["file_name"],sentences,display_area,threshold,(uploaded_file is not None),(len(custom_model_selection) != 0),cluster_types[clustering_type]["type"])
            display_area.empty()
            with display_area.container():
                if ("error" in results):
                    st.error(results["error"])
                else:
                    device = 'GPU' if torch.cuda.is_available() else 'CPU'
                    response_info = f"Computation time on {device}: {time.time() - start:.2f} secs for {len(sentences)} sentences"
                    if (len(custom_model_selection) != 0):
                        st.info("Custom model overrides model selection in step 2 above. So please clear the custom model text box to choose models from step 2")
                    display_results(sentences,results,response_info,app_mode,run_model)
                    #st.json(results)
      st.download_button(
         label="Download results as json",
         data= st.session_state["download_ready"] if st.session_state["download_ready"] != None else "",
         disabled = False if st.session_state["download_ready"] != None else True,
         file_name= (st.session_state["model_name"] + "_" +  str(st.session_state["threshold"]) + "_" + st.session_state["overlapped"] + "_" +  '_'.join(st.session_state["file_name"].split(".")[:-1]) + ".json").replace("/","_"),
         mime='text/json',
         key ="download" 
        )
      
      

  except Exception as e:
    st.error("Some error occurred during loading" + str(e))
    st.stop()  
	
  st.markdown(markdown_str, unsafe_allow_html=True)
  
 

if __name__ == "__main__":
   #print("comand line input:",len(sys.argv),str(sys.argv))
   #app_main(sys.argv[1],sys.argv[2],sys.argv[3])
   #app_main("1","sim_app_examples.json","sim_app_models.json")
   app_main("3","clus_app_examples.json","clus_app_models.json","clus_app_clustypes.json")

