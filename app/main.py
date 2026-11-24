"""Streamlit dashboard for training and quantization experiments."""
import pandas as pd
import streamlit as st
from app.core import generate_baskets,train_recommender,recommend_next,probe_itemsets,model_size_bytes,quantize_dynamic,quantize_static,make_calibration_data,evaluate_model

st.set_page_config(page_title="Quantized Retail Recommender",page_icon="🛒",layout="wide")
st.title("🛒 Quantized Retail Recommender")
st.caption("Train a compact next-item model on synthetic shopping baskets and compare post-training quantization.")
with st.sidebar:
    st.header("Experiment settings")
    n=st.slider("Synthetic baskets",300,4000,1600,100)
    epochs=st.slider("Training epochs",3,50,18)
    hidden=st.select_slider("Hidden layer width",[16,32,64,96,128],value=64)
    seed=st.number_input("Random seed",0,99999,42)
if st.button("Generate data and train",type="primary"):
    with st.spinner("Training model and quantizing weights..."):
        baskets=generate_baskets(n,int(seed))
        result=train_recommender(baskets,epochs,hidden,int(seed))
        calibration=make_calibration_data(baskets,result.items)
        st.session_state["experiment"]={"baskets":baskets,"result":result,"calibration":calibration,
          "dynamic":quantize_dynamic(result.model),"static":quantize_static(result.model,calibration)}
if "experiment" not in st.session_state:
    st.info("Choose settings and train the model to start.")
    st.stop()
e=st.session_state["experiment"]; result=e["result"]; baskets=e["baskets"]
models=[("FP32 baseline",result.model),("Dynamic INT8",e["dynamic"]),("Static INT8",e["static"])]
base=model_size_bytes(result.model); rows=[]
for label,model in models:
    try: acc=evaluate_model(model,baskets,result.items)
    except Exception as exc: acc=float("nan"); st.warning(f"{label} evaluation failed: {exc}")
    size=model_size_bytes(model)
    rows.append({"Model":label,"Size (KB)":round(size/1024,2),"Accuracy":acc,"Size reduction":f"{(1-size/base)*100:.1f}%"})
a,b,c,d=st.columns(4)
a.metric("Training examples",f"{result.examples:,}"); b.metric("Validation accuracy",f"{result.accuracy:.1%}")
c.metric("FP32 model size",f"{base/1024:.1f} KB"); d.metric("Catalog products",len(result.items))
st.subheader("Quantization comparison")
st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
left,right=st.columns(2)
with left:
    st.subheader("Next-item suggestions")
    history=st.multiselect("Items already in basket",result.items,default=[x for x in ["bread","milk"] if x in result.items])
    k=st.slider("Number of suggestions",1,10,5)
    st.dataframe(pd.DataFrame(recommend_next(result.model,history,result.items,k),columns=["Suggested item","Score"]),hide_index=True)
with right:
    st.subheader("Frequent itemset probing")
    support=st.slider("Minimum pair support",.01,.5,.08,.01)
    pairs=probe_itemsets(baskets,support)
    st.dataframe(pd.DataFrame([{"Item pair":" + ".join(p["items"]),"Support":p["support"]} for p in pairs[:20]]),use_container_width=True,hide_index=True)
st.subheader("Sample synthetic baskets")
st.dataframe(pd.DataFrame({"Basket":[" · ".join(x) for x in baskets[:12]]}),use_container_width=True,hide_index=True)
