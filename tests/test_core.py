from app.core import generate_baskets,train_recommender,recommend_next,probe_itemsets,model_size_bytes,quantize_dynamic,quantize_static,make_calibration_data
import torch

def test_baskets_are_reproducible():
    assert generate_baskets(10,7)==generate_baskets(10,7)

def test_training_and_recommendations():
    baskets=generate_baskets(120,3); result=train_recommender(baskets,epochs=2,hidden=16,seed=3)
    assert result.examples>0 and result.train_loss>0
    recs=recommend_next(result.model,["bread"],result.items,3)
    assert len(recs)<=3 and all(name!="bread" for name,_ in recs)

def test_itemset_support():
    pairs=probe_itemsets([["a","b"],["a","b"],["a","c"]],.5)
    assert pairs[0]["items"]==("a","b") and pairs[0]["support"]==2/3

def test_dynamic_quantization_smoke():
    result=train_recommender(generate_baskets(100,5),epochs=2,hidden=16)
    quantized=quantize_dynamic(result.model)
    assert model_size_bytes(quantized)>0
    assert len(recommend_next(quantized,[],result.items,3))==3

def test_static_quantization_smoke():
    baskets=generate_baskets(100,9); result=train_recommender(baskets,epochs=2,hidden=16)
    q=quantize_static(result.model,make_calibration_data(baskets,result.items))
    assert model_size_bytes(q)>0
    assert torch.isfinite(q(torch.zeros(1,len(result.items)))).all()
