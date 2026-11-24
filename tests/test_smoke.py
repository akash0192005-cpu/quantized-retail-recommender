from app.core import generate_baskets,train_recommender

def test_end_to_end_smoke():
    result=train_recommender(generate_baskets(80,11),epochs=1,hidden=16,seed=11)
    assert result.items and 0<=result.accuracy<=1
