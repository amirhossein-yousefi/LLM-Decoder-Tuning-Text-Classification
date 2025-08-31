import pandas as pd
from llm_cls.data import build_text_column

def test_build_text_column():
    df = pd.DataFrame({"a": ["x", ""], "b": [None, "y"]})
    out = build_text_column(df, ["a", "b"], "text")
    assert out.loc[0, "text"].endswith(".")
    assert out.loc[1, "text"] == "y."
