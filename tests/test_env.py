"""Environment and dependencies verification."""
import pytest

def test_imports():
    import torch
    import sentence_transformers
    import chromadb
    import sklearn
    import xgboost
    import fastapi
    import streamlit
    import rank_bm25
    import pypdf
    import datasets
    
    assert torch.__version__ is not None
    assert chromadb.__version__ is not None
    assert sklearn.__version__ is not None
    print("\n[OK] All core packages imported successfully!")

if __name__ == "__main__":
    test_imports()
