
import sys
import os

# Add project root to path （DEV）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer
from aoe.data import AngleDataCollator, Prompts
from aoe.model import SentenceEncoder
from aoe.loss import cosine_loss, contrastive_with_negative_loss, angle_loss

def test_data_collator():
    print("Testing AngleDataCollator...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    collator = AngleDataCollator(
        tokenizer=tokenizer,
        text_prompt=Prompts.A,
        dataset_format="A"
    )
    
    features = [
        {"text1": "hello", "text2": "world", "score": 1.0},
        {"text1": "foo", "text2": "bar", "score": 0.0}
    ]
    
    batch = collator(features)
    print("Batch keys:", batch.keys())
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Labels shape:", batch["labels"].shape)
    
    # Check if prompts are applied (roughly)
    # "Summarize sentence "hello" in one word:" -> tokenized length should be > len("hello")
    # "hello" is 1 token. Prompt is ~8 tokens.
    assert batch["input_ids"].shape[1] > 5, "Prompts might not be applied correctly"
    print("AngleDataCollator test passed!")

def test_pooling():
    print("\nTesting Pooling (cls_avg)...")
    encoder = SentenceEncoder("bert-base-uncased", pooling="cls_avg")
    texts = ["hello world", "foo bar"]
    
    # Run encode
    emb = encoder.encode(texts)
    print("Embedding shape:", emb.shape)
    assert emb.shape == (2, 768)
    print("Pooling test passed!")

def test_losses():
    print("\nTesting Losses...")
    # Dummy data
    y_true = torch.tensor([1.0, 1.0, 0.0, 0.0]) # zigzag: pair1 (1.0), pair2 (0.0)
    y_pred = torch.randn(4, 768) # 2 pairs, 4 sentences
    
    loss_cos = cosine_loss(y_true, y_pred)
    print("Cosine Loss:", loss_cos.item())
    
    # Angle loss
    # Angle loss expects complex input (2D) or we need to mock it
    # The current angle_loss implementation in loss.py expects [batch, feat]
    # If feat is even, it splits it.
    loss_angle = angle_loss(y_true, y_pred)
    print("Angle Loss:", loss_angle.item())
    
    print("Losses test passed!")

if __name__ == "__main__":
    test_data_collator()
    test_pooling()
    test_losses()
