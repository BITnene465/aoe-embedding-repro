
import unittest
from transformers import AutoTokenizer
from aoe.data import AngleDataCollator

class TestAngleDataCollator(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def test_format_d_auto_detection(self):
        # Case 1: No prompt provided -> Should detect Format D (raw text)
        collator = AngleDataCollator(tokenizer=self.tokenizer, text_prompt=None)
        features = [{"text1": "hello", "text2": "world", "score": 1.0}]
        
        # We can't directly inspect dataset_format before call, but we can check behavior
        # However, the collator sets self.dataset_format on first call if None.
        
        # Let's mock sample_from_list to spy or just check output tokens if possible.
        # But easier: check if prompts are applied.
        
        batch = collator(features)
        # BERT adds [CLS] hello [SEP] world [SEP] ... wait, AngleDataCollator tokenizes list of texts.
        # It flattens pairs: [text1, text2]
        # So input_ids[0] is "hello", input_ids[1] is "world"
        
        decoded_0 = self.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
        decoded_1 = self.tokenizer.decode(batch["input_ids"][1], skip_special_tokens=True)
        
        self.assertEqual(decoded_0, "hello")
        self.assertEqual(decoded_1, "world")
        self.assertEqual(collator.dataset_format, "D")

    def test_format_a_auto_detection(self):
        # Case 2: Prompt provided -> Should detect Format A
        prompt = "Summarize: {text}"
        collator = AngleDataCollator(tokenizer=self.tokenizer, text_prompt=prompt)
        features = [{"text1": "hello", "text2": "world", "score": 1.0}]
        
        batch = collator(features)
        decoded_0 = self.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
        
        self.assertEqual(decoded_0, "summarize : hello")
        self.assertEqual(collator.dataset_format, "A")

if __name__ == "__main__":
    unittest.main()
