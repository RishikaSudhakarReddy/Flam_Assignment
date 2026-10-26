"""
tests/test_pipeline.py - Comprehensive Unit and Integration Test Suite.
Verifies all components of the Indic Corpus Curation pipeline.
"""

import unittest
import tempfile
import json
import gzip
from pathlib import Path

from src.normalizer import (
    normalize_text,
    normalize_unicode,
    normalize_devanagari_nukta,
    clean_invisible_chars,
    normalize_for_comparison
)
from src.lang_id import identify_language
from src.filter import filter_quality, is_legitimate_unusual_text
from src.pii import redact_pii, validate_verhoeff
from src.dedup import Deduplicator, get_shingles, exact_jaccard
from src.contamination import ContaminationChecker, transliterate_devanagari_to_latin
from src.reader import StreamingReader


class TestNormalizer(unittest.TestCase):
    def test_nukta_canonicalization(self):
        # Decomposed Ka + Nukta -> Precomposed Qa (\u0958)
        decomposed = "\u0915\u093Cदम"
        expected = "\u0958\u0926\u092e"
        self.assertEqual(normalize_devanagari_nukta(decomposed), expected)

    def test_invisible_characters(self):
        # Text with Zero-Width Space (ZWSP), BOM, and directional marks
        messy = "\uFEFFयह\u200B एक\u200E वाक्य\u200F है।"
        cleaned = clean_invisible_chars(messy)
        self.assertEqual(cleaned, "यह एक वाक्य है।")

    def test_pipe_to_danda(self):
        text = "यह भारत देश है |"
        norm = normalize_text(text)
        self.assertTrue("।" in norm)

    def test_whitespace_and_newlines(self):
        text = "  पहला   शब्द \r\n\r\n\r\n\r\nदूसरा   शब्द  "
        norm = normalize_text(text)
        self.assertEqual(norm, "पहला शब्द\n\nदूसरा शब्द")


class TestLanguageIdentifier(unittest.TestCase):
    def test_devanagari_hindi(self):
        self.assertEqual(identify_language("भारत एक विशाल और सुंदर देश है।"), "hi")

    def test_code_mixed_devanagari(self):
        # Requirement: "hi : Hindi in Devanagari, including Devanagari text with English words mixed in"
        self.assertEqual(identify_language("मैं Python और Machine Learning सीख रहा हूँ।"), "hi")

    def test_hinglish_latin(self):
        self.assertEqual(identify_language("bhai kal office me bohot kaam tha, tu kab aa raha hai?"), "hi-Latn")
        self.assertEqual(identify_language("Zomato pe refund kab tak aayega please batao?"), "hi-Latn")

    def test_english_and_other(self):
        self.assertEqual(identify_language("The quick brown fox jumps over the lazy dog."), "en or other")
        self.assertEqual(identify_language("El rápido zorro marrón salta sobre el perro perezoso."), "en or other")
        self.assertEqual(identify_language("1234567890 !@#$%^&*()"), "en or other")


class TestQualityFilter(unittest.TestCase):
    def test_proverbs_preserved(self):
        proverb1 = "नाच न जाने आँगन टेढ़ा।"
        proverb2 = "जल ही जीवन है।"
        self.assertTrue(is_legitimate_unusual_text(proverb1))
        passed1, reason1 = filter_quality(proverb1)
        self.assertTrue(passed1, f"Proverb falsely dropped: {reason1}")
        passed2, reason2 = filter_quality(proverb2)
        self.assertTrue(passed2, f"Short proverb falsely dropped: {reason2}")

    def test_poetry_preserved(self):
        poem = (
            "लहरों से डर कर नौका पार नहीं होती,\n"
            "कोशिश करने वालों की कभी हार नहीं होती।\n"
            "नन्हीं चींटी जब दाना लेकर चलती है,\n"
            "चढ़ती दीवारों पर, सौ बार फिसलती है।"
        )
        passed, reason = filter_quality(poem)
        self.assertTrue(passed, f"Poetry falsely dropped: {reason}")

    def test_numeric_tables_preserved(self):
        table = (
            "विराट कोहली 76 (59) 6 चौके 2 छक्के\n"
            "रोहित शर्मा 47 (31) 7 चौके 1 छक्का\n"
            "अक्षर पटेल 47 (31) 1 चौका 4 छक्के"
        )
        passed, reason = filter_quality(table)
        self.assertTrue(passed, f"Cricket table falsely dropped: {reason}")

    def test_spam_rejected(self):
        spam = "बहुत बहुत बहुत बहुत बहुत बहुत aaaaaaaaaaaaaaa ??????????"
        passed, reason = filter_quality(spam)
        self.assertFalse(passed)
        self.assertIn("spam", reason)

    def test_mojibake_rejected(self):
        mojibake = "Ã¢â‚¬â„¢ à¤¹à¤¿à¤‚à¤¦à¥€ \ufffd\ufffd\ufffd broken text"
        passed, reason = filter_quality(mojibake)
        self.assertFalse(passed)
        self.assertIn("garbled", reason)


class TestPIIRedaction(unittest.TestCase):
    def test_verhoeff(self):
        # Valid sample Aadhaar passing Verhoeff check (check digit 6)
        self.assertTrue(validate_verhoeff("367598346016"))
        # Invalid checksum
        self.assertFalse(validate_verhoeff("367598346013"))
        # Starts with 0 or 1
        self.assertFalse(validate_verhoeff("012345678901"))

    def test_pii_replacement(self):
        text = (
            "Contact info: Rahul Sharma, email: rahul.sharma@example.com, "
            "mobile: +91 98765 43210, Aadhaar: 3675 9834 6016, PAN: ABCDE1234F."
        )
        redacted, stats = redact_pii(text)
        self.assertIn("<EMAIL>", redacted)
        self.assertIn("<PHONE>", redacted)
        self.assertIn("<AADHAAR>", redacted)
        self.assertIn("<PAN>", redacted)
        self.assertEqual(stats["email"], 1)
        self.assertEqual(stats["phone"], 1)
        self.assertEqual(stats["aadhaar"], 1)
        self.assertEqual(stats["pan"], 1)

    def test_non_pii_numbers_survive(self):
        text = "In year 2024, at pincode 110001, we sold 10,000 units for 5000 INR with score 47/2."
        redacted, stats = redact_pii(text)
        self.assertEqual(redacted, text)
        self.assertEqual(stats["total"], 0)


class TestDeduplication(unittest.TestCase):
    def test_exact_and_near_dedup(self):
        dedup = Deduplicator(near_threshold=0.80)
        
        doc1 = "यह भारत देश का एक बहुत ही महत्वपूर्ण और ऐतिहासिक दस्तावेज है जिसमें प्राचीन समृद्ध संस्कृति का वर्णन किया गया है।"
        doc2 = "यह भारत देश का एक बहुत ही महत्वपूर्ण और ऐतिहासिक दस्तावेज है जिसमें प्राचीन समृद्ध संस्कृति का वर्णन किया गया है।" # exact
        doc3 = "यह भारत देश का एक बहुत ही महत्वपूर्ण और ऐतिहासिक दस्तावेज है जिसमें प्राचीन समृद्ध संस्कृति का विस्तृत वर्णन किया गया है।" # near
        doc4 = "आज शाम को बाहर का मौसम बहुत सुहावना है और तेज ठंडी हवा के साथ बारिश हो रही है।" # distinct
        
        # Doc 1
        self.assertIsNone(dedup.check_exact("d1", doc1))
        near_match, _ = dedup.check_near("d1", doc1)
        self.assertIsNone(near_match)
        
        # Doc 2 (exact duplicate of d1)
        exact_match = dedup.check_exact("d2", doc2)
        self.assertEqual(exact_match, "d1")
        
        # Doc 3 (near duplicate of d1)
        self.assertIsNone(dedup.check_exact("d3", doc3))
        near_match, sim = dedup.check_near("d3", doc3)
        self.assertEqual(near_match, "d1")
        self.assertGreaterEqual(sim, 0.80)
        
        # Doc 4 (unique)
        self.assertIsNone(dedup.check_exact("d4", doc4))
        near_match, _ = dedup.check_near("d4", doc4)
        self.assertIsNone(near_match)


class TestContamination(unittest.TestCase):
    def test_transliteration_and_cross_script(self):
        devanagari = "नमस्ते भारत आप कैसे हैं"
        trans = transliterate_devanagari_to_latin(devanagari)
        self.assertIn("namaste", trans)
        self.assertIn("bharat", trans)
        
        eval_items = [{
            "id": "eval_001",
            "text": "नमस्ते भारत आप कैसे हैं",
            "lang": "hi"
        }]
        checker = ContaminationChecker(eval_items)
        
        # Matching Latin text
        latin_leak = "namaste bharat aap kaise hain"
        res = checker.check_training_record("train_101", latin_leak, latin_leak, lang="hi-Latn")
        self.assertIsNotNone(res)
        eval_id, method = res
        self.assertEqual(eval_id, "eval_001")
        self.assertEqual(method, "cross_script_transliteration")


class TestFaultTolerantReader(unittest.TestCase):
    def test_survives_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_data.jsonl"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write('{"id": "doc1", "text": "Valid document"}\n')
                f.write('MALFORMED_JSON_LINE_WITHOUT_BRACES\n')
                f.write('{"id": "doc2", "text": "Another valid document"}\n')
                
            reader = StreamingReader(tmpdir)
            records = list(reader.iterate_records())
            
            self.assertEqual(len(records), 3)
            # Row 1: valid
            self.assertTrue(records[0][1])
            # Row 2: malformed
            self.assertFalse(records[1][1])
            self.assertIn("malformed_json_syntax", records[1][2])
            # Row 3: valid
            self.assertTrue(records[2][1])


if __name__ == "__main__":
    unittest.main()
