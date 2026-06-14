import re

def extract_answer(text):
    # Try \boxed{} first
    boxed_match = re.search(r'\\boxed\{([^}]*)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Fallback to "The answer is"
    ans_match = re.search(r'[Tt]he answer is[:\s]+([^\s\.]+)', text)
    if ans_match:
        return ans_match.group(1).strip()

    # Fallback to last numeric value
    nums = re.findall(r'[-+]?\d*\.?\d+', text)
    if nums:
        return nums[-1]

    return text.strip()

def is_correct(pred, target, tolerance=1e-5):
    pred = str(pred).strip()
    target = str(target).strip()
    if pred == target:
        return True
    try:
        p_val = float(pred)
        t_val = float(target)
        return abs(p_val - t_val) <= tolerance * max(abs(t_val), 1e-9)
    except:
        return False

# Test
test_texts = [
    "The result is \\boxed{123}.",
    "Final answer: \\boxed{10.01}",
    "The answer is 42",
    "It is 3.14159",
    "The bitstring is 10101010"
]
targets = ["123", "10.01", "42", "3.14159", "10101010"]

for txt, tgt in zip(test_texts, targets):
    extracted = extract_answer(txt)
    print(f"Text: {txt} -> Extracted: {extracted} | Correct: {is_correct(extracted, tgt)}")
