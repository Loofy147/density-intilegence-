import random
import pandas as pd
import string

def generate_gravity(num_samples=1500):
    samples = []
    for _ in range(num_samples):
        g = round(random.uniform(5.0, 30.0), 2)
        num_examples = random.randint(3, 5)
        examples = []
        for _ in range(num_examples):
            t = round(random.uniform(1.0, 5.0), 2)
            d = round(0.5 * g * t**2, 2)
            examples.append(f"For t = {t}s, distance = {d} m")

        target_t = round(random.uniform(1.0, 5.0), 2)
        answer = round(0.5 * g * target_t**2, 2)

        prompt = "In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:\n"
        prompt += "\n".join(examples)
        prompt += f"\nNow, determine the falling distance for t = {target_t}s given d = 0.5*g*t^2."

        samples.append({"prompt": prompt, "answer": str(answer)})
    return samples

def generate_unit_conversion(num_samples=1500):
    samples = []
    for _ in range(num_samples):
        ratio = round(random.uniform(0.5, 2.5), 2)
        num_examples = random.randint(3, 5)
        examples = []
        for _ in range(num_examples):
            val = round(random.uniform(5.0, 50.0), 2)
            converted = round(val * ratio, 2)
            examples.append(f"{val} m becomes {converted:.2f}")

        target_val = round(random.uniform(5.0, 50.0), 2)
        answer = f"{target_val * ratio:.2f}"

        prompt = "In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:\n"
        prompt += "\n".join(examples)
        prompt += f"\nNow, convert the following measurement: {target_val} m"

        samples.append({"prompt": prompt, "answer": answer})
    return samples

def int_to_roman(n):
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syb = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ""
    i = 0
    while n > 0:
        for _ in range(n // val[i]):
            roman_num += syb[i]
            n -= val[i]
        i += 1
    return roman_num

def generate_numeral_system(num_samples=1500):
    samples = []
    for _ in range(num_samples):
        num_examples = random.randint(3, 5)
        used_nums = random.sample(range(1, 4000), num_examples + 1)
        examples = []
        for i in range(num_examples):
            examples.append(f"{used_nums[i]} -> {int_to_roman(used_nums[i])}")

        target_num = used_nums[-1]
        answer = int_to_roman(target_num)

        prompt = "In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:\n"
        prompt += "\n".join(examples)
        prompt += f"\nNow, write the number {target_num} in the Wonderland numeral system."

        samples.append({"prompt": prompt, "answer": answer})
    return samples

def generate_text_encryption(num_samples=1500):
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "alice", "wonderland", "queen", "king", "knight", "dragon", "castle", "magic", "secret", "crystal", "forest", "mountain"]
    samples = []
    for _ in range(num_samples):
        shuffled_alphabet = list(string.ascii_lowercase)
        random.shuffle(shuffled_alphabet)
        mapping = dict(zip(string.ascii_lowercase, shuffled_alphabet))

        def encrypt(text):
            return "".join(mapping.get(c, c) for c in text)

        num_examples = random.randint(3, 5)
        examples = []
        for _ in range(num_examples):
            sentence_words = random.sample(words, random.randint(3, 5))
            sentence = " ".join(sentence_words)
            examples.append(f"{encrypt(sentence)} -> {sentence}")

        target_words = random.sample(words, random.randint(2, 4))
        target_sentence = " ".join(target_words)
        encrypted_target = encrypt(target_sentence)

        prompt = "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:\n"
        prompt += "\n".join(examples)
        prompt += f"\nNow, decrypt the following text: {encrypted_target}"

        samples.append({"prompt": prompt, "answer": target_sentence})
    return samples

def generate_bit_manipulation(num_samples=1500):
    samples = []
    for _ in range(num_samples):
        # Use more complex rules: XOR + rotation
        mask = random.randint(0, 255)
        rot = random.randint(0, 7)

        def transform(n):
            n = n ^ mask
            return ((n << rot) | (n >> (8 - rot))) & 0xFF

        num_examples = random.randint(6, 10)
        examples = []
        for _ in range(num_examples):
            inp = random.randint(0, 255)
            out = transform(inp)
            examples.append(f"{bin(inp)[2:].zfill(8)} -> {bin(out)[2:].zfill(8)}")

        target_inp = random.randint(0, 255)
        answer = bin(transform(target_inp))[2:].zfill(8)

        prompt = "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.\n\nHere are some examples of input -> output:\n"
        prompt += "\n".join(examples)
        prompt += f"\n\nNow, determine the output for: {bin(target_inp)[2:].zfill(8)}"

        samples.append({"prompt": prompt, "answer": answer})
    return samples

all_samples = []
all_samples.extend(generate_gravity())
all_samples.extend(generate_unit_conversion())
all_samples.extend(generate_numeral_system())
all_samples.extend(generate_text_encryption())
all_samples.extend(generate_bit_manipulation())

df_synthetic = pd.DataFrame(all_samples)
df_synthetic.to_csv('data/synthetic_train.csv', index=False)
print(f"Generated {len(df_synthetic)} synthetic samples.")
