import matplotlib.pyplot as plt
import pandas as pd

data = {
    "method": ["baseline", "kv_cache", "4_bit", "8_bit"],
    "memory_usage": [639539200, 550231040, 191598592, 235212288]
}

df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(8,6))
bars = plt.bar(df["method"], df["memory_usage"], color="skyblue")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, f'{yval:.2f} MB', ha='center', va='bottom')

plt.xlabel("Method")
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Usage by Method")
plt.show()

data = {
    "method": ["baseline", "kv_cache", "4_bit", "8_bit"],
    "tokens_per_sec": [74.00000494003329, 74.842427870128, 54.18991114868213, 25.327017295033897]
}

df = pd.DataFrame(data)

plt.figure(figsize=(8,6))
bars = plt.bar(df["method"], df["tokens_per_sec"], color="lightgreen")


for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.2f} tokens/sec', ha='center', va='bottom')

plt.xlabel("Method")
plt.ylabel("Tokens per Second")
plt.title("Tokens per Second by Method")
plt.show()
