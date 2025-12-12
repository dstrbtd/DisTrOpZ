import os
import json
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

STARTING_LOSS = 10.80

# -------- Load JSON --------
# Save your data as, e.g., results.json
with open(
    "/root/DisTrOpZ/results/metrics-gpt-small-owt-2-10000-2025-12-06.json", "r"
) as f:
    # with open("/root/DisTrOpZ/results/metrics-gpt-base-owt-2-1000-2025-12-09.json", "r") as f:
    data = json.load(f)

new_data = {}

for key in data.keys():
    data[key]["loss"] = STARTING_LOSS - data[key]["loss"]
    if key == "5EvvqR8EJhQYVyk6avp2dpkLymR95StUqPoRSSN7sD9FUSWj":
        new_data["SparseLoCo"] = data[key]
    elif key == "5EEqeZe2KmWTHKRr48xZNgfDXZCJScfTMvt2daoMxKz1Zifw":
        new_data["DiLoCo"] = data[key]
    elif key == "5HW6iTCNfk9xRmNbFv7PKGpJL99JU2wzco4ABJxywKZGgjJA":
        new_data["DeMo"] = data[key]
        path = "/root/DisTrOpZ/miner/miner_demo.py"
    elif key == "5EvFbREcHj3gC9tRCbQ5E4CF25UCAVsJj4pFyzFqHrbgn9Rg":
        new_data["MuLoCo"] = data[key]
        path = "/root/DisTrOpZ/miner/miner_muloco.py"

methods = list(new_data.keys())

throughput = np.array(
    [new_data[m]["throughput"] for m in methods if "throughput" in new_data[m].keys()],
    dtype=float,
)
loss = np.array(
    [new_data[m]["loss"] for m in methods if "loss" in new_data[m].keys()], dtype=float
)
communication = np.array(
    [
        new_data[m]["communication"]
        for m in methods
        if "communication" in new_data[m].keys()
    ],
    dtype=float,
)

# -------- Bubble sizes (scaled by throughput) --------
bubble_size = (throughput / throughput.max()) * 1200

# -------- Plot --------
plt.figure(figsize=(8, 6))

plt.scatter(
    communication,
    loss,
    s=bubble_size,
    marker="o",  # circles
    alpha=0.6,
    edgecolors="none",
)

# for x, y, name in zip(communication, loss, methods):
#     plt.text(x, y, name, fontsize=10, ha="left", va="bottom")

# plt.xlabel("Communication Volume (Bytes)")
# plt.ylabel("Loss Reduction")
# plt.title("Loss Reduction vs Communication Volume — Bubble Size = Throughput")
# plt.grid(True)

# # Optional: uncomment this if you want a log x-axis
# # plt.xscale("log")

# plt.tight_layout()
# plt.show()
# plt.savefig(f"{os.path.dirname(os.path.realpath(__file__))}/bubble_plot.png", dpi=200)

# # -------- Plot --------
# plt.figure(figsize=(8, 6))

# plt.scatter(
#     communication,
#     loss,
#     s=bubble_size,
#     marker="o",
#     alpha=0.6,
#     edgecolors="none"
# )

# texts = []
# for x, y, name in zip(communication, loss, methods):
#     # start labels near the points
#     t = plt.text(x, y, name, fontsize=10)
#     texts.append(t)

# # Let adjustText move labels so they don't overlap
# adjust_text(
#     texts,
#     expand_points=(1.2, 1.2),
#     arrowprops=dict(arrowstyle="-", lw=0.5)  # draws subtle connectors
# )

# plt.xlabel("Communication Volume (Bytes)")
# plt.ylabel("Loss Reduction")
# plt.title("Loss Reduction vs Communication Volume — Bubble Size = Throughput")
# plt.grid(True)

# plt.tight_layout()
# plt.savefig(f"{os.path.dirname(os.path.realpath(__file__))}/bubble_plot.png", dpi=200)
# plt.show()

# -------- Manual label offsets --------
offsets = {
    "SparseLoCo": (-20, -10),
    "DiLoCo": (10, -10),
    "DeMo": (-10, 10),
    "MuLoCo": (15, 10),
}

for x, y, name in zip(communication, loss, methods):
    dx, dy = offsets.get(name, (5, 5))  # default offset
    plt.annotate(
        name,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=10,
        ha="center",
        va="center",
    )

plt.xlabel("Communication Volume (Bytes)")
plt.ylabel("Loss Reduction (Higher is Better)")
plt.title("Loss Reduction vs Communication Volume — Bubble Size = Throughput")
plt.grid(True)

plt.tight_layout()

plt.savefig(f"{os.path.dirname(os.path.realpath(__file__))}/bubble_plot.png", dpi=200)
plt.show()
