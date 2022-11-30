import seaborn as sns
import json
import matplotlib.pyplot as plt

with open("my_dict.json", "r") as read_content:
    data = json.load(read_content)

with open("positive_words.json") as read_content:
    positive_data = json.load(read_content)

negative_good = data["good"]
positive_good = positive_data["good"]
# sns.distplot(data["good"], hist=False, kde=True,
#              kde_kws={'linewidth': 3})
sns.distplot(data["bad"], hist=False, kde=True,
             kde_kws={'linewidth': 3})
# sns.distplot(positive_good, hist=False, kde=True,
#              kde_kws={'linewidth': 3})
# sns.distplot(positive_data["bad"], hist=False, kde=True,
#              kde_kws={'linewidth': 3})
# sns.distplot(positive_good, hist=False, kde=True,
#              kde_kws={'linewidth': 3})
# Plot formatting
plt.title('Attention weight distribution of word "bad"')
plt.xlabel('Weight')
plt.ylabel('Density')
plt.show()


if __name__ == "__main__":
    print("")
