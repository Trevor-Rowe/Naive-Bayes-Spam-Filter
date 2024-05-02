from typing import Set, NamedTuple, List, Tuple, Dict, Iterable
from collections import defaultdict, Counter
import math
import glob, re, random
from scratch.machine_learning import split_data

def tokenize(text: str) -> Set[str]:
    text = text.lower() # Convert to lowercase
    all_words = re.findall("[a-z0-9']+", text)
    return set(all_words)

assert tokenize("Data Science is science")  == {"data", "science", "is"}

class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5):
        self.k = k
        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    class Message(NamedTuple):
        text: str
        is_spam: bool

    def _probabilities(self, token: str) -> Tuple[float, float]:
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]
        # smoothing factor k used here
        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k) 
        p_token_ham  =  (ham + self.k)  / (self.ham_messages + 2 * self.k)
        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)
            # If token appears in the message, add the probability of seeing it
            if token in text_tokens: # log(ab) = log(a) + log(b)
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham  += math.log(prob_if_ham) 
            # Otherwise add the log probability of _not_ seeing it,
            # which is log(1 - probability of seeing it)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)
        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        # Naive Bayes probability
        return prob_if_spam / (prob_if_spam + prob_if_ham)
    
    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # Increment message counts
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1
            # Increment word counts
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

messages = [
NaiveBayesClassifier.Message("spam rules", is_spam=True),
NaiveBayesClassifier.Message("ham rules", is_spam=False),
NaiveBayesClassifier.Message("hello ham", is_spam=False)]
model = NaiveBayesClassifier(k = 0.5)
model.train(messages)

assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

text = "hello spam"

probs_if_spam = [
    (1 + 0.5) / (1 + 2 * 0.5),      # "spam"  (present)
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
    (0 + 0.5) / (1 + 2 * 0.5)       # "hello" (present)
]       

probs_if_ham = [
    (0 + 0.5) / (2 + 2 * 0.5),      # "spam"  (present)
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
    (1 + 0.5) / (2 + 2 * 0.5),      # "hello" (present)
]

p_if_spam: float = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham: float = math.exp(sum(math.log(p) for p in probs_if_ham))
assert model.predict(text) == p_if_spam / (p_if_spam + p_if_ham)
 
print("Spam filter online >:)")

path = 'spam_data/*/*'
data: List[NaiveBayesClassifier.Message] = []
for filename in glob.glob(path):
    is_spam = "ham" not in filename
    with open(filename, errors="ignore") as email_file:
        for line in email_file:
            if line.startswith("Subject:"):
                subject = line.lstrip("Subject:")
                data.append(NaiveBayesClassifier.Message(subject, is_spam))
                break

random.seed(0)
train_messages, test_messages = split_data(data, 0.75)
model = NaiveBayesClassifier() # k = 0.5
model.train(train_messages)

predictions = [(message, model.predict(message.text)) for message in test_messages]
confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                           for message, spam_probability in predictions)

print("\n Confustion Matrix:", confusion_matrix, "\n")
email_subjects = [
    "Credit Card Resolution Needed Immediately", # spam
    "Regarding your barking dog", # ham
    "You've won, congratulations!", # spam
    "Your Chase Credit Score", # ham
    "Yo Dude", # ham
    "Join Millions in getting a bigger member today!" #spam
]
for email in email_subjects:
    prob_spam = model.predict(email)
    print(f"Subject: {email} | spam confidence: {prob_spam}", "\n")