
##reading positive and negative mark_safe statements
pos_file_name = "pos.txt"
neg_file_name = "neg.txt"

mark_safe_model = []
with open(pos_file_name, "r", encoding='utf-8') as f:
    for line in f:
        mark_safe_model.append((line.split(),"positive"))

with open(neg_file_name, "r", encoding='utf-8') as f:
    for line in f:
        mark_safe_model.append((line.split(),"negative"))
        
f.close()

print(mark_safe_model)

train_mark_safe = [
    (['love', 'this', 'car'], 'positive'),
    (['this', 'view', 'amazing'], 'positive'),
    (['feel', 'great', 'this', 'morning'], 'positive'),
    (['excited', 'about', 'the', 'concert'], 'positive'),
    (['best', 'friend'], 'positive'),
    (['not', 'like', 'this', 'car'], 'negative'),
    (['this', 'view', 'horrible'], 'negative'),
    (['feel', 'tired', 'this', 'morning'], 'negative'),
    (['not', 'looking', 'forward', 'the', 'concert'], 'negative'),
    (['enemy'], 'negative')]