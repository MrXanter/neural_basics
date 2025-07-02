import torch
from model import MySimpleModel
from model_loader import input_dim, num_classes, val_loader
from model_loader import X



model = MySimpleModel(input_dim, num_classes)


model.load_state_dict(torch.load("model.pth"))
model.eval()

x = X[0].unsqueeze(0)

correct = 0
total = 0


with torch.no_grad():
    for x_batch, y_batch in val_loader:

        logits = model(x_batch)
        predicted_class = torch.argmax(logits, dim=1).item()
        print(f"Predicted class: {predicted_class}") #мне не нравится вывод в каждом проходе по циклу
        correct += (predicted_class == y_batch).sum().item()
        total += y_batch.size(0)
    

        
accuracy = correct / total
print(f"Val Accuracy: {accuracy:.2%}")


