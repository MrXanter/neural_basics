import torch
from model_loader import val_loader, train_loader
import torch.optim as optim
import torch.nn as nn
from model import MySimpleModel
from model_loader import input_dim, num_classes




model = MySimpleModel(input_dim, num_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    train_loss = 0
    
    #Training loop
    model.train()
    for x_batch, y_batch in train_loader:
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()


    val_correct = 0
    val_total = 0
    val_loss = 0
    accuracy = 0.0

    #Validation
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            pred = torch.argmax(y_pred, dim=1)
            val_loss += loss.item()
            val_correct += (pred == y_batch).sum().item()
            val_total += y_batch.size(0)



    accuracy = val_correct / val_total
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Accuracy = {accuracy:.2%}")





torch.save(model.state_dict(), "model.pth")
print("Model saved.")