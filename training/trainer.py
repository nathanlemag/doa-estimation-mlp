import torch

def train_model(model, train_loader, val_loader, criterion, epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def run_epoch(loader, is_train):
        if is_train:
            model.train()
        else:
            model.eval()

        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(is_train):
            for x, true_class, true_doas, _ in loader:
                x, true_class, true_doas = x.to(device), true_class.to(device), true_doas.to(device)

                class_logits, doa_preds = model(x)
                loss, _, _ = criterion(class_logits, doa_preds, true_class, true_doas)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    # Count the number of correctly predicted K sources in the batch
                    correct += (class_logits.argmax(dim=1) == true_class).sum().item()
                    # Add the batch size to the total number of samples evaluated
                    total += true_class.size(0)

                # Accumulate the average loss of each batch over the entire epoch
                total_loss += loss.item()

        # Average of the losses across all batches in the epoch
        avg_loss = total_loss / len(loader)
        # Percentage of correct predictions by K over the entire validation set
        acc = 100.0 * correct / total if not is_train else None
        return avg_loss, acc

    for epoch in range(epochs):
        train_loss, _ = run_epoch(train_loader, is_train=True)
        val_loss, val_acc = run_epoch(val_loader, is_train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    return history