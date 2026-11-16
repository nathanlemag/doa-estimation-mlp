import torch


def train_model(model, train_loader, val_loader, criterion, epochs=50, lr=1e-3):
    # Move model to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Adam optimizer as used in the paper
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Keep track of metrics at each epoch for plotting later
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        # --- 1. TRAINING PHASE ---
        model.train()  # enable BatchNorm updates
        running_train_loss = 0.0

        for x, true_class, true_doas, _ in train_loader:
            x = x.to(device)
            true_class = true_class.to(device)
            true_doas = true_doas.to(device)

            # Reset gradients from previous batch
            optimizer.zero_grad()

            # Forward pass + compute loss
            class_logits, doa_preds = model(x)
            loss, _, _ = criterion(class_logits, doa_preds, true_class, true_doas)

            # Backward pass + update weights
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # --- 2. VALIDATION PHASE ---
        model.eval()  # freeze BatchNorm stats
        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # no gradient calculation needed for validation
            for x, true_class, true_doas, _ in val_loader:
                x = x.to(device)
                true_class = true_class.to(device)
                true_doas = true_doas.to(device)

                class_logits, doa_preds = model(x)
                loss, _, _ = criterion(class_logits, doa_preds, true_class, true_doas)
                running_val_loss += loss.item()

                # Compute classification accuracy (predicted K vs true K)
                pred_class = torch.argmax(class_logits, dim=1)
                correct += (pred_class == true_class).sum().item()
                total += true_class.size(0)

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    return history