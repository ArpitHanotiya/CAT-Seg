from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_score

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks in dataloader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_masks.extend(masks.cpu().numpy())

    # Compute IoU
    iou = jaccard_score(all_masks.flatten(), all_preds.flatten(), average='macro')
    print(f"IoU: {iou:.4f}")

# Usage:
test_dataset = get_dataset(...)
test_loader = DataLoader(test_dataset)
evaluate(model, test_loader)