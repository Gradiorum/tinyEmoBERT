import torch
from torch.utils.data import DataLoader, TensorDataset

# Function to load and debug .pt dataset
def debug_pt_dataset(pt_file_path):
    try:
        # Load the TensorDataset from disk
        dataset = torch.load(pt_file_path)
        
        # Make sure that the dataset is a TensorDataset
        assert isinstance(dataset, TensorDataset), "Dataset should be of type TensorDataset"

        # Count how many items are not tensors
        non_tensor_count = 0
        for i, data_tuple in enumerate(dataset):
            for item in data_tuple:
                if not isinstance(item, torch.Tensor):
                    non_tensor_count += 1
                    print(f"Non-tensor item found at index {i}: {item}")

        print(f"Total non-tensor items: {non_tensor_count}")

        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Type and Shape Checks
        print("---- Checking Types and Shapes ----")
        for batch in dataloader:
            print(f"Batch type: {type(batch)}")
            print(f"Batch length: {len(batch)}")
            
            # Ensure all items in the batch are tensors
            assert all(isinstance(item, torch.Tensor) for item in batch), "All batch items should be of type torch.Tensor"
            
            # Ensure all tensors have the same size in the first dimension
            batch_size = batch[0].size(0)
            assert all(item.size(0) == batch_size for item in batch), "All tensors should have the same size in the first dimension"

            print(f"Component shapes: {[item.shape for item in batch]}")
            break  # Only check the first batch

        # Data Consistency Checks
        print("---- Checking Data Consistency ----")
        for i, (input_ids, attention_masks, labels) in enumerate(dataloader):
            if i >= 1:  # Limit to one batch for demonstration
                break
            print(f"Sample input_ids: {input_ids[0]}")
            print(f"Sample attention_masks: {attention_masks[0]}")
            print(f"Sample label: {labels[0]}")
        
        # Additional checks can go here...

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    debug_pt_dataset(r"E:\text_datasets\saved\val_sentimen_32.pt")  # Replace with your actual .pt file path
