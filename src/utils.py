from src.dataset import MemeDataset
from torchvision import transforms
import os


def get_data(args, dataset, split='train'):
    data_path = os.path.join(args.data_path, dataset) + f'/{split}.jsonl'
    '''
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    '''
    data = MemeDataset(args.data_path,
                       dataset, split,
                       args.bert_model,
                       args.max_token_length,
                       transform=transforms.Compose([
                           transforms.Resize((256, 256)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                       ]))
    return data
