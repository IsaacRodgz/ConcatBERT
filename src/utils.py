from src.dataset import MemeDataset
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
    data = MemeDataset(args.data_path, dataset, split, args.bert_model, args.max_token_length)
    return data
