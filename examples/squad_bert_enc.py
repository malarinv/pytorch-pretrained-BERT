import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel

cur_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(cur_dir, 'models/pytorch_objects/squad_bert')
pretrained_path = os.path.join(models_dir,'a803ce83ca27fecf74c355673c434e51c265fb8a3e0e57ac62a80e38ba98d384.681017f415dfb33ec8d0e04fe51a619f3f01532ecea04edbfd48c5d160550d9c')
tokenizer_path = os.path.join(models_dir,'5e8a2b4893d13790ed4150ca1906be5f7a03d6c4ddf62296c383f6db42814db2.e13dbb970cb325137104fb2e5f36fe865f27746c6b526f6352861b1980eb80b1')
finetuned_path = os.path.join(models_dir,'pytorch_model.bin')

class BertEncoder(object):
    """docstring for BertEncoder."""
    def __init__(self):
        super(BertEncoder, self).__init__()
        # Load pre-trained model tokenizer (vocabulary)

    def load(self):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained(pretrained_path)
        self.model.load_state_dict(torch.load(finetuned_path,map_location='cpu'))
        self.model.eval()

    def encode(self,text_list):
# Tokenized input
        # text = "Who was Jim Henson"
        output = []
        for text in text_list:
            tokenized_text = self.tokenizer.tokenize(text)
            # tokenized_text
            # Mask a token that we will try to predict back with `BertForMaskedLM`
            # masked_index = 6
            # tokenized_text[masked_index] = '[MASK]'
            # assert tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']

            # Convert token to vocabulary indices
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            # indexed_tokens
            # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
            segments_ids = [0]*len(indexed_tokens)#[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            # Predict hidden states features for each layer
            encoded_layers, encoding = self.model(tokens_tensor, segments_tensors)
            output.append(encoding[0].detach().numpy())
        # We have a hidden states for each of the 12 layers in model bert-base-uncased
        # _
        # assert len(encoded_layers) == 12
        return output

    @staticmethod
    def cosine(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


if __name__ == '__main__':
    be = BertEncoder()
    encs = be.encode(["I want to make pizza","kill that guy","I want to build pizza"])
    print(be.cosine(encs[0],encs[1]))
