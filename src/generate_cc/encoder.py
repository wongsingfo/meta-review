import torch
import torch.nn as nn
from transformers import BartTokenizer, BartModel 


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        
        return hidden_states



class MetaModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = BartModel.from_pretrained("facebook/bart-large-cnn")
        
        self.classifier = BartClassificationHead(input_dim = 1024, inner_dim = 512, num_classes = 9, pooler_dropout = 0.2)
        self.counter = BartClassificationHead(input_dim = 1024, inner_dim = 512, num_classes = 9, pooler_dropout = 0.2)

        self.freeze_layers(12)

    def freeze_layers(self, numFreeze):
        unfreeze_layers = ["pooler"]
        for i in range(numFreeze, 12):
            unfreeze_layers.append("layer."+str(i))

        for name ,param in self.model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

    def forward(self, text):
        inputs = self.tokenizer(text, padding = True, truncation = True, max_length = 1024, return_tensors="pt")
        # print(inputs.keys())
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
        
        input_ids = inputs['input_ids']

        outputs = self.model(**inputs)
        hidden_states = outputs[0]  # last hidden state
        eos_mask = input_ids.eq(self.model.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
                    raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]

        logits = self.classifier(sentence_representation)
        c_logits = self.counter(sentence_representation)

        return logits, c_logits



