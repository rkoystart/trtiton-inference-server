import torch
from transformers import *

class WrappedModel(torch.nn.Module):
    def __init__(self):
        super(WrappedModel, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased').cuda()
    def forward(self, data):
        return self.model(data.cuda())

example = torch.zeros((4,128), dtype=torch.long) # bsz , seqlen
pt_model = WrappedModel().eval()
traced_script_module = torch.jit.trace(pt_model, example)
traced_script_module.save("model_repository/bert_pt/1/model.pt")
