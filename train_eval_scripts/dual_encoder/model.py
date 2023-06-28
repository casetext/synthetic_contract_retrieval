from transformers import AutoModel, PreTrainedModel, AutoTokenizer
import torch
from torch.nn import functional as F
import ipdb

class AutoDualEncoder(PreTrainedModel):
    def __init__(self,config,init_model=True,margin=0.1,scale=1):
        super().__init__(config)
        self.pooling = 'mean'
        self.model = None
        if init_model:
            self.model = AutoModel(config)
        self.passage_encoder = None
        self.cosine_loss = torch.nn.CosineSimilarity()
        self.mse_loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.scale = scale
        self.margin = margin
        
    def load_passage_encoder(self,model_name_or_path, freeze=False):
        self.passage_encoder = AutoModel.from_pretrained(model_name_or_path)

        if freeze:
            for param in self.passage_encoder.parameters():
                param.requires_grad = False
        
    def forward_triplet(
            self,
            input_ids_anchor=None,
            attention_mask_anchor=None,
            input_ids_positive=None,
            attention_mask_positive=None,            
            input_ids_negative=None,
            attention_mask_negative=None,            
            margin=None,
            **kwargs
        ):

        
        if not margin:
            margin = self.margin
        pooled_embeddings_anchor = self.forward_pool_query(input_ids_anchor,attention_mask_anchor)
        pooled_embeddings_positive = self.forward_pool_passage(input_ids_positive,attention_mask_positive)
        pooled_embeddings_negative = self.forward_pool_passage(input_ids_negative,attention_mask_negative)
        
        pos_similarity = torch.cosine_similarity(pooled_embeddings_anchor,pooled_embeddings_positive) 
        neg_similarity = torch.cosine_similarity(pooled_embeddings_anchor,pooled_embeddings_negative)         
        
        loss = F.relu(neg_similarity-pos_similarity + margin).mean()
        
        return loss

    def forward_triplet_ce(
            self,
            input_ids_anchor=None,
            attention_mask_anchor=None,
            input_ids_positive=None,
            attention_mask_positive=None,            
            input_ids_negative=None,
            attention_mask_negative=None,
            **kwargs
        ):
        
        pooled_embeddings_anchor = self.forward_pool_query(input_ids_anchor,attention_mask_anchor)
        pooled_embeddings_positive = self.forward_pool_passage(input_ids_positive,attention_mask_positive)
        pooled_embeddings_negative = self.forward_pool_passage(input_ids_negative,attention_mask_negative)

        pooled_embeddings_anchor = torch.unsqueeze(pooled_embeddings_anchor,1)
        pooled_embeddings_positive = torch.unsqueeze(pooled_embeddings_positive,-1)
        pooled_embeddings_negative = torch.unsqueeze(pooled_embeddings_negative,-1)
        
        pos_similarity = torch.bmm(pooled_embeddings_anchor,pooled_embeddings_positive) 
        neg_similarity = torch.bmm(pooled_embeddings_anchor,pooled_embeddings_negative)         
        
        preds = torch.cat([pos_similarity,neg_similarity],dim=1)
        preds = torch.squeeze(preds,-1)
        labels = torch.zeros(preds.shape[0], dtype=torch.long).to(device='cuda')
        loss = self.ce_loss(preds,labels)

        return loss
    
    def forward_pool_query(self,input_ids,attention_mask):
        hidden_states = self.model(input_ids=input_ids,
                                   attention_mask=attention_mask)[0]
        return self.mean_pool(hidden_states,attention_mask)
    
    def forward_pool_passage(self,input_ids,attention_mask):
        hidden_states = self.passage_encoder(input_ids=input_ids,
                                   attention_mask=attention_mask)[0]
        return self.mean_pool(hidden_states,attention_mask)
            
    def mean_pool(self,embeddings,attention_mask):
        """ mean pools along sentence direction by taking into account 
        individual sentence lengths by using attention masks """                        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
        return sum_embeddings/input_mask_expanded.sum(1)
    
    def encode(self,
                input_ids=None,
                attention_mask=None,
                **kwargs):
        with torch.no_grad():
            embeddings = self.passage_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask)[0]
            embeddings= self.mean_pool(embeddings,attention_mask)
            return embeddings/torch.norm(embeddings,dim=1).unsqueeze(dim=1)
        
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        model = AutoModel.from_pretrained(model_name_or_path)
        instance = cls(model.config,init_model=False)
        instance.model = model
        return instance