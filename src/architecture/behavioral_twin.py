import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

        
class SimpleTextEncoder(nn.Module):
    def __init__(self, encoder_name="BAAI/bge-base-en-v1.5", pooling="cls", tune_layers=4):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)
        self.pooling = pooling
        self.tune_layers = tune_layers

        # Freeze all parameters first
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze last few layers
        if tune_layers > 0:
            encoder_layers = self.text_encoder.encoder.layer
            for layer in encoder_layers[-tune_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Unfreeze final LayerNorm
        if hasattr(self.text_encoder.encoder, 'LayerNorm'):
            for param in self.text_encoder.encoder.LayerNorm.parameters():
                param.requires_grad = True
                

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.pooling == "mean":
            embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        elif self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings.unsqueeze(1)  # [B, 1, H]


class BehavioralEncoder(nn.Module):
    def __init__(self, encoder_name="BAAI/bge-base-en-v1.5", pooling="cls"):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)
        self.pooling = pooling
        
        # Freeze all parameters - behavioral encoder should be frozen
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids, attention_mask):
        B, P, L = input_ids.shape
        
        input_ids = input_ids.view(B * P, L)
        attention_mask = attention_mask.view(B * P, L)
        
        # Remove torch.no_grad() to allow gradient flow if needed
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.pooling == "mean":
            embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        elif self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        H = embeddings.size(-1)
        embeddings = embeddings.view(B, P, H)
        token_padding_mask = (attention_mask.view(B, P, L).sum(-1) == 0)

        return embeddings, token_padding_mask


class QFormer(nn.Module):
    def __init__(self, hidden_size, num_queries=4, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, hidden_size))
        nn.init.xavier_uniform_(self.query_tokens)
        
        # Simplified cross-attention layers
        self.cross_attn_beh = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, 
            dropout=dropout, batch_first=True
        )
        self.cross_attn_inp = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        
        # Simplified transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, 
            dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, behavioral_embs, input_embs, behavioral_mask=None, input_mask=None):
        B = behavioral_embs.size(0)
        queries = self.query_tokens.expand(B, -1, -1)
        
        behavioral_embs = behavioral_embs.to(queries.dtype)
        input_embs = input_embs.to(queries.dtype)
        
        # Cross-attention to behavior
        q_beh, _ = self.cross_attn_beh(
            queries, behavioral_embs, behavioral_embs, 
            key_padding_mask=behavioral_mask
        )
        queries = self.norm1(queries + q_beh)

        # Cross-attention to input
        if input_mask is None:
            input_mask = torch.zeros(input_embs.size()[:2], dtype=torch.bool, device=input_embs.device)
            
        q_inp, _ = self.cross_attn_inp(
            queries, input_embs, input_embs,
            key_padding_mask=input_mask
        )
        queries = self.norm2(queries + q_inp)
        
        # Self-attention between queries
        queries = self.transformer(queries)
        
        return queries


class FusionModel(nn.Module):
    def __init__(self, llm_name="google/flan-t5-base", beh_hidden_size=768, system_prompt=None):
        super().__init__()

        if 'qwen' in llm_name.lower() or 'llama' in llm_name.lower():
            self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
            self.is_encoder_decoder = False
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
            llm_hidden_size = self.llm.config.hidden_size
        else:
            self.llm = AutoModelForSeq2SeqLM.from_pretrained(llm_name)
            self.is_encoder_decoder = True
            llm_hidden_size = self.llm.config.d_model

        # System prompt для decoder-only моделей
        self.system_prompt = system_prompt or "Answer with only [1] or [2]. "
        self.system_prompt_ids = None
        self.system_prompt_embs = None

        # Simplified prefix projection
        self.prefix_proj = nn.Sequential(
            nn.Linear(beh_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.LayerNorm(llm_hidden_size),
        )
        # self.prefix_proj = nn.Sequential(
        #     nn.Linear(beh_hidden_size, llm_hidden_size * 2),
        #     nn.GELU(),
        #     nn.LayerNorm(llm_hidden_size * 2),
        #     nn.Dropout(0.1),
        #     nn.Linear(llm_hidden_size * 2, llm_hidden_size),
        #     nn.GELU(),
        #     nn.LayerNorm(llm_hidden_size),
        # )

    def _prepare_system_prompt(self, device):
        """Подготовить эмбеддинги системного промпта"""
        if self.system_prompt_ids is None and not self.is_encoder_decoder:
            self.system_prompt_ids = self.llm_tokenizer(
                self.system_prompt,
                return_tensors='pt',
                add_special_tokens=False
            )['input_ids'].to(device)
            
            with torch.no_grad():
                self.system_prompt_embs = self.llm.get_input_embeddings()(self.system_prompt_ids)

    def forward(self, input_ids, attention_mask, qformer_output, labels=None, **kwargs):
        prefix_embs = self.prefix_proj(qformer_output)
        
        if self.is_encoder_decoder:
            inputs_embeds = self.llm.encoder.embed_tokens(input_ids)
            inputs_embeds = torch.cat([prefix_embs, inputs_embeds], dim=1)
            
            prefix_mask = torch.ones(prefix_embs.size()[:2], device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            self._prepare_system_prompt(input_ids.device)
            system_embs = self.system_prompt_embs.expand(inputs_embeds.size(0), -1, -1)
            inputs_embeds = torch.cat([system_embs, prefix_embs, inputs_embeds], dim=1)
            
            # Обновляем attention mask
            system_mask = torch.ones(
                (attention_mask.size(0), self.system_prompt_ids.size(1)),
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            prefix_mask = torch.ones(prefix_embs.size()[:2], device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([system_mask, prefix_mask, attention_mask], dim=1)

        if labels is None:
            raise ValueError("For generation use .generate() method!")

        if not self.is_encoder_decoder and labels is not None:
            batch_size = labels.size(0)
            current_labels_len = labels.size(1)
            inputs_embeds_len = inputs_embeds.size(1)
            
            if current_labels_len < inputs_embeds_len:
                padding_len = inputs_embeds_len - current_labels_len
                padding = torch.full((batch_size, padding_len), -100, 
                                   device=labels.device, dtype=labels.dtype)
                labels = torch.cat([padding, labels], dim=1)
            elif current_labels_len > inputs_embeds_len:
                labels = labels[:, :inputs_embeds_len]
    
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        return outputs

    def generate(self, input_ids, attention_mask, qformer_output, **generate_kwargs):
        prefix_embs = self.prefix_proj(qformer_output)
        
        if self.is_encoder_decoder:
            inputs_embeds = self.llm.encoder.embed_tokens(input_ids)
            inputs_embeds = torch.cat([prefix_embs, inputs_embeds], dim=1)
            prefix_mask = torch.ones(prefix_embs.size()[:2], device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
      
            self._prepare_system_prompt(input_ids.device)
            system_embs = self.system_prompt_embs.expand(inputs_embeds.size(0), -1, -1)
            inputs_embeds = torch.cat([system_embs, prefix_embs, inputs_embeds], dim=1)
            
            system_mask = torch.ones(
                (attention_mask.size(0), self.system_prompt_ids.size(1)),
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            prefix_mask = torch.ones(prefix_embs.size()[:2], device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([system_mask, prefix_mask, attention_mask], dim=1)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs
        )
        return outputs


class BehavioralTwin(nn.Module):
    def __init__(self, beh_encoder, input_encoder, qformer, fusion):
        super().__init__()
        self.beh_encoder = beh_encoder
        self.input_encoder = input_encoder
        self.qformer = qformer
        self.fusion = fusion

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            # Freeze LLM and behavioral encoder
            self.beh_encoder.eval()
            self.fusion.llm.eval()
            
            # Train: qformer, layer projections, input encoder
            self.input_encoder.train()
            self.qformer.train()
            self.fusion.prefix_proj.train()
            
            # Disable gradients for frozen components
            for p in self.beh_encoder.parameters():
                p.requires_grad = False
            for p in self.fusion.llm.parameters():
                p.requires_grad = False
            
            # Enable gradients for trainable components
            for p in self.qformer.parameters():
                p.requires_grad = True
            for p in self.fusion.prefix_proj.parameters():
                p.requires_grad = True
            for p in self.input_encoder.parameters():
                p.requires_grad = True

        return self

    def eval(self):
        super().eval()
        # В eval режиме отключаем градиенты для всех компонентов
        for module in [self.beh_encoder, self.input_encoder, self.qformer, self.fusion]:
            module.eval()
            
        # Отключаем все градиенты
        for p in self.parameters():
            p.requires_grad = False
            
        return self

    def forward(self, input_ids, attention_mask, llm_input_ids, llm_attention_mask, beh_input_ids, beh_attention_mask, labels=None):
        beh_embs, beh_padding_mask = self.beh_encoder(beh_input_ids, beh_attention_mask)
        input_embs = self.input_encoder(input_ids, attention_mask)
        qformer_out = self.qformer(beh_embs, input_embs, beh_padding_mask)
        return self.fusion(llm_input_ids, llm_attention_mask, qformer_out, labels)

    def generate(self, input_ids, attention_mask, llm_input_ids, llm_attention_mask, beh_input_ids, beh_attention_mask, **generate_kwargs):
        beh_embs, beh_padding_mask = self.beh_encoder(beh_input_ids, beh_attention_mask)
        input_embs = self.input_encoder(input_ids, attention_mask)
        qformer_out = self.qformer(beh_embs, input_embs, beh_padding_mask)
        return self.fusion.generate(llm_input_ids, llm_attention_mask, qformer_out, **generate_kwargs)


class MemoryOptimizedBehavioralTwin(BehavioralTwin):
    def __init__(self, beh_encoder, input_encoder, qformer, fusion, use_gradient_checkpointing=True):
        super().__init__(beh_encoder, input_encoder, qformer, fusion)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if use_gradient_checkpointing:
            if hasattr(self.fusion.llm, 'gradient_checkpointing_enable'):
                self.fusion.llm.gradient_checkpointing_enable()
            
            self.beh_encoder.text_encoder.gradient_checkpointing_enable()
            self.input_encoder.text_encoder.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, llm_input_ids, llm_attention_mask, beh_input_ids, beh_attention_mask, labels=None):
        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            
            def forward_behavioral():
                return self.beh_encoder(beh_input_ids, beh_attention_mask)
            
            def forward_input():
                return self.input_encoder(input_ids, attention_mask)
            
            beh_embs = checkpoint(forward_behavioral, use_reentrant=False)
            input_embs = checkpoint(forward_input, use_reentrant=False)
            qformer_out = checkpoint(self.qformer, beh_embs, input_embs, use_reentrant=False)
            
            return self.fusion(llm_input_ids, llm_attention_mask, qformer_out, labels)
        else:
            return super().forward(input_ids, attention_mask, llm_input_ids, llm_attention_mask, 
                                 beh_input_ids, beh_attention_mask, labels)