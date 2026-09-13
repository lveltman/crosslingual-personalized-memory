import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSeq2SeqLM

class PPlugModel(nn.Module):
    def __init__(self, 
                 beh_encoder_name="BAAI/bge-base-en-v1.5",
                 input_encoder_name="BAAI/bge-base-en-v1.5",
                 llm_name="google/flan-t5-base"):
        super().__init__()
        
        # 1. Enchis — замороженный
        self.beh_encoder = AutoModel.from_pretrained(beh_encoder_name, trust_remote_code=True)
        for param in self.beh_encoder.parameters():
            param.requires_grad = False
        self.beh_encoder.eval()
        
        # 2. Encinput — обучаемый
        self.input_encoder = AutoModel.from_pretrained(input_encoder_name, trust_remote_code=True)
        # параметры остаются trainable
        
        # 3. Proj — 2-layer MLP
        beh_hidden = self.beh_encoder.config.hidden_size
        llm_hidden = AutoModelForSeq2SeqLM.from_pretrained(llm_name).config.d_model
        
        # 5. Фиксированная LLM
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(llm_name)
        for param in self.llm.parameters():
            param.requires_grad = False

        self.llm_dtype = self.llm.dtype
        
        self.llm.eval()

        self.proj = nn.Sequential(
            nn.Linear(beh_hidden, beh_hidden * 2),
            nn.GELU(),
            nn.Linear(beh_hidden * 2, llm_hidden)
        ).to(self.llm_dtype)

        # 4. Instruction embedding I (один токен!)
        self.instruction_embedding = nn.Parameter(torch.randn(1, 1, llm_hidden, dtype=self.llm_dtype))
        
        self.llm_hidden = llm_hidden

    def encode_behaviors(self, beh_input_ids, beh_attention_mask):
        """beh_input_ids: [B, P, L] → возвращает [B, P, H]"""
        B, P, L = beh_input_ids.shape
        beh_input_ids = beh_input_ids.view(B * P, L)
        beh_attention_mask = beh_attention_mask.view(B * P, L)
        
        with torch.no_grad():
            outputs = self.beh_encoder(
                input_ids=beh_input_ids,
                attention_mask=beh_attention_mask
            )
        # Используем mean pooling (BGE не имеет [CLS] для агрегации)
        embeddings = self.mean_pooling(outputs.last_hidden_state, beh_attention_mask)
        return embeddings.view(B, P, -1).to(self.llm_dtype)  # [B, P, H]

    def encode_input(self, input_ids, attention_mask):
        """input_ids: [B, L] → [B, H]"""
        outputs = self.input_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.mean_pooling(outputs.last_hidden_state, attention_mask).to(self.llm_dtype)  # [B, H]

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, 
                input_ids, attention_mask,
                llm_input_ids, llm_attention_mask,
                beh_input_ids, beh_attention_mask,
                labels=None):

        proj_dtype = next(self.proj.parameters()).dtype
        # 1. Кодируем поведение и запрос
        beh_embs = self.encode_behaviors(beh_input_ids, beh_attention_mask)  # [B, P, H_beh]
        input_emb = self.encode_input(input_ids, attention_mask)            # [B, H_beh]
        beh_embs = beh_embs.to(proj_dtype)
        input_emb = input_emb.to(proj_dtype)
        # 2. Проекция поведения в пространство LLM
        beh_embs_proj = self.proj(beh_embs)    # [B, P, H_llm]
        input_emb_proj = self.proj(input_emb)  # [B, H_llm] — не обязательно, но логично
        
        # 3. Dot-product attention: w_i = softmax(input_emb_proj @ beh_embs_proj^T)
        # Но: input_emb_proj — [B, H], beh_embs_proj — [B, P, H]
        # → нужно unsqueeze
        query = input_emb_proj.unsqueeze(1)    # [B, 1, H]
        scores = torch.bmm(query, beh_embs_proj.transpose(1, 2))  # [B, 1, P]
        weights = torch.softmax(scores, dim=-1)  # [B, 1, P]
        
        # 4. Агрегация: P_u = Σ w_i * beh_embs_proj
        personal_emb = torch.bmm(weights, beh_embs_proj)  # [B, 1, H_llm]
        personal_emb = personal_emb.squeeze(1)           # [B, H_llm]
        
        # 5. Формируем вход для LLM: [I; P_u; Emb(x)]
        instruction = self.instruction_embedding.expand(input_emb.size(0), -1, -1)  # [B, 1, H]
        personal_emb = personal_emb.unsqueeze(1)  # [B, 1, H]
        
        llm_inputs_embeds = self.llm.encoder.embed_tokens(llm_input_ids)  # [B, L, H]
        full_embeds = torch.cat([instruction, personal_emb, llm_inputs_embeds], dim=1)  # [B, 2+L, H]
        
        # Маска
        instr_mask = torch.ones(instruction.size()[:2], device=llm_attention_mask.device)
        personal_mask = torch.ones(personal_emb.size()[:2], device=llm_attention_mask.device)
        full_mask = torch.cat([instr_mask, personal_mask, llm_attention_mask], dim=1)
        
        # 6. Прогон через LLM
        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            labels=labels
        )
        return outputs

    
    @torch.no_grad()
    def generate(self,
                 input_ids, 
                 attention_mask, 
                 llm_input_ids, 
                 llm_attention_mask, 
                 beh_input_ids, 
                 beh_attention_mask,
                 max_new_tokens=10,
                 num_beams=4,
                 **kwargs):
        """
        Генерация персонализированного ответа с использованием PPlug.
        
        Args:
            input_ids, attention_mask: токены и маска для Encinput (текущий запрос)
            beh_input_ids, beh_attention_mask: токены и маска для Enchis (история поведения)
            llm_input_ids, llm_attention_mask: токены и маска для LLM (инструкция + запрос)
            max_new_tokens: сколько токенов генерировать (для LaMP-1 достаточно 3–5)
            num_beams: размер бим-серча (по умолчанию 4, как в статье)
        """
        # 1. Кодируем поведение и текущий запрос
        beh_embs = self.encode_behaviors(beh_input_ids, beh_attention_mask)  # [B, P, H_beh]
        input_emb = self.encode_input(input_ids, attention_mask)            # [B, H_beh]

        proj_dtype = next(self.proj.parameters()).dtype
        beh_embs = beh_embs.to(proj_dtype)
        input_emb = input_emb.to(proj_dtype)
    
        # 2. Проекция в пространство LLM
        beh_embs_proj = self.proj(beh_embs)    # [B, P, H_llm]
        input_emb_proj = self.proj(input_emb)  # [B, H_llm]

        # 3. Dot-product attention для агрегации
        query = input_emb_proj.unsqueeze(1)    # [B, 1, H_llm]
        scores = torch.bmm(query, beh_embs_proj.transpose(1, 2))  # [B, 1, P]
        weights = torch.softmax(scores, dim=-1)
        personal_emb = torch.bmm(weights, beh_embs_proj).squeeze(1)  # [B, H_llm]

        # 4. Формируем полный эмбеддинг: [I; P_u; Emb(x)]
        B = input_emb.size(0)
        instruction = self.instruction_embedding.expand(B, -1, -1)  # [B, 1, H_llm]
        personal_emb = personal_emb.unsqueeze(1)                   # [B, 1, H_llm]
        
        llm_inputs_embeds = self.llm.encoder.embed_tokens(llm_input_ids)  # [B, L, H_llm]
        full_embeds = torch.cat([instruction, personal_emb, llm_inputs_embeds], dim=1)  # [B, 2+L, H_llm]

        # 5. Маска внимания
        instr_mask = torch.ones(instruction.size()[:2], device=llm_attention_mask.device, dtype=llm_attention_mask.dtype)
        personal_mask = torch.ones(personal_emb.size()[:2], device=llm_attention_mask.device, dtype=llm_attention_mask.dtype)
        full_attention_mask = torch.cat([instr_mask, personal_mask, llm_attention_mask], dim=1)

        # 6. Генерация с beam search
        outputs = self.llm.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            # pad_token_id=self.llm.config.pad_token_id,
            # eos_token_id=self.llm.config.eos_token_id,
            **kwargs
        )
        return outputs