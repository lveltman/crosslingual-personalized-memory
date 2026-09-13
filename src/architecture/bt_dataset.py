import json
import torch
from torch.utils.data import Dataset, DataLoader

import ijson
ijson.backend


class LaMPDataset(Dataset):
    def __init__(self, json_path, llm_tokenizer, beh_tokenizer, input_tokenizer, max_len_llm=256, max_len_beh=512, llm_type="encoder-decoder"):
        """
        llm_type: "encoder-decoder" или "decoder-only"
        """
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.llm_tokenizer = llm_tokenizer
        self.beh_tokenizer = beh_tokenizer
        self.input_tokenizer = input_tokenizer
        self.max_len_llm = max_len_llm
        self.max_len_beh = max_len_beh
        self.llm_type = llm_type
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input_text"]
        output_text = item["output_text"]
        profile_texts = item["behavior_profile_text"]


        # Обрабатываем каждую статью отдельно
        beh_input_ids_list = []
        beh_attention_mask_list = []
        
        for profile_item in profile_texts:
            text = profile_item
            
            enc = self.beh_tokenizer(
                text, 
                truncation=True, 
                padding="max_length",
                max_length=self.max_len_beh,
                return_tensors="pt"
            )
            beh_input_ids_list.append(enc["input_ids"].squeeze(0))  # [L]
            beh_attention_mask_list.append(enc["attention_mask"].squeeze(0))  # [L]
        
        # Stack: [num_articles, L]
        beh_input_ids = torch.stack(beh_input_ids_list, dim=0)  # [P, L]
        beh_attention_mask = torch.stack(beh_attention_mask_list, dim=0)  # [P, L]
        
        input_enc = self.input_tokenizer(
            input_text, truncation=True, padding="max_length",
            max_length=self.max_len_beh, return_tensors="pt"
        )

        llm_input_enc = self.llm_tokenizer(
            input_text, truncation=True, padding="max_length",
            max_length=self.max_len_llm, return_tensors="pt"
        )
        llm_output_enc = self.llm_tokenizer(
            output_text, truncation=True, padding="max_length",
            max_length=self.max_len_llm, return_tensors="pt"
        )
        
        if self.llm_type == "encoder-decoder":
            labels = llm_output_enc["input_ids"].squeeze(0)
            labels[labels == self.llm_tokenizer.pad_token_id] = -100
        else:
            input_ids = torch.cat(
                [llm_input_enc["input_ids"], llm_output_enc["input_ids"]], dim=1
            )
            attention_mask = torch.cat(
                [llm_input_enc["attention_mask"], llm_output_enc["attention_mask"]], dim=1
            )
            labels = input_ids.clone()
            # Маскируем только input часть (instruction + prefix добавятся позже)
            labels[:, :llm_input_enc["input_ids"].size(1)] = -100
            target_start = llm_input_enc["input_ids"].size(1)
            labels[:, target_start:][labels[:, target_start:] == self.llm_tokenizer.pad_token_id] = -100
    
            llm_input_enc = {
                "input_ids": input_ids.squeeze(0),
                "attention_mask": attention_mask.squeeze(0)
            }
            labels = labels.squeeze(0)

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "llm_input_ids": llm_input_enc["input_ids"].squeeze(0),
            "llm_attention_mask": llm_input_enc["attention_mask"].squeeze(0),
            "labels": labels,
            "beh_input_ids": beh_input_ids,  # [P, L]
            "beh_attention_mask": beh_attention_mask,  # [P, L]
        }


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    llm_input_ids = torch.stack([item["llm_input_ids"] for item in batch])
    llm_attention_mask = torch.stack([item["llm_attention_mask"] for item in batch])

    max_label_len = max(item["labels"].size(0) for item in batch)

    labels_padded = []
    for item in batch:
        # Проверка на валидность token_id
        assert (item["labels"] >= -100).all(), f"Found token_id < -100 in labels: {item['labels']}"
        assert (item["labels"] < 32100).all() or (item["labels"] == -100).all(), \
               f"Found token_id >= vocab_size in labels: {item['labels']} (vocab_size: {32100})"
    
        pad_len = max_label_len - item["labels"].size(0)
        if pad_len > 0:
            padded = torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=item["labels"].dtype, device=item["labels"].device)])
        else:
            padded = item["labels"]
        labels_padded.append(padded)
    
    labels = torch.stack(labels_padded)

    # Behavioral padding
    max_articles = max(item["beh_input_ids"].size(0) for item in batch)
    beh_input_ids_padded = []
    beh_attention_mask_padded = []

    for item in batch:
        num_articles = item["beh_input_ids"].size(0)
        pad_size = max_articles - num_articles

        beh_ids = torch.cat([
            item["beh_input_ids"],
            torch.zeros(pad_size, item["beh_input_ids"].size(1), dtype=torch.long, device=item["beh_input_ids"].device)
        ], dim=0)

        beh_mask = torch.cat([
            item["beh_attention_mask"],
            torch.zeros(pad_size, item["beh_attention_mask"].size(1), dtype=item["beh_attention_mask"].dtype, device=item["beh_attention_mask"].device)
        ], dim=0)

        beh_input_ids_padded.append(beh_ids)
        beh_attention_mask_padded.append(beh_mask)

    beh_input_ids = torch.stack(beh_input_ids_padded)  # [B, max_articles, L]
    beh_attention_mask = torch.stack(beh_attention_mask_padded)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "llm_input_ids": llm_input_ids,
        "llm_attention_mask": llm_attention_mask,
        "labels": labels,
        "beh_input_ids": beh_input_ids,
        "beh_attention_mask": beh_attention_mask,
    }
