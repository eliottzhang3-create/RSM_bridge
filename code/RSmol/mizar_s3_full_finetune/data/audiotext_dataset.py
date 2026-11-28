"""S3 direct metadata: no prompt EOT; supervised answer EOT; no truncation."""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from contract import ANSWER_TOKENS,PROMPT_TOKENS,data_order,validate_checkpoint_tokenizer
from data.fixed20 import load_fixed20
from data.audio_paths import AudioPaths
from data.indexed_jsonl import IndexedRows

def text_pair(row):
    prompt=row["question"]+" "+" ".join(f"{letter}) {choice}" for letter,choice in zip("abcd",row["choices"]))
    if row.get("prompt_contract")!="alignkd_direct_mcq_v1" or len(row["choices"])!=4 or row["input"]!=prompt or any(tag in prompt.lower() for tag in ("<|endoftext|>","<think>","</think>","<answer>")):
        raise ValueError("invalid canonical S3 direct question/options")
    return prompt.lower(),row["answer"].strip().lower()

class AudioTextDataset(Dataset):
    def __init__(self,metadata_path,stage_root,tokenizer_path,read_policy="staged_only"):
        self.rows=IndexedRows(metadata_path);self.order=data_order(len(self.rows))
        self.stage_root=Path(stage_root).resolve();self.audio_paths=AudioPaths(self.stage_root,read_policy)
        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
        self.tokenizer.padding_side=self.tokenizer.truncation_side="right"
        validate_checkpoint_tokenizer(self.tokenizer,49152);self.pad_id=self.tokenizer.pad_token_id
    def __len__(self):return len(self.rows)
    def tokenize(self,sentence,length,*,answer=False):
        text=sentence+" <|endoftext|>" if answer else sentence
        if len(self.tokenizer.encode(text,add_special_tokens=True))>length:
            raise ValueError("S3 question or answer/EOS truncation is forbidden")
        result=self.tokenizer.encode_plus(text=text,add_special_tokens=True,max_length=length,
            padding="max_length",truncation=True,return_tensors="pt")
        if answer:
            count=int(result["attention_mask"].sum())
            if count<1 or int(result["input_ids"][0,count-1])!=0:raise ValueError("S3 answer lost supervised EOS")
        return result
    def original_item(self,original_index):
        row=self.rows[original_index];paths=self.audio_paths.resolve(row)
        if len(paths)!=1:raise ValueError("S3 requires one mapped recording")
        try:first,second=load_fixed20(paths[0])
        except Exception as exc:raise RuntimeError(f"S3 decode failed: row={original_index} source={row['s1_source']} path={paths}") from exc
        prompt,answer=text_pair(row)
        return {"audio1":first,"audio2":second,"input":self.tokenize(prompt,PROMPT_TOKENS),
                "answer":self.tokenize(answer,ANSWER_TOKENS,answer=True),"row_index":original_index,"source":row["s1_source"]}
    def __getitem__(self,index):return self.original_item(self.order[index])

def collate_fn(items):
    return {"audio1":torch.stack([x["audio1"] for x in items]),"audio2":torch.stack([x["audio2"] for x in items]),
            "input":{k:torch.cat([x["input"][k] for x in items]) for k in items[0]["input"]},
            "answer":{k:torch.cat([x["answer"][k] for x in items]) for k in items[0]["answer"]},
            "row_index":[x["row_index"] for x in items],"source":[x["source"] for x in items]}
