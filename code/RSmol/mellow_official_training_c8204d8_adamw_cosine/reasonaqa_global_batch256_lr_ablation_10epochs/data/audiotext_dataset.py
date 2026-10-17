import json
import random
import torch
import torchaudio
from torch.utils.data import Dataset
import os
import random
from tqdm import tqdm
from scipy.signal import fftconvolve
import numpy as np
from transformers import AutoTokenizer
from utils.audio_io import load_audio
from data.template import DETAIL, WORD, BOTH, FIRST, SECOND, EMOTION, LONGLINE, BOTH_SPK, FIRST_SPK, SECOND_SPK
from data.template import DETAILONLY, WORDONLY, LONGLINEONLY

class AudioTextDataset(Dataset):
    """Can sample data from audio-text databases
    Params:
    sampling_rate: audio sampling rate
    max_clip_len: max length (seconds) of audio clip to be sampled
    """
    def __init__(
        self,
        data_path="",
        datafiles=[''],
        sampling_rate=44100, 
        max_clip_len=10,
        tokenizer_type="gpt2",
        ip_text_len=40,
        op_text_len=300,
    ):
        self.collect_data_jsons(data_path, datafiles)
        self.ip_text_len = ip_text_len
        self.op_text_len = op_text_len

        self.sampling_rate = sampling_rate
        self.max_length = max_clip_len * sampling_rate
        self.tokenizer_type = tokenizer_type
        self.tokenizer = self._create_tokenizer(tokenizer_type)
    
    def _create_tokenizer(self, tokenizer_type):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        # if 'gpt' in tokenizer_type or:
        tokenizer.add_special_tokens({'pad_token': '!'})
        return tokenizer

    def _tokenize_text(self, sentence, max_len, padding='max_length'):
        # if 'gpt' in self.tokenizer_type:
        sentence = sentence + ' <|endoftext|>'
        tokens = self.tokenizer.encode_plus(
            text=sentence, 
            add_special_tokens=True,
            max_length=max_len, 
            padding=padding,
            truncation=True,  # Explicitly enable truncation
            return_tensors="pt"
        )
        if self.tokenizer.eos_token_id is None:
            raise ValueError('tokenizer must provide an EOS token')
        last_valid = int(tokens['attention_mask'][0].sum()) - 1
        if last_valid < 0:
            raise ValueError('tokenized text is empty')
        tokens['input_ids'][0, last_valid] = self.tokenizer.eos_token_id
        return tokens

    def collect_data_jsons(self, data_path, datafiles):
        # collect per class data json
        all_data_json = []
        self.data_path = data_path
        for i, d in enumerate(datafiles):
            with open(d, 'r') as fp:
                data_json = json.load(fp)
                # Only print from rank 0 to avoid duplicate messages
                import os
                local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
                if local_rank == 0:
                    print(f"Dataset: {d.split(os.path.sep)[-1]}, \t Examples: {str(len(data_json))}")
                all_data_json.extend(data_json)
        self.all_data_json = all_data_json
        self.exists_filepaths1 = list(set([self.all_data_json[i]["filepath1"] for i in range(len(self.all_data_json)) if self.all_data_json[i]["filepath1"] != '']))

    def __len__(self):
        return len(self.all_data_json)

    def _cut_or_randomcrop(self, waveform):
        # waveform: [1, samples]
        if waveform.shape[0] > 1:
            waveform = (waveform[0] + waveform[1]) / 2
            waveform = waveform.unsqueeze(0)
        # random crop
        if waveform.size(1) > self.max_length:
            random_idx = random.randint(0, waveform.size(1)-self.max_length)
            waveform = waveform[:, random_idx:random_idx+self.max_length]
        else:
            temp_wav = torch.zeros(1, self.max_length)
            temp_wav[:, 0:waveform.size(1)] = waveform
            waveform = temp_wav

        assert waveform.size(1) == self.max_length, \
            f"number of audio samples is {waveform.size(1)}"

        return waveform

    def _read_audio(self, index):
        file_path1, file_path2 = self.all_data_json[index]["filepath1"], self.all_data_json[index]["filepath2"]
        if self.all_data_json[index]["filepath1"] == '':
            file_path1 = random.choice(self.exists_filepaths1)

        # Match the historical LR-ablation data contract: a missing second
        # slot is filled with a randomly selected existing first-slot audio.
        if file_path2 == '':
            file_path2 = random.choice(self.exists_filepaths1)
        
        file_path1 = os.path.join(self.data_path, file_path1)
        file_path1 = file_path1.replace("/",os.path.sep).replace("\\",os.path.sep)
        file_path2 = os.path.join(self.data_path, file_path2)
        file_path2 = file_path2.replace("/",os.path.sep).replace("\\",os.path.sep)
        try:
            audio_data1, audio_rate1 = load_audio(file_path1, channels_first=True)
            audio_data2, audio_rate2 = load_audio(file_path2, channels_first=True)

            # resample audio clip
            if audio_rate1 != self.sampling_rate:
                audio_data1 = torchaudio.functional.resample(audio_data1, orig_freq=audio_rate1, new_freq=self.sampling_rate)
            if audio_rate2 != self.sampling_rate:
                audio_data2 = torchaudio.functional.resample(audio_data2, orig_freq=audio_rate2, new_freq=self.sampling_rate)
            
            audio_data1 = self._cut_or_randomcrop(audio_data1)
            audio_data2 = self._cut_or_randomcrop(audio_data2)

            return audio_data1, audio_data2, self.sampling_rate, self.sampling_rate, file_path1, file_path2
        
        except Exception as e:
            print(f'error: {e} occurs, when loading {file_path1} or {file_path2}')
            random_index = random.randint(0, len(self.all_data_json)-1)
            return self._read_audio(index=random_index)
        
    def _create_answer_input(self, index):
        input = self.all_data_json[index]['input']
        if input == "caption both audios":
            input = random.choice(BOTH)
            answer = "The audio 1 is " + self.all_data_json[index]['caption1'].lower() + ". The audio 2 is " + self.all_data_json[index]['caption2'].lower() + "."
            answer = answer.replace("..",".")
        elif input == "caption first audio":
            input = random.choice(FIRST)
            answer = "The audio 1 is " + self.all_data_json[index]['caption1'].lower() + "." 
            answer = answer.replace("..",".")
        elif input == "caption second audio":
            input = random.choice(SECOND)
            answer = "The audio 2 is " + self.all_data_json[index]['caption2'].lower() + "."
            answer = answer.replace("..",".")
        elif input == "explain the difference in few words":
            input = random.choice(WORDONLY)
            answer = self.all_data_json[index]['answer']
        elif input == "explain the difference in a sentence":
            input = random.choice(LONGLINEONLY)
            answer = self.all_data_json[index]['answer']
        elif input == "explain the difference in detail":
            input = random.choice(DETAILONLY)
            answer = self.all_data_json[index]['answer']
        elif "emo_emo_emo" in input:
            input = input.replace("emo_emo_emo", random.choice(EMOTION))
            answer = self.all_data_json[index]['answer']
        else:
            input = self.all_data_json[index]['input'].lower()
            answer = self.all_data_json[index]['answer'].lower()
            
        return answer, input
    
    def _read_text(self, index):
        answer, input = self._create_answer_input(index)

        # Keep prompt length variable.  The answer remains capped/padded so the
        # batch can expose a stable answer target tensor and mask.
        tok_answer = self._tokenize_text(answer, self.op_text_len, padding='max_length')
        tok_input = self._tokenize_text(input, self.ip_text_len, padding=False)
        return tok_answer, tok_input, answer

    def __getitem__(self, index):
        # create a audio tensor
        audio_data1, audio_data2, audio_rate1, audio_rate2, file_path1, file_path2 = self._read_audio(index)
        # audio_data1, audio_data2, audio_rate1, audio_rate2, file_path1, file_path2 = 0,0,0,0,0,0 #self._read_audio(index)
        tok_answer, tok_input, answer_text = self._read_text(index)
        
        # resample audio clip
        # if audio_rate1 != self.sampling_rate:
        #     audio_data1 = torchaudio.functional.resample(audio_data1, orig_freq=audio_rate1, new_freq=self.sampling_rate)
        # if audio_rate2 != self.sampling_rate:
        #     audio_data2 = torchaudio.functional.resample(audio_data2, orig_freq=audio_rate2, new_freq=self.sampling_rate)
        
        # # audio_data1 = audio_data1.unsqueeze(0)
        # # audio_data2 = audio_data2.unsqueeze(0)
        
        # audio_data1 = self._cut_or_randomcrop(audio_data1)
        # audio_data2 = self._cut_or_randomcrop(audio_data2)

        data_dict = {
            'waveform1': audio_data1,
            'waveform2': audio_data2,
            'answer': tok_answer,
            'answer_text': answer_text,
            'input': tok_input,
            'file_path1': file_path1,
            'file_path2': file_path2,
            'pad_token_id': self.tokenizer.pad_token_id,
        }

        return data_dict
    

def collate_fn(list_data_dict):
    pad_id = list_data_dict[0]['pad_token_id']
    if any(entry['pad_token_id'] != pad_id for entry in list_data_dict):
        raise ValueError('mixed tokenizer pad IDs in one batch')

    texts, starts = [], []
    for entry in list_data_dict:
        prompt = entry['input']
        answer = entry['answer']
        prompt_ids = prompt['input_ids'][0, prompt['attention_mask'][0].bool()]
        answer_ids = answer['input_ids'][0, answer['attention_mask'][0].bool()]
        if prompt_ids.numel() == 0 or answer_ids.numel() == 0:
            raise ValueError('prompt and answer must contain EOS')
        starts.append(prompt_ids.numel())
        texts.append(torch.cat((prompt_ids, answer_ids)))

    max_length = max(text.numel() for text in texts)
    ids = torch.full((len(texts), max_length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(texts), max_length), dtype=torch.long)
    for row, text in enumerate(texts):
        ids[row, :text.numel()] = text
        mask[row, :text.numel()] = 1

    return {
        'waveform1': torch.stack([entry['waveform1'] for entry in list_data_dict]).squeeze(1),
        'waveform2': torch.stack([entry['waveform2'] for entry in list_data_dict]).squeeze(1),
        'input': {
            'input_ids': ids,
            'attention_mask': mask,
            'answer_start': torch.tensor(starts, dtype=torch.long),
        },
        'answer': {
            'input_ids': torch.stack([entry['answer']['input_ids'][0] for entry in list_data_dict]),
            'attention_mask': torch.stack([entry['answer']['attention_mask'][0] for entry in list_data_dict]),
        },
        'file_path1': [entry['file_path1'] for entry in list_data_dict],
        'file_path2': [entry['file_path2'] for entry in list_data_dict],
        'answer_text': [entry['answer_text'] for entry in list_data_dict],
    }
