import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm


from dataset import En_to_Ur_Dataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]

def build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        # [UNK] = unknown 
        # [PAD] = padding 
        # [SOS] = start of sentence
        # [EOS] = end of sentence
        # min_frequency = for a word to appear in the vocab it has to present at least x times
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    train_data = load_dataset('Helsinki-NLP/opus-100', f'{config['lang_src']}-{config['lang_trgt']}', split='train')
    val_data = load_dataset('Helsinki-NLP/opus-100', f'{config['lang_src']}-{config['lang_trgt']}', split='validation')

    dataset_raw = concatenate_datasets([train_data, val_data])

    tokenizer_src = build_tokenizer(config, dataset_raw, config['lang_src'])
    tokenizer_trgt = build_tokenizer(config, dataset_raw, config['lang_trgt'])

    train_dataset = En_to_Ur_Dataset(train_data, tokenizer_src, tokenizer_trgt, config['lang_src'], config['lang_trgt'], config['seq_len'])
    val_dataset = En_to_Ur_Dataset(val_data, tokenizer_src, tokenizer_trgt, config['lang_src'], config['lang_trgt'], config['seq_len'])

    max_len_src = 0
    max_len_trgt = 0

    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        trgt_ids = tokenizer_src.encode(item['translation'][config['lang_trgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_trgt = max(max_len_trgt, len(trgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_trgt}")

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trgt

def get_model(config, vocab_src_len, vocab_trgt_len):
    model = build_transformer(vocab_src_len, vocab_trgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader , val_dataloader, tokenizer_src, tokenizer_trgt = get_dataset(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_trgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer'])
        global_step = state['global']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device) 

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch{epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.projection(decoder_output)

            target = batch['target'].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_trgt.get_vocab_size()), target.view(-1))
            batch_iterator.set_postfix({f"loss:" f"{loss.item():6.3f}"})

            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step+=1

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    config = get_config()
    train_model(config)      
