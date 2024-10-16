
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
import sacrebleu
from nltk.tokenize import word_tokenize
import math

# class Attention(nn.Module):

#     def __init__(self, hidden_size, headnum, device):
#         super(Attention, self).__init__()

#         self.hidden_size = hidden_size
#         self.headnum = headnum
#         self.device = device
#         self.query = nn.Linear(hidden_size, hidden_size * headnum).to(self.device)
#         self.key = nn.Linear(hidden_size, hidden_size * headnum).to(self.device)
#         self.value = nn.Linear(hidden_size, hidden_size * headnum).to(self.device)
#         self.fc = nn.Linear(hidden_size * headnum, hidden_size).to(self.device)
#         self.softmax = nn.Softmax(dim=-1).to(self.device)

#     def forward(self, hidden, encoder_outputs):

#         # hidden: (batchsize, 1, hidden_size)
#         # encoder_outputs: (batchsize, seq_len, hidden_size)

#         q = self.query(hidden)  # (batchsize, 1, hidden_size * headnum)
#         k = self.key(encoder_outputs)  # (batchsize, seq_len, hidden_size * headnum)
#         v = self.value(encoder_outputs)  # (batchsize, seq_len, hidden_size * headnum)

#         q = q.view(q.size(0), q.size(1), self.headnum, self.hidden_size)
#         k = k.view(k.size(0), k.size(1), self.headnum, self.hidden_size)
#         v = v.view(v.size(0), v.size(1), self.headnum, self.hidden_size)

#         q = q.permute(0, 2, 1, 3)  # (batchsize, headnum, 1, hidden_size)
#         k = k.permute(0, 2, 1, 3)  # (batchsize, headnum, seq_len, hidden_size)
#         v = v.permute(0, 2, 1, 3)  # (batchsize, headnum, seq_len, hidden_size)

#         scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_size ** 0.5)  # (batchsize, headnum, 1, seq_len)
#         attn_weights = self.softmax(scores)  # (batchsize, headnum, 1, seq_len)

#         context = torch.matmul(attn_weights, v)  # (batchsize, headnum, 1, hidden_size)
#         context = context.permute(0, 2, 1, 3).contiguous()  # (batchsize, 1, headnum, hidden_size)
#         context = context.view(context.size(0), context.size(1), -1)  # (batchsize, 1, hidden_size * headnum)

#         context = self.fc(context)  # (batchsize, 1, hidden_size)
#         return context

class Attention(nn.Module):

    def __init__(self, hidden_size, headnum, device):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.headnum = headnum
        self.device = device

    def forward(self, hidden, encoder_outputs):
        # hidden: (batchsize, 1, hidden_size)
        # encoder_outputs: (batchsize, seq_len, hidden_size)

        # Compute attention scores
        scores = torch.bmm(encoder_outputs, hidden.transpose(1, 2))  # (batchsize, seq_len, 1)
        attn_weights = F.softmax(scores, dim=1)  # (batchsize, seq_len, 1)

        # Compute context vector
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)  # (batchsize, 1, hidden_size)

        return context
    
# class Attention(nn.Module):

#     def __init__(self, hidden_size, headnum, device):
#         super(Attention, self).__init__()

#         self.hidden_size = hidden_size
#         self.headnum = headnum
#         self.device = device
#         self.fc = nn.Linear(hidden_size * 2, 1).to(self.device)

#     def forward(self, hidden, encoder_outputs):
#         # hidden: (batchsize, 1, hidden_size)
#         # encoder_outputs: (batchsize, seq_len, hidden_size)

#         # Compute attention scores using a linear layer
#         combined = torch.cat((hidden.expand(-1, encoder_outputs.size(1), -1), encoder_outputs), dim=2)
#         attn_scores = self.fc(combined)  # (batchsize, seq_len, 1)
#         attn_weights = F.softmax(attn_scores, dim=1)  # (batchsize, seq_len, 1)

#         # Compute context vector
#         context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)  # (batchsize, 1, hidden_size)

#         return context


class LSTMModel(nn.Module): 

    """
    LSTMModel: A class to define the LSTM model
    """

    def __init__(self, input_size, hidden_size, attention_module, device, skipgram):
        super(LSTMModel, self).__init__()

        self.embedding_size = input_size    # The embedding dimension
        self.hidden_size = hidden_size
        self.device = device    # The device to run the model on
        self.en_vocab = skipgram.en_vocab_dict
        self.jp_embedding_model = skipgram.jp_embedding_model
        self.en_embedding_model = skipgram.en_embedding_model

        self.encoder = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size,
                            batch_first=True).to(self.device)
        self.decoder = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size,
                            batch_first=True).to(self.device)
        
        self.attention = attention_module

        self.fc_out = nn.Linear(self.hidden_size, len(self.en_vocab)).to(self.device)
        self.fc_att_out = nn.Linear(self.hidden_size + self.embedding_size, self.embedding_size).to(self.device)

    def forward(self, jp_embed, en_embed):
        
        encoder_output, (encoder_h, encoder_c) = self.encoder(jp_embed)

        en_seqlen = en_embed.size(1) - 1
        decoder_hidden = encoder_h
        decoder_c = encoder_c
        decoder_input = en_embed[:, 0].unsqueeze(1)

        output = []
        for t in range(en_seqlen):
            decoder_output, (decoder_hidden, decoder_c) = self.decoder(decoder_input, (decoder_hidden, decoder_c))
            context_vec = self.attention(decoder_output, encoder_output)
            output.append(self.fc_out(decoder_output.squeeze(1)))
            
            if t + 1 == en_seqlen:
                break 
            concat_input = torch.cat((context_vec, en_embed[:, t+1].unsqueeze(1)), dim=-1)
            decoder_input = self.fc_att_out(concat_input)

        
        output = torch.stack(output, dim=0).transpose(0, 1)

        return output
    
    def inference(self, jp_embed):
        encoder_output, (encoder_h, encoder_c) = self.encoder(jp_embed)

        english_startindex = self.en_vocab["<START>"]
        english_startindex = torch.tensor([[english_startindex]]).to(self.device)
        decoder_input = self.en_embedding_model.embed(english_startindex)
        decoder_hidden = encoder_h
        decoder_c = encoder_c

        output = []
        for t in range(1, 25):
            decoder_output, (decoder_hidden, decoder_c) = self.decoder(decoder_input, (decoder_hidden, decoder_c))
            context_vec = self.attention(decoder_output, encoder_output)

            cur_output = self.fc_out(decoder_output.squeeze(1))
            # get the index with the highest value in the output
            predict_index = cur_output.argmax(1).item()
            output.append(predict_index)
            if predict_index == self.en_vocab["<END>"]:
                break
            predict_embed = self.en_embedding_model.embed(torch.tensor([[predict_index]]).to(self.device))

            concat_input = torch.cat((context_vec, predict_embed), dim=-1)
            decoder_input = self.fc_att_out(concat_input)
        
        return output
    


class LSTMTrain(): 

    def __init__(self, config, en_vocab_dict, skipgram, data):
        
        self.assertConfig(config)
        self.attnmodel = Attention(config["hidden_size"], config["attention_headnum"], config["device"])
        self.lstmmodel = LSTMModel(config["embedding_dim"], config["hidden_size"],self.attnmodel , config["device"], skipgram)
        
        self.jp_embedding_model = skipgram.jp_embedding_model
        self.en_embedding_model = skipgram.en_embedding_model
        self.config = config
        self.skipgram = skipgram
        self.data = data
        self.device = config["device"]
        self.en_index2word : dict = {v: k for k, v in en_vocab_dict.items()}

    def assertConfig(self, config):
        assert "embedding_dim" in config, "<LSTMTrain>: embedding_dim not found in config"
        assert "device" in config, "<LSTMTrain>: device not found in config"
        assert "lstm_epochs" in config, "<LSTMTrain>: lstm_epochs not found in config"

    def train(self, train_data, test_data, val_data):

        wandb.init(project=self.config["wandb_project_name"], name=self.config["wandb_run_name"])
        wandb.config.update(self.config, allow_val_change=True)

        optimizer = torch.optim.Adam(self.lstmmodel.parameters(), lr=self.config["lstm_lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=self.config["lstm_lr_decay"])
        criterion = nn.CrossEntropyLoss()
        jp_sentences = [d[0] for d in self.data]
        en_sentences = [d[1] for d in self.data]
        for epoch in tqdm.tqdm(range(self.config["lstm_epochs"]), desc="Epochs"):
            batchsize = self.config["lstm_batchsize"]
            for i in tqdm.tqdm(range(0, len(jp_sentences), batchsize), desc="Batches", leave=False):
                if i + batchsize > len(jp_sentences):
                    batchsize = len(jp_sentences) - i
                jp_sentencebatch = jp_sentences[i:i+batchsize]
                en_sentencesbatch = en_sentences[i:i+batchsize]

                jp_indexbatch = self.skipgram.embeddingSentence(jp_sentencebatch, "jp").to(self.device)
                en_indexbatch = self.skipgram.embeddingSentence(en_sentencesbatch, "en").to(self.device)

                optimizer.zero_grad()

                jp_embedbatch = self.jp_embedding_model.embed(jp_indexbatch)
                en_embedbatch = self.en_embedding_model.embed(en_indexbatch)
                output = self.lstmmodel(jp_embedbatch, en_embedbatch)
                output = output.reshape(-1, output.size(-1))
                target = en_indexbatch[:, 1:].contiguous().view(-1)
                loss = criterion(output, target)
                loss.backward()
                
                optimizer.step()
                
                wandb.log({"loss": loss.item()})

            if epoch >= self.config["lstm_epochs"] // 3:
                scheduler.step()
            wandb.log({"epoch": epoch})
            
            # save models
            torch.save(self.lstmmodel.state_dict(), self.config["lstm_model_path"])
            torch.save(self.jp_embedding_model.state_dict(), self.config["jp_embedding_model_path"])
            torch.save(self.en_embedding_model.state_dict(), self.config["en_embedding_model_path"])

            # evaluate
            train_hypothesis = self.translate(train_data[0][:200])
            test_hypothesis = self.translate(test_data[0][:200])
            val_hypothesis = self.translate(val_data[0][:200])

            train_references = []
            test_references = []
            val_references = []

            for sentence in train_data[1][0][:200]: 
                train_references.append(" ".join(word_tokenize(sentence)))

            for sentence in test_data[1][0][:200]: 
                test_references.append(" ".join(word_tokenize(sentence)))

            for sentence in val_data[1][0][:200]: 
                val_references.append(" ".join(word_tokenize(sentence)))
            

            train_bleu = sacrebleu.corpus_bleu(train_hypothesis, [train_references])
            test_bleu = sacrebleu.corpus_bleu(test_hypothesis, [test_references])
            val_bleu = sacrebleu.corpus_bleu(val_hypothesis, [val_references])

            wandb.log({"train_bleu": train_bleu.score})
            wandb.log({"test_bleu": test_bleu.score})
            wandb.log({"val_bleu": val_bleu.score})

    def calPerplexity(self):

        criterion = nn.CrossEntropyLoss()
        jp_sentences = [d[0] for d in self.data]
        en_sentences = [d[1] for d in self.data]

        total_loss = 0
        total_tokens = 0

        for epoch in tqdm.tqdm(range(1), desc="Epochs"):
            batchsize = self.config["lstm_batchsize"]
            for i in tqdm.tqdm(range(0, len(jp_sentences), batchsize), desc="Batches", leave=False):
                if i + batchsize > len(jp_sentences):
                    batchsize = len(jp_sentences) - i
                jp_sentencebatch = jp_sentences[i:i+batchsize]
                en_sentencesbatch = en_sentences[i:i+batchsize]

                jp_indexbatch = self.skipgram.embeddingSentence(jp_sentencebatch, "jp").to(self.device)
                en_indexbatch = self.skipgram.embeddingSentence(en_sentencesbatch, "en").to(self.device)

                jp_embedbatch = self.jp_embedding_model.embed(jp_indexbatch)
                en_embedbatch = self.en_embedding_model.embed(en_indexbatch)
                output = self.lstmmodel(jp_embedbatch, en_embedbatch)
                output = output.reshape(-1, output.size(-1))
                target = en_indexbatch[:, 1:].contiguous().view(-1)
                loss = criterion(output, target)

                total_loss += loss.item()
                total_tokens += target.size(0)

        perplexity = math.exp(total_loss / total_tokens)
        return perplexity

    def translate(self, jp_sentences):

        # load model
        self.lstmmodel.load_state_dict(torch.load(self.config["lstm_model_path"]))
        self.skipgram.jp_embedding_model.load_state_dict(torch.load(self.config["jp_embedding_model_path"]))
        self.skipgram.en_embedding_model.load_state_dict(torch.load(self.config["en_embedding_model_path"]))

        sentence_list = []

        for jp_sentence in tqdm.tqdm(jp_sentences):
            jp_sentence = [jp_sentence]
            jp_index = self.skipgram.embeddingSentence(jp_sentence, "jp").to(self.device)
            jp_embed = self.jp_embedding_model.embed(jp_index)
            en_indexlist = self.lstmmodel.inference(jp_embed)
            # from value to key
            eb_sentence = ""
            for index in en_indexlist:
                word = self.en_index2word[index]
                if word == "<END>":
                    break
                eb_sentence += word + " "
            sentence_list.append(eb_sentence)
        return sentence_list