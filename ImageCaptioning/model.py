import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]# removes the last fully connected layer of the pretrained CNN
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
## TODO: define the decoder RNN
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)        
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # create embedded word vectors for each word in a sentence
        embeds_captions = self.word_embeddings(captions)        
        all_inputs = torch.cat((features.unsqueeze(1),embeds_captions[:,:-1]),dim=1)# remove last caption        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        all_hidden, last_hidden = self.lstm(all_inputs)        
        
        all_outputs = self.fc(all_hidden)        
        return all_outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        for i in range(max_len):
            all_hidden, last_hidden = self.lstm(inputs, states)
            all_outputs = self.fc(all_hidden)
            value, index = torch.max(all_outputs, 2) #find the index of the maximum value to find the predicted word   
            #print(inputs.size(), all_outputs.size(), index.size())
            output.append(index.squeeze(1))
           
            inputs = self.word_embeddings(index)
            states = last_hidden # update state (Thanks to Udacity discussion forum)
            
        output=torch.stack(output,1)   #(sourced from github and PyTorch forums)     
        output=output[0].cpu().numpy().tolist()# to convert into a list of integers from torch tensor list
        
        return output