from src import models
from src.utils import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
import torch
from transformers import BertModel
from transformers import BertTokenizer
import numpy as np
import time
import sys

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader=None):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)
    bert = BertModel.from_pretrained(hyp_params.bert_model)
    tokenizer = BertTokenizer.from_pretrained(hyp_params.bert_model)

    feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', hyp_params.cnn_model, pretrained=True)
    for param in feature_extractor.features.parameters():
        param.requires_grad = False

    if hyp_params.use_cuda:
        model = model.cuda()
        bert = bert.cuda()
        feature_extractor = feature_extractor.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)

    settings = {'model': model,
                'bert': bert,
                'tokenizer': tokenizer,
                'feature_extractor': feature_extractor,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    bert = settings['bert']
    tokenizer = settings['tokenizer']
    feature_extractor = settings['feature_extractor']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    scheduler = settings['scheduler']


    def train(model, bert, tokenizer, feature_extractor, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        total_loss = 0.0
        losses = []
        results = []
        truths = []
        n_examples = hyp_params.n_train
        start_time = time.time()

        for i_batch, data_batch in enumerate(train_loader):
            
            input_ids = data_batch["input_ids"]
            targets = data_batch["label"]
            images = data_batch['image']
            
            text_encoded = tokenizer.batch_encode_plus(
                input_ids,
                add_special_tokens=True,
                max_length=hyp_params.max_token_length,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    input_ids = text_encoded['input_ids'].cuda()
                    attention_mask = text_encoded['attention_mask'].cuda()
                    targets = targets.cuda()
                    images = images.cuda()

            if images.size()[0] != input_ids.size()[0]:
                continue

            with torch.no_grad():
                feature_images = feature_extractor.features(images)
                feature_images = feature_extractor.avgpool(feature_images)
                feature_images = torch.flatten(feature_images, 1)
                feature_images = feature_extractor.classifier[0](feature_images)
            
            last_hidden, pooled_output = bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            outputs = model(
                last_hidden=last_hidden,
                pooled_output=pooled_output,
                feature_images=feature_images
            )
    
            if hyp_params.dataset == 'meme_dataset':
                _, preds = torch.max(outputs, dim=1)
            else:
                preds = outputs
                
            preds_round = (preds > 0.5).float()
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            #optimizer.zero_grad()
            
            total_loss += loss.item() * hyp_params.batch_size
            results.append(preds)
            truths.append(targets)

            proc_loss += loss * hyp_params.batch_size
            proc_size += hyp_params.batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                train_acc, train_f1 = metrics(preds_round, targets)
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f} | Train Acc {:5.4f} | Train f1-score {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss, train_acc, train_f1))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
        avg_loss = total_loss / hyp_params.n_train
        results = torch.cat(results)
        truths = torch.cat(truths)
        return results, truths, avg_loss

    def evaluate(model, bert, tokenizer, feature_extractor, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []
        correct_predictions = 0

        with torch.no_grad():
            for i_batch, data_batch in enumerate(loader):
                input_ids = data_batch["input_ids"]
                targets = data_batch["label"]
                images = data_batch['image']
                
                text_encoded = tokenizer.batch_encode_plus(
                    input_ids,
                    add_special_tokens=True,
                    max_length=hyp_params.max_token_length,
                    return_token_type_ids=False,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        input_ids = text_encoded['input_ids'].cuda()
                        attention_mask = text_encoded['attention_mask'].cuda()
                        targets = targets.cuda()
                        images = images.cuda()

                if images.size()[0] != input_ids.size()[0]:
                    continue

                with torch.no_grad():
                    feature_images = feature_extractor.features(images)
                    feature_images = feature_extractor.avgpool(feature_images)
                    feature_images = torch.flatten(feature_images, 1)
                    feature_images = feature_extractor.classifier[0](feature_images)
                    
                last_hidden, pooled_output = bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                outputs = model(
                    last_hidden=last_hidden,
                    pooled_output=pooled_output,
                    feature_images=feature_images
                )

                if hyp_params.dataset == 'meme_dataset':
                    _, preds = torch.max(outputs, dim=1)
                else:
                    preds = outputs
                
                total_loss += criterion(outputs, targets).item() * hyp_params.batch_size
                correct_predictions += torch.sum(preds == targets)

                # Collect the results into dictionary
                results.append(preds)
                truths.append(targets)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return results, truths, avg_loss

    best_valid = 1e8
    writer = SummaryWriter('runs/'+hyp_params.model)
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_results, train_truths, train_loss = train(model, bert, tokenizer, feature_extractor, optimizer, criterion)
        results, truths, val_loss = evaluate(model, bert, tokenizer, feature_extractor, criterion, test=False)
        #if test_loader is not None:
        #    results, truths, val_loss = evaluate(model, feature_extractor, criterion, test=True)

        end = time.time()
        duration = end-start
        scheduler.step(val_loss)

        train_acc, train_f1 = metrics(train_results, train_truths)
        val_acc, val_f1 = metrics(results, truths)
        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f} | Valid Acc {:5.4f} | Valid f1-score {:5.4f}'.format(epoch, duration, train_loss, val_loss, val_acc, val_f1))
        print("-"*50)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('F1-score/train', train_f1, epoch)

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1-score/val', val_f1, epoch)

        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    if test_loader is not None:
        model = load_model(hyp_params, name=hyp_params.name)
        results, truths, val_loss = evaluate(model, bert, tokenizer, feature_extractor, criterion, test=True)
        test_acc, test_f1 = metrics(results, truths)
        
        print("\n\nTest Acc {:5.4f} | Test f1-score {:5.4f}".format(test_acc, test_f1))

    sys.stdout.flush()
    