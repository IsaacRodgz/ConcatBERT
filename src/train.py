from src import models
from src.utils import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
import torch
import numpy as np
import time
import sys

####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader=None):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)

    feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', hyp_params.cnn_model, pretrained=True)
    for param in feature_extractor.features.parameters():
        param.requires_grad = False

    if hyp_params.use_cuda:
        model = model.cuda()
        feature_extractor = feature_extractor.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)

    settings = {'model': model,
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
    feature_extractor = settings['feature_extractor']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    scheduler = settings['scheduler']


    def train(model, feature_extractor, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        losses = []
        correct_predictions = 0
        n_examples = hyp_params.n_train
        start_time = time.time()

        for i_batch, data_batch in enumerate(train_loader):
            input_ids = data_batch["input_ids"]
            attention_mask = data_batch["attention_mask"]
            targets = data_batch["label"]
            images = data_batch['image']

            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    targets = targets.cuda()
                    images = images.cuda()

            if images.size()[0] != input_ids.size()[0]:
                continue

            with torch.no_grad():
                feature_images = feature_extractor.features(images)
                feature_images = feature_extractor.avgpool(feature_images)
                feature_images = torch.flatten(feature_images, 1)
                feature_images = feature_extractor.classifier[0](feature_images)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                feature_images=feature_images
            )

            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            #optimizer.zero_grad()

            proc_loss += loss * hyp_params.batch_size
            proc_size += hyp_params.batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()

        return correct_predictions.double() / n_examples, np.mean(losses)

    def evaluate(model, feature_extractor, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []
        correct_predictions = 0
        n_examples = hyp_params.n_valid

        with torch.no_grad():
            for i_batch, data_batch in enumerate(loader):
                input_ids = data_batch["input_ids"]
                attention_mask = data_batch["attention_mask"]
                targets = data_batch["label"]
                images = data_batch['image']

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        input_ids = input_ids.cuda()
                        attention_mask = attention_mask.cuda()
                        targets = targets.cuda()
                        images = images.cuda()

                if images.size()[0] != input_ids.size()[0]:
                    continue

                with torch.no_grad():
                    feature_images = feature_extractor.features(images)
                    feature_images = feature_extractor.avgpool(feature_images)
                    feature_images = torch.flatten(feature_images, 1)
                    feature_images = feature_extractor.classifier[0](feature_images)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    feature_images=feature_images
                )

                _, preds = torch.max(outputs, dim=1)
                total_loss += criterion(outputs, targets).item() * hyp_params.batch_size
                correct_predictions += torch.sum(preds == targets)

                # Collect the results into dictionary
                results.append(preds)
                truths.append(targets)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return correct_predictions.double() / n_examples, avg_loss

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_acc, train_loss = train(model, feature_extractor, optimizer, criterion)
        val_acc, val_loss, = evaluate(model, feature_extractor, criterion, test=False)
        #test_loss, _, _ = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True)

        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Train Acc {:5.4f} | Valid Acc {:5.4f} | Valid Loss {:5.4f}'.format(epoch, duration, train_acc, val_acc, val_loss))
        print("-"*50)

        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    #model = load_model(hyp_params, name=hyp_params.name)
    #_, results, truths = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True)
    '''
    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)
    '''

    sys.stdout.flush()
    input('[Press Any Key to start another run]')
