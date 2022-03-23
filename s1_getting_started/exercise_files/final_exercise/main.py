import argparse
import sys

import numpy as np

import torch
from torch import optim
from torch import nn

from data import mnist
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.001)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        trainset, testset = mnist()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        epochs = 50
        
        best_val_loss = np.inf

        train_losses, test_losses = [], []
        for e in range(epochs):
            running_loss = 0.0
            for images, labels in trainloader:
                optimizer.zero_grad()

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            else:
                accuracy = 0
                test_loss = 0

                with torch.no_grad():
                    model.eval()

                    for images, labels in testloader:
                        log_ps = model(images)
                        test_loss += criterion(log_ps, labels)

                        ps = torch.exp(log_ps)

                        _, top_class = ps.topk(k=1, dim=1)

                        equals = top_class == labels.view(*top_class.shape)

                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                
                    loss = test_loss / len(testloader)
                    if loss < best_val_loss:
                        torch.save(model.state_dict(), 'checkpoint.pth')
                        best_val_loss = loss

                train_losses.append(running_loss / len(trainloader))
                test_losses.append(test_loss / len(testloader))
                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss / len(trainloader)),
                        "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                        "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))
            
            model.train()

        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        state_dict = torch.load(args.load_model_from)
        model = MyAwesomeModel()
        model.load_state_dict(state_dict)
        _, testset = mnist()
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

        model.eval()

        criterion = nn.NLLLoss()

        loss = 0.0
        accuracy = 0.0

        for images, labels in testloader:
            log_ps = model(images)
            loss += criterion(log_ps, labels)
            ps = torch.exp(log_ps)

            _, top_class = ps.topk(k=1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        print(f'Loss: {loss / len(testloader)}')
        print(f'Accuracy: {accuracy / len(testloader)}')

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    