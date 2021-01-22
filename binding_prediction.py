from statistics import stdev

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from data_loading_general_old import Dataset
# from data_loading_general import Dataset
from neural_net import PcNet
from stats_and_output import *
import random

from tqdm import tqdm

# Leave the Data that you want to analyze uncommented


'''
# for the Davis Data
data_used = ["pkd", "Davis"]
data_set = Dataset("davis_folder/reduced_embeddings_file.h5",
                   "davis_folder/smile_vectors_with_cids.h5",
                   # "davis_folder/testing_lignads.h5",
                   "davis_folder/pkd_cleaned_interactions.csv",
                   "davis_folder/mapping_file.csv",
                   "pkd")
'''
# for the Kiba Data
data_used = ["kiba", "Kiba"]
data_set = Dataset("kiba_folder/reduced_embeddings_file.h5",
                   "kiba_folder/smile_vectors_with_cids.h5",
                   "kiba_folder/cleaned_interactions.csv",
                   "kiba_folder/mapping_file.csv",
                   "kiba")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#######################################################################################################################
# ################################################parameters and hyperparameters#######################################

number_of_random_draws = 5  # used to be 10
batch_sizes = list(range(20, 100))
learning_rates = [0.01, 0.001, 0.0001]
numbers_of_epochs = list(range(100, 300))

# for testing:
'''
number_of_random_draws = 1
batch_sizes = list(range(20, 25))
learning_rates = [0.01, 0.001, 0.0001]
numbers_of_epochs = [1, 2, 3]
'''

#######################################################################################################################
# ###############################################################train/test split######################################

train_main, test = train_test_split(data_set, test_size=1/6, random_state=42)
train, train1 = train_test_split(train_main, test_size=1/5, random_state=42)
train, train2 = train_test_split(train, test_size=1/4, random_state=42)
train, train3 = train_test_split(train, test_size=1/3, random_state=42)
train4, train5 = train_test_split(train, test_size=1/2, random_state=42)

#######################################################################################################################
# ########################################training and testing formulas################################################


def trainer(data_for_training, amount_of_epochs, optimi, batch_size_, tuning=True):
    all_losses = []
    for epoch_index in range(amount_of_epochs):

        running_loss = 0.0
        for i, (prot_comps, label) in enumerate(data_for_training):
            optimi.zero_grad()
            inputs = prot_comps
            label = label.unsqueeze(1).to(device)
            output = model(inputs)
            loss = criterion1(output.double().to(device), label)
            loss.backward()
            optimi.step()
            running_loss += loss.item()

            if not tuning:
                if i % batch_size_ == (batch_size_-1):  # print every n mini-batches
                    print('[%d, %5d] loss: %.7f' % (epoch_index + 1, i + 1, running_loss / batch_size_))
                    all_losses += [running_loss/batch_size_]
                    running_loss = 0.0

    if not tuning:
        print_loss(all_losses)


def tester(data_for_testing, tuning=True, bootstrap=False):
    correct = torch.tensor([0.0], dtype=torch.float64)
    total = 0.0
    all_labels = []
    all_predicted = []
    with torch.no_grad():
        for data in data_for_testing:
            prot_comp, labels = data
            outputs = model(prot_comp).double().to(device)

            outputs = outputs.tolist()
            outputs = [i[0] for i in outputs]
            outputs = torch.tensor(outputs, dtype=torch.float64)
            total += labels.size(0)
            all_labels += labels.tolist()
            all_predicted += outputs
            correct = torch.cat((correct, abs(outputs - labels)), 0)

    if bootstrap:
        return bootstrap_stats(all_predicted, all_labels, data_used)

    if not tuning:
        print_output(all_predicted, all_labels, data_used)
        print_stats(all_predicted, all_labels, data_used)

    else:
        return only_rm2(all_predicted, all_labels)


#######################################################################################################################
# ###############################################################tuning################################################


criterion1 = torch.nn.MSELoss(reduction='sum')

best_parameters_overall = [0, 0, 0]

current_best_r2m = 0


for test_train_index in tqdm(range(1)):
# for test_train_index in tqdm(range(5)):
    for optimization in tqdm(range(number_of_random_draws)):
        model = PcNet()
        batch_size = random.choice(batch_sizes)
        learning_rate = random.choice(learning_rates)
        number_of_epochs = random.choice(numbers_of_epochs)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # There is definitely a nicer way to create these train and testloaders, but it is functional
        trainloader_1 = torch.utils.data.DataLoader(dataset=(train2 + train3 + train4 + train5), batch_size=batch_size,
                                                    shuffle=False)
        trainloader_2 = torch.utils.data.DataLoader(dataset=(train1 + train3 + train4 + train5), batch_size=batch_size,
                                                    shuffle=False)
        trainloader_3 = torch.utils.data.DataLoader(dataset=(train1 + train2 + train4 + train5), batch_size=batch_size,
                                                    shuffle=False)
        trainloader_4 = torch.utils.data.DataLoader(dataset=(train1 + train2 + train3 + train5), batch_size=batch_size,
                                                    shuffle=False)
        trainloader_5 = torch.utils.data.DataLoader(dataset=(train1 + train2 + train3 + train4), batch_size=batch_size,
                                                    shuffle=False)

        testloader_1 = torch.utils.data.DataLoader(dataset=train1, batch_size=batch_size, shuffle=False)
        testloader_2 = torch.utils.data.DataLoader(dataset=train2, batch_size=batch_size, shuffle=False)
        testloader_3 = torch.utils.data.DataLoader(dataset=train3, batch_size=batch_size, shuffle=False)
        testloader_4 = torch.utils.data.DataLoader(dataset=train4, batch_size=batch_size, shuffle=False)
        testloader_5 = torch.utils.data.DataLoader(dataset=train5, batch_size=batch_size, shuffle=False)

        training_sets = [trainloader_1, trainloader_2, trainloader_3, trainloader_4, trainloader_5]
        testing_tests = [testloader_1, testloader_2, testloader_3, testloader_4, testloader_5]

        trainer(training_sets[test_train_index], number_of_epochs, optimizer, batch_size)
        performance = tester(testing_tests[test_train_index])
        print(performance)
        if performance > current_best_r2m:
            current_best_r2m = performance
            best_parameters_overall = [batch_size, learning_rate, number_of_epochs]


print('Finished Tuning')
print(current_best_r2m)
print(best_parameters_overall)


#######################################################################################################################
# ###############################################################training##############################################


model = PcNet()

trainloader = torch.utils.data.DataLoader(dataset=train_main, batch_size=best_parameters_overall[0], shuffle=False)
testloader = torch.utils.data.DataLoader(dataset=test, batch_size=best_parameters_overall[0], shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=best_parameters_overall[1])

trainer(trainloader, best_parameters_overall[2], optimizer, best_parameters_overall[0], tuning=False)
print('Finished Training')


PATH = "./model.pth"
torch.save(model.state_dict(), PATH)
model.load_state_dict(torch.load(PATH))


#######################################################################################################################
# ###############################################################testing###############################################


tester(testloader, tuning=False)

# for standard deviations
r2ms = []
AUPRs = []
CIs = []

for bootstrapping in tqdm(range(1000)):
    boot = resample(test, replace=True, n_samples=1000, random_state=1)
    bootloader = torch.utils.data.DataLoader(dataset=boot, batch_size=best_parameters_overall[0], shuffle=False)

    bootvalues = tester(bootloader, bootstrap=True)
    r2ms += [bootvalues[0]]
    AUPRs += [bootvalues[1]]
    CIs += [bootvalues[2]]


print("r2m std is: ", round(stdev(r2ms), 3))
print("AUPR std is: ", round(stdev(AUPRs), 3))
print("CIs std is: ", round(stdev(CIs), 3))

print("Best r2m was: ", current_best_r2m)
print("Best parameters were:", best_parameters_overall)

f = open("output.txt", "a")
f.write("r2m std is: "+str(round(stdev(r2ms), 3))+"\n")
f.write("AUPR std is: "+str(round(stdev(AUPRs), 3))+"\n")
f.write("CIs std is: "+str(round(stdev(CIs), 3))+"\n")
f.write("Best r2m was: "+str(current_best_r2m)+"\n")
f.write("Best parameters were:"+str(best_parameters_overall)+"\n")
f.close()
