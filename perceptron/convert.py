from tqdm import tqdm

def divide_DataSet(data_path, train_path, test_path):
    head = []
    lines = []
    with open(data_path) as f:
        lines = f.readlines()
    
    head = lines[0]
    trainSet = lines[1:30001]
    testSet = lines[30001:42001]
    with open(train_path, 'w') as trainf:
        trainf.write(head)
        for line in tqdm(trainSet):
            trainf.write(line)
    with open(test_path, 'w') as testf:
        testf.write(head)
        for line in tqdm(testSet):
            testf.write(line)

divide_DataSet('train.csv', 'trainSet.csv', 'testSet.csv')