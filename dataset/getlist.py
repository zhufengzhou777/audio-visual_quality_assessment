import random

namelist = ['Goose', 'RedKayak', 'Stream', 'Boxing', 'Fountain', 'BigGreenRabbit', 'CrowdRun', 'PowerDig', 'Speech',
            'FootMusic', 'Town', 'Drama', 'Sparks', 'Car']

train_percent = 0.8

train_list = random.sample(namelist, int(len(namelist) * train_percent))
test_list = list(set(namelist).difference(set(train_list)))


def gettrainlist():
    with open('../datalist/train.txt', 'w') as f1:
        final_train_list = []
        for tl in train_list:
            with open('../datalist/train/{}.txt'.format(tl)) as f2:
                f_lines = f2.readlines()
                for fl in f_lines:
                    final_train_list.append(fl)
        last = final_train_list[-1]
        final_train_list.pop()
        final_train_list.append(last[:-2])
        f1.writelines(final_train_list)


def gettestlist():
    with open('../datalist/test.txt', 'w') as f1:
        final_test_list = []
        for tl in test_list:
            with open('../datalist/train/{}.txt'.format(tl)) as f2:
                f_lines = f2.readlines()
                for fl in f_lines:
                    final_test_list.append(fl)
        last = final_test_list[-1]
        final_test_list.pop()
        final_test_list.append(last[:-2])
        f1.writelines(final_test_list)


if __name__ == "__main__":
    gettrainlist()
    gettestlist()
