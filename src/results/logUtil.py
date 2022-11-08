import re


def log_util(read_file, write_file):
    lines = []
    with open(read_file, 'r') as f:
        lines = [_.strip() for _ in f.readlines()]
        f.close()

    pattern = re.compile('task-metrics')
    output = []
    epoch = 0
    for line in lines:
        if re.findall(pattern, line):
            epoch += 1
            arr = re.sub(r'[^0-9\.]', ' ', line).split(' ')
            arr = ' '.join(arr).split()
            output.append(arr)
    print(epoch)

    # with open(write_file, 'w') as f:
    #     for line in output:
    #         f.writelines(' '.join(line))
    #         f.writelines('\n')
    #     f.close()


if __name__ == '__main__':
    log_util('tmp/train_1.txt', 'tmp/train_1_result.txt')
