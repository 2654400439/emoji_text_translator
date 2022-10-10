from preprocess.load_data import load_data_from_file

best_file = '../dataset/submit_result.csv'
current_file = '../result/1009_1.csv'

data_best = load_data_from_file(best_file)
data_curr = load_data_from_file(current_file)

output_file_name = '../result/diff/' + best_file.split('/')[-1].split('.')[0] + '-'\
                   + current_file.split('/')[-1].split('.')[0] + '.txt'

with open(output_file_name, 'w', newline='') as f:
    if len(data_best) == len(data_curr):
        for i in range(len(data_best)):
            if data_best[i][0].split('\t')[1].replace('🐸️'[1], '').replace(' ', '').replace('跑步‍'[-1], '')\
                    == data_curr[i][0].split('\t')[1].replace('🐸️'[1], '').replace(' ', '').replace('跑步‍'[-1], ''):
                pass
            else:
                print(data_best[i][0].split('\t')[0])
                print('best:'+data_best[i][0].split('\t')[1].replace('🐸️'[1], '').replace(' ', ''))
                print('curr:'+data_curr[i][0].split('\t')[1].replace('🐸️'[1], '').replace(' ', '')+'\n')
