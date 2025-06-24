import json
from itertools import product

def main():
    print('='*50)
    learning_rate = [5e-5, 1e-5, 5e-6]
    batch_size = [32, 64, 128]

    for lr, bs in list(product(learning_rate, batch_size)):
        print(f'lr: {lr}, bs: {bs}')
        print('='*50)
        with open(f'./BEA/result/lr_{lr}_bs_{bs}_results.json', 'r') as f:
            data = json.load(f)

        five_fold_best_avg = data[0]
        best_rst = data[1]

        # for task in range(4):
        #     task_rst = five_fold_best_avg[str(task)] # list with 5 dicts
        #     avg_best = 0
        #     for d in task_rst:
        #         avg_best += d["ex_macro_f1"]
        #     avg_best = avg_best / 5
        #     print(f'Task_{task} best_5_fold_avg_macro_f1: {avg_best:.4f}')
        
        for task in range(4):
            task_rst = best_rst[str(task)] # dict
            print(f'Task_{task} best_macro_f1: {task_rst["ex_macro_f1"]} best_rst: {task_rst["eval_rst"]}')
        print('='*50)

    return

if __name__ == '__main__':
    main()