import os
import csv

def append_dict_to_csv(dict, filename):
    # CSV 파일이 존재하지 않을 경우, 파일을 생성하고 딕셔너리 데이터를 추가함
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dict.keys())
            writer.writeheader()
            writer.writerow(dict)
    # CSV 파일이 존재할 경우, append 모드로 열고 딕셔너리 데이터를 추가함
    else:
        with open(filename, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=dict.keys())
            writer.writerow(dict)
