import pandas as pd
import textwrap


def process_files_full(folder="content"):
    all_texts1 = read_file(folder + '\\habr.txt')
    df1 = pd.DataFrame(all_texts1)
    df1['generated'] = 'H'
    df1.to_json(folder + '\\habr.json', index=False)
    return 0

    all_texts1 = read_file(folder + '\\atd_gen.txt')
    df1 = pd.DataFrame(all_texts1)
    df1['generated'] = 'M'
    df1.to_json(folder + '\\atd_gen.json', index=False)
    return 0

    all_texts1 = read_file(folder + '\\gen.txt')
    df1 = pd.DataFrame(all_texts1)
    df1['generated'] = 'M'
    df1.to_json(folder + '\\generated.json', index=False)

    all_texts1_2 = read_file(folder + '\\hum.txt')
    df2 = pd.DataFrame(all_texts1_2)
    df2['generated'] = 'H'
    df2.to_json(folder + '\\human.json', index=False)

    # df_result = pd.concat([df1, df2])
    # df_result = df_result.sample(frac=1)
    # #print(df_result)
    # df_result.to_excel(folder + '\\texts.xlsx', index=False)

def process_files(folder=""):
    all_texts1 = read_file(folder + '\\Конструктивные.txt')
    df1 = pd.DataFrame(all_texts1)
    df1['type'] = 'H'
    all_texts1_2 = read_file(folder + '\\конструктив+.txt')
    df1_2 = pd.DataFrame(all_texts1_2)
    df1_2['type'] = 'H'
    all_texts2 = read_file(folder + '\\Информационные.txt')
    df2 = pd.DataFrame(all_texts2)
    df2['type'] = 'H'
    all_texts2_2 = read_file(folder + '\\информатив+.txt')
    df2_2 = pd.DataFrame(all_texts2_2)
    df2_2['type'] = 'H'
    all_texts3 = read_file(folder + '\\Деструктивные.txt')
    df3 = pd.DataFrame(all_texts3)
    df3['type'] = 'H'
    all_texts3_2 = read_file(folder + '\\деструктив+.txt')
    df3_2 = pd.DataFrame(all_texts3_2)
    df3_2['type'] = 'H'

    df_result = pd.concat([df1, df2, df3, df1_2, df2_2, df3_2])
    df_result = df_result.sample(frac=1)
    #print(df_result)
    df_result.to_excel(folder + '\\texts_256.xlsx', index=False)


def read_file(file_name, split=256):
    file1 = open(file_name, 'r', encoding='utf8')
    Lines = file1.readlines()
    count = 0
    text1 = ""
    all_texts = []
    for line in Lines:
        if '****' in line:
            count = count + 1
            all_texts.append(text1)
            text1 = ''
        else:
            text1 = text1 + line.strip();
    result = all_texts
    if split > 0:
        result = []
        for text in all_texts:
            text_s = textwrap.wrap(text, split)
            result.extend(text_s)
    return result
