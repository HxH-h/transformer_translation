# 规范化json文件
def standardize_json(input_file, output_file):
    with open(input_file, 'r' , encoding='utf-8') as infile, open(output_file, 'w' , encoding='utf-8') as outfile:
        outfile.write('[')

        first_line = True
        for line in infile:
            if not first_line:
                outfile.write(',\n')
            else:
                first_line = False

            outfile.write(line.strip())

        outfile.write('\n]')
#standardize_json(path, './translation2019zh/translation_valid.json')

# 创建分词