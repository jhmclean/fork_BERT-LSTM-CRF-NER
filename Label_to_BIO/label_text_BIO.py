### 将原始文本和标注文本合并，得到BIO格式文件

def label_text_BIO(raw_path, label_path):
    raw_txts = []
    with open(raw_path, "r") as read_raw:
        for line in read_raw:
            if len(line.strip())==0:
                continue
            raw_txts.append(line.strip())
            
    label_txts = []
    with open(label_path, "r") as read_label:
        for line in read_label:
            if len(line.strip())==0:
                continue
            label_txts.append(line.strip())
    
    if len(raw_txts) != len(label_txts):
        raise ValueError("The length of raw.txt is not the same as the length of label.txt")
    
    BIO_list = []
    for i in range(len(raw_txts)):
        tokens = list(raw_txts[i])
        labels = label_txts[i].split(",")
        if len(tokens) != len(labels):
            raise ValueError("The number of tokens is not the same as the number of labels")
            
        BIO_list.append([tokens[j]+" "+labels[j] for j in range(len(tokens))])
    
    with open("BIO_file.txt", "w") as write_BIO:
        for sent in BIO_list:
            for token in sent:
                write_BIO.write(token+"\n")
            write_BIO.write("\n")

if __name__ == "__main__":
    label_text_BIO("raw.txt", "label.txt")