label_dict={"click":1,"collect":-0.3,"cart_add":-0.5}
file_train_writer=open("data/example_train_v4.csv","w",encoding="utf-8")
with open("data/user_exp_action_list.data","r",encoding="utf-8") as file:
    for line in file.readlines():
        line_arr=line.strip().split("\t")
        if(len(line_arr)!=5):
            continue

        score=0
        if(line_arr[4]=="NULL"):
            score=0
        else:
            arr=line_arr[4].split("|")
            for score_key in arr:
                if(score_key in label_dict.keys()):
                    score=score+label_dict[score_key]
        label = 0
        if(line_arr[3]=="NULL"):
            label=1
        else:
            count=int(line_arr[3])
            score=score/count
            if(score>0.2):
               label=1
            elif(score > 0):
                print("skip",line.strip())
                continue

        #print(len(line_arr),line_arr[0],label)

        file_train_writer.write(str(label)+"\t" + line_arr[0] +"\t"+line_arr[1]+"\t"+line_arr[2]+ "\n")


file_train_writer.close()

