#task-start
from flask import Flask, jsonify, request
import torch

app = Flask(__name__)
model = torch.jit.load('ner.pt')
model.eval()
index2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC'}

def process(inputs):
    # TODO
    outputs = model(inputs).detach().numpy()
    result = []
    index = 0
    for output in outputs:
        result.append([])
        start = -1
        end = -1
        label = ""
        for i in range(len(output)):# 遍历每一个元素
            if((output[i] == 0)): # 若该元素为0:'O'，则表示为不合理实体
                start = end = -1
                label = ""
            if(start != -1):    # 若已存在实体的起始位置
                if(output[start] == 1 and output[i] in [1,2]):  # 若该位置元素与起始位置一致
                    end = i
                if(output[start] == 3 and output[i] in [3,4]):  # 若该位置元素与起始位置一致
                    end = i
                if(output[start] == 1 and output[i] in [3,4]):  # 若该位置元素与起始位置不一致
                    start = end = -1
                    label = ""
                if(output[start] == 3 and output[i] in [1,2]):  # 若该位置元素与起始位置不一致
                    start = end = -1
                    label = ""
            if(output[i] in [1, 3] and start == -1):    # 若该位置元素可以作为起始位置且没有其他实体的起始位置
                start = end = i
                label = index2label[output[i]][2:]

            if(start != end):   # 存在实体  # 有问题,会重复传入实体
                result[index].append({"end": end, "label": label, "start": start})

        index = index + 1

    return result



@app.route('/ner', methods=['POST'])
def ner():

    data = request.get_json()
    inputs = data['inputs']
    outputs = process(inputs)
    return jsonify(outputs)


if __name__ == '__main__':
    app.run(debug=True)
#task-end