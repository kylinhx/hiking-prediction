from flask import Flask, request, render_template
from HPM.inference import HPM_inference
import matplotlib.pyplot as plt
import numpy as np

# 初始化app
# 设置templates和static文件夹的路径
app = Flask(__name__)

HPM_inf = HPM_inference('my_model.pth')

def plot_result(result):
    # 创建一个画布
    fig = plt.figure()
    # 创建一个子图
    ax = fig.add_subplot(1, 1, 1)
    # 绘制折线图
    ax.plot(result)
    # 保存图片
    
    fig.savefig('app/static/images/result.png')

    plt.close()
    # 读取图片
    # picture = open('static/result.png', 'rb').read()
    # return picture

@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('main.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        # 检查后缀是否为 .kml文件
        if f.filename.split('.')[-1] != 'kml':
            return render_template('error.html', message='请上传.kml文件')
        else:
            result = HPM_inf.convert_kml_to_data(f)
            # 绘制result的折线图并将其传到result.html中
            # 两位小数
            end_time = result[-1] / (60*60)
            half_time = end_time / (2)
            plot_result(result)
            
            end_time = np.round(end_time, 2)
            half_time = np.round(half_time, 2)
            return render_template('result.html', end_time=end_time, half_time=half_time)

    return render_template('main.html')
if __name__ == '__main__':
    app.run(debug=True)
