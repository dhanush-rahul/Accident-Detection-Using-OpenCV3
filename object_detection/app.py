from flask import *
from crash_detection_model import main_process

app=Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
       data = main_process()
    return render_template('index.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)  