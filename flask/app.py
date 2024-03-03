from flask import Flask, render_template, Response
from ultralytics import YOLO
from flask_socketio import SocketIO
from camera import VideoCamera

model = YOLO('yolov8s-pose.pt')

app = Flask(__name__)
socketio = SocketIO(app)
video_stream = VideoCamera(model)


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, host='0.0.0.0', port='5002')
