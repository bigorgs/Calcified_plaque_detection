import os
from flask import Flask, request, abort, jsonify,send_from_directory
from flask_cors import CORS
import time

from service.Cac_Unet.main_two import Plaque_detection
from service.Coronary_Resnet.main_one import Coronary_classify

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 限制上传文件大小为 16MB

ALLOWED_EXTENSIONS = {'gz'}

# http://127.0.0.1:5000/files/a.txt
@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory('../service/Cac_Unet/data/output/visual/', filename)


def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_request
def before_request():
    if request.endpoint == 'slow_view':
        flask.request.timeout = 20 * 60

@app.route('/slice', methods=['POST'])
def slice():
  if 'file' not in request.files:
    abort(400, 'No file uploaded')

  uploaded_file = request.files['file']
  if not allowed_file(uploaded_file.filename):
    abort(400, 'Invalid file type')

  if uploaded_file.content_length > app.config['MAX_CONTENT_LENGTH']:
    abort(400, 'File size exceeds the limit')


  # # 验证通过，将文件保存到本地文件系统中
  uploaded_file.save(os.path.join('../service/Coronary_Resnet/data/file', uploaded_file.filename))

  axis_start, axis_end, coronal_start, coronal_end, sagittal_start, sagittal_end = Coronary_classify('../service/Coronary_Resnet/data/file/' + uploaded_file.filename )
  print(axis_start, axis_end, coronal_start, coronal_end, sagittal_start, sagittal_end)

  return jsonify({
    'coronal_start': coronal_start,  'coronal_end': coronal_end,
    'sagittal_start': sagittal_start, 'sagittal_end': sagittal_end,
    'axis_start': axis_start, 'axis_end': axis_end
  })



@app.route('/plaque', methods=['POST'])
def plaque():
  if 'file' not in request.files:
    abort(400, 'No file uploaded')

  uploaded_file = request.files['file']
  if not allowed_file(uploaded_file.filename):
    abort(400, 'Invalid file type')

  if uploaded_file.content_length > app.config['MAX_CONTENT_LENGTH']:
    abort(400, 'File size exceeds the limit')

  # # 验证通过，将文件保存到本地文件系统中

  uploaded_file.save(os.path.join('../service/Cac_Unet/data/file/', uploaded_file.filename))

  files_list = Plaque_detection('../service/Cac_Unet/data/file/' + uploaded_file.filename)


  files_list = [os.path.basename(path) for path in files_list]

  files_list = ['http://127.0.0.1:5000/files/' + item for item in files_list]



  # return jsonify([
  #   "https://gw.alipayobjects.com/zos/antfincdn/LlvErxo8H9/photo-1503185912284-5271ff81b9a8.webp",
  #   "https://gw.alipayobjects.com/zos/antfincdn/LlvErxo8H9/photo-1503185912284-5271ff81b9a8.webp",
  #   "https://gw.alipayobjects.com/zos/antfincdn/cV16ZqzMjW/photo-1473091540282-9b846e7965e3.webp"
  # ])

  return jsonify(files_list)



if __name__ == '__main__':
  app.run()
