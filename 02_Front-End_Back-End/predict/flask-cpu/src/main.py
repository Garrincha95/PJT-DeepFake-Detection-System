from flask import Flask, jsonify, request
import os
import numpy as np
import cv2
import boto3
import mysql.connector
import config
from tensorflow.keras.models import load_model
from facenet_pytorch import MTCNN
# from mtcnn import MTCNN

# config 변수 선언
service_name = 's3'
endpoint_url = 'https://kr.object.ncloudstorage.com'
region_name = 'kr-standard'
access_key = config.access_key
secret_key = config.secret_key
vid_path = '/static/'

# model load
model = load_model("IR.h5")

# app start
app = Flask(__name__)

@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    # 얼굴 좌표찾기
    def get_face_coord(img, mtcnn):
        face_coords = []
        # 얼굴 detect
        # face = detector.detect_faces(img) # just mtcnn
        boxes, probs = mtcnn.detect(img) # facenet_pytorch mtcnn

        # 얼굴 없으면 다음 프레임
        if boxes is None:
            return None

        for i, box in enumerate(boxes):
            if probs[i] < 0.8:
                continue
            # 얼굴 위치
            # x1,y1,w,h = face[0]['box']
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            x2 = int(min(x1+w, img.shape[1]))
            y2 = int(min(y1+h, img.shape[0]))
            x1 = int(max(x1, 0))
            y1 = int(max(y1, 0))

            face_coords.append((y1,y2,x1,x2))

        return face_coords

    # 이미지 자르기
    def crop_img(img, face_coord):
        y1,y2,x1,x2 = face_coord
        crop_img = img[y1:y2, x1:x2]
        crop_img = cv2.resize(crop_img, (input_shape[0], input_shape[1]))
        return crop_img

    # 영상 후처리
    def draw_face_box(image, face_coords, predicts, writer):
        for i, face_coord in enumerate(face_coords):
            y1,y2,x1,x2 = face_coord
            (color, label) = ((0, 0, 255), 'fake') if predicts[i] > 0.5 else ((0, 255, 0), 'real')
            # output_list = ['{0:.2f}'.format(float(x)) for x in output.detach().cpu().numpy()[0]]
            cv2.putText(image, str(predicts[i])+'=>'+label, (x1, y2+30),
                        font_face, font_scale, color, thickness, 2)
            # draw box over face
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        writer.write(image)

    def predict_model(vid_path, vid_name, model, start_frame=0, end_frame=None, n_frames=None, stop_frame=None):
        predictions = []
        origin_vid = vid_path + vid_name
        vid_name_no_ext = '.'.join(vid_name.split('.')[:-1])
        thumbnail_fn = vid_name_no_ext + '.jpg'
        video_fn = vid_name_no_ext + '.webm'

        # 비디오 불러오기
        reader = cv2.VideoCapture(origin_vid) # '/static/test1.mp4'
        fps = 1 if stop_frame else reader.get(cv2.CAP_PROP_FPS)
        num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        # 영상이 이상하면 return
        if not start_frame < num_frames - 1:
            return num_frames, predictions, '', 0

        # 후처리 영상
        fourcc = cv2.VideoWriter_fourcc(*'VP90')
        writer = None

        # 가져올 프레임 결정
        end_frame = end_frame if end_frame else num_frames - 1
        n_frames = n_frames if n_frames else end_frame
        if n_frames > end_frame:
            return num_frames, predictions, '', 0
        pred_frames = [int(round(x)) for x in np.linspace(start_frame, end_frame, n_frames)]

        mtcnn = MTCNN( margin=50, keep_all=False, post_process=False,thresholds=[.9,.9,.9]) # ,device='cuda:0', 

        fake_num = 0
        now_frame_num = 0
        while reader.isOpened():
            success = reader.grab()
            if not success:
                break
            if stop_frame is not None and len(predictions) >= stop_frame:
                break

            # 가져올 프레임 안에 있으면
            if now_frame_num in pred_frames:
                _, frame = reader.retrieve()
                # 얼굴 좌표 가져오기
                face_coords = get_face_coord(frame, mtcnn)
                if face_coords is None:
                    now_frame_num += 1
                    continue

                # 얼굴 여러개 검출
                predicts = []
                for face_coord in face_coords:

                    # 얼굴 crop
                    croped_img = crop_img(frame, face_coord)
                    croped_img = np.array(cv2.resize(np.array(croped_img),(160 ,160)))
                    if not predictions:
                        cv2.imwrite(vid_path + thumbnail_fn, croped_img)

                    croped_img = (croped_img.flatten() / 255.0).reshape(-1, 160, 160, 3)

                    predict = model.predict(croped_img)[0][0]
                    predicts.append(predict)
                    predictions.append(predict)
                    
                    
                    fake_num += 1 if predict > 0.5 else 0

                # writer 정의
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(vid_path + video_fn, fourcc, fps, (width, height))
                draw_face_box(frame, face_coords, predicts, writer)

            now_frame_num += 1
        if writer is not None:
            writer.release()
        if reader is not None:
            reader.release()
        return num_frames, predictions, video_fn, fake_num



    # if request.method == 'POST':
    f = request.files['file']
    ididx = request.form['ididx']
    r = int(request.form['getframe'])
    end_frame = None
    n_frames = None
    if r:
        stop_frame = r
    #저장할 경로 + 파일명
    f.save(vid_path + f.filename)
    file_name = f.filename

    input_shape = (160,160,3)
    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # try:
    # 영상예측
    num_frames, predictions, video_fn, fake_num = predict_model(vid_path, file_name, model, start_frame=0, end_frame=end_frame, n_frames=n_frames, stop_frame=stop_frame)
    count = len(predictions)

    # 얼굴이 검출 안됐으면 acc = 0.5 return
    if count == 0:
        acc = 0.5
        print('NO FACE ERROR', 'num_frames', num_frames, 'video_fn', video_fn, 'count', count, 'acc', acc)
        return jsonify(acc=acc)
    acc = sum(predictions)/(count)
    print('COMPLETE', 'num_frames', num_frames, 'video_fn', video_fn, 'count', count, 'acc', acc)


    # DB 에 영상 예측 정보 저장
    cnx = mysql.connector.connect(user='root', password=config.password,
                                host=config.host,
                                database='mydb')
    cursor = cnx.cursor()
    cursor.execute(f"INSERT INTO item (memIdx, filename, acc) values ({ididx}, '{file_name}', {acc})")

    # autoincrease 한값 가져와서 파일이름으로 만들기
    cnx.commit()
    cursor.execute("SELECT LAST_INSERT_ID()")
    rows = cursor.fetchall()
    cnx.close()

    vid_object_name = str(rows[0][0]) + '.webm'
    thumb_object_name = str(rows[0][0]) + '.jpg'
    vid_local_file_path = vid_path + video_fn
    thumb_local_file_path = vid_path + '.'.join(file_name.split('.')[:-1]) + '.jpg'

    # 후처리영상 s3서버에 전송
    s3 = boto3.client(service_name, endpoint_url=endpoint_url, aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key)
    try:
        s3.delete_object(Bucket='deepfake', Key=vid_object_name)
        s3.delete_object(Bucket='deepfake-thumb', Key=thumb_local_file_path)
    except:
        pass
    s3.upload_file(vid_local_file_path, 'deepfake', vid_object_name)
    s3.upload_file(thumb_local_file_path, 'deepfake-thumb', thumb_object_name)


    # 알수없는 에러
    # except:
    # acc = 0.5
    # count = 0
    # filename = ''
    # num_frames = 0
    # count = 0
    # fake_num = 0
    # print('UNKNOWN ERROR', 'num_frames', 0, 'video_fn', '', 'count', 0, 'acc', None)


    # 후처리영상, ,원본영상 서버에서 삭제
    try:
        os.remove(thumb_local_file_path)
        os.remove(vid_local_file_path)
        os.remove(vid_path + file_name)
    except:
        pass

    return jsonify(acc=acc, filename=rows[0][0], num_frames=num_frames, count=count, fake_num=fake_num)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
