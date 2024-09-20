# https://github.com/serengil/deepface

from deepface import DeepFace

models = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
]

find_models = [
 'VGG-Face',
  'Facenet',
  'OpenFace',
  'DeepFace',
  'DeepID',
  'Dlib',
  'Ensemble'
]

# retinaface/mtcnn X VGG-Face = sometimes poor
# retinaface/mtcnn X Facenet = fek
# retinaface/mtcnn X Facenet512 gives only one?
# retinaface/mtcnn X OpenFace = fek
# retinaface X DeepFace gives only one?  / mtcnn X DeepFace = fek
# retinaface/mtcnn X DeepID = fek
# retinaface/mtcnn X ArcFace gives only several?
# retinaface/mtcnn X Dlib = very poor
# retinaface/mtcnn X SFace = fek
# retinaface/mtcnn X Ensemble = fek

# nice: cosine X retinaface X [Facenet512 / DeepFace / ArcFace / ]  ? [Facenet / OpenFace / DeepID / SFace / Ensemble]
# I would prefer ArcFace (for variants), but it has cosine dists much greater than Facenet512 and DeepFace

# retinaface / mtcnn
# DeepFace.analyze(img_path ="./people/index3.jpg", detector_backend='dlib')['dominant_emotion']
# df = DeepFace.verify(img1_path="ray.jpg", img2_path="./people/index9.jpg", model_name=models[0])
# df = DeepFace.find(img_path="ray.jpg", db_path="./people",
#                    detector_backend='retinaface', model_name='Ensemble', distance_metric='cosine')
"""
from PIL import Image

num_key_frames = 8

with Image.open('cut_0_10.gif') as im:
    for i in range(num_key_frames):
        im.seek(im.n_frames // num_key_frames * i)
        im.save('cut_0_10_{0}.png'.format(i))

for j in range(num_key_frames):
    im = 'cut_0_10_{0}.png'.format(j)
    df = DeepFace.find(img_path=im, db_path="./people")
"""