# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

template = """<|im_start|>system
Generate Python3 Code to answer a question about a scene as described by a dictionay named scene_dict.
The scene is the result of API calls for an image to multiple vision tasks
Vision tasks detect objects, captions, people, Optical Character Recognition (OCR)

Use box intersection to relate different entities by their location
<|im_end|>

Bounding boxes are in the form of [x1,y1,x2,y2]

'description' is the description of the scene
'size' is the size of the image in pixels
'captions' contain the description of some entities in the image
'objects' contain the object descriptions in the image
'faces' are the faces detected in the scene
'celebrities' contain the name of celebrities detected for their 'faces' detected by facial recognition

Generate Python3 Code to answer the Question at the end

For example:
<|im_start|>Human
# The scene dictionary
scene_dict = {{
  'description': 'a dog playing in the backyard'
  'size': {{'height': 640, 'width': 810}},
  'objects': [
    ['person', [2, 7, 120, 170]],
    ['tv screen', [2, 7, 120, 170]],
  ],
  'faces': [
    ['a man', [341, 31, 451, 129]],
    ['a man', [1, 234, 160, 456]],
  ],
  'captions': [
    ['a man cooking on a grill', [98, 32, 198, 431]],
    ['a man playing with a dog', [127, 0, 246, 162]],
    ['a german shepard dog jumping', [359, 162, 494, 325]],
  ],
  'celebrities': [
    ['Brad Pitt', [341, 31, 451, 128]],
  ],
}}
# Question: what is Brad Pitt doing?
<|im_sep|>AI
# 1. A dog is playing in the backyard
# 2. Two faces are detected
# 3. One of the faces is recognized as Brad Pitt

def get_intersection(box_a, box_b):
  # get intersection of two boxes
  a_x1, a_y1, a_x2, a_y2 = box_a
  b_x1, b_y1, b_x2, b_y2 = box_b
  max_x1 = max(a_x1, b_x1)
  max_y1 = max(a_y1, b_y1)
  min_x2 = min(a_x2, b_x2)
  min_y2 = min(a_y2, b_y2)
  inter = (min_x2 > max_x1) * (min_y2 > max_y1)
  intersection = inter * (min_x2 - max_x1) * (min_y2 - max_y1)
  return intersection

def best_match(boxes, box):
  # find the best match for box among boxes
  max_intersection = 0
  matched_box = None
  for name, box in boxes:
    intersection = get_intersection(box, [341, 31, 451, 128])
    if intersection > max_intersection:
      matched_box = face_box
  return matched_box

# find which face belongs to Brad Pitt at [341, 31, 451, 128]
faces = scene_dict['faces']
max_intersection = 0
matched_face_box = None
for face, face_box in faces:
  intersection = get_intersection(face_box, [341, 31, 451, 128])
  if intersection > max_intersection:
    matched_face_box = face_box

if matched_face_box is None:
  answer = "Did not find the answer"

for key in ['faces']:
  face_intersection = get_intersection

def solution():
  for box in 
    
answer = solution()

<|im_end|>

<|im_start|>Human
# The scene dictionary
{scene}
# Question: {question}
<|im_sep|>AI
"""

SPATIAL_UNDERSTANDING_PROMPT = PromptTemplate(input_variables=["scene", "question"], template=template)
