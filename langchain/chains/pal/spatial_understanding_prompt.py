# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

template = """<|im_start|>system
Generate Python3 Code to answer a question about a scene as described by a dictionay named scene_dict.
The scene is the result of multiple vision API tasks for an image
Vision tasks detect objects, captions, people, celebrities and OCR words (Optical Character Recognition)

Use box intersection to relate different entities by their location
<|im_end|>

Bounding boxes are in the form of [x1,y1,x2,y2] for left,top,right,bottom

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
    ['tv screen', [11, 100, 40, 110]],
  ],
  'faces': [
    ['a man', [341, 110, 391, 128]],
    ['a man', [1, 224, 40, 260]],
  ],
  'captions': [
    ['a man cooking on a grill', [99, 32, 188, 541]],
    ['a man playing with a dog', [120, 0, 246, 162]],
    ['a german shepard dog jumping', [301, 134, 392, 600]],
  ],
  'celebrities': [
    ['Brad Pitt', [341, 110, 391, 128]],
  ],
}}
# Question: what is Brad Pitt doing?
<|im_sep|>AI
# 1. a man playing with a dog
# 2. a man is cooking on a grill
# 3. Two faces are detected
# 4. Brad Pitt's face detected at [341, 110, 391, 128]

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

# Find the best matches for Brad Pitt in the scene 
best = {{}}
for key in ['captions', 'faces', 'objects']:
  max_intersection = 0
  for entity, entity_box in scene_dict[key]:
    intersection = get_intersection(entity_box, [341, 110, 391, 128])
    if intersection > max_intersection:
      best[key] = entity

# now go through the best matches
# and depending on the type of entity add appropriate description to the answer
def solution():
  description = ""
  for key, entity in best.items():
    if key == "faces":
      description += f"Brad Pitt matches with face of {{entity}}\n"
    elif key == "captions":
      description += f"Brad Pitt described with caption {{entity}}\n"
    elif key == "objects":
      description += f"Brad Pitt seen near object {{entity}}\n"
  return description

# the final answer is a string
answer = solution()

<|im_end|>

<|im_start|>Human
# The scene dictionary
scene_dict = {scene}
# Question: {question}
<|im_sep|>AI
"""

SPATIAL_UNDERSTANDING_PROMPT = PromptTemplate(input_variables=["scene", "question"], template=template)
